"""Ingestion worker.

Polls the documents table for pending jobs, runs the ingestion pipeline and
records the outcome. Run standalone with `python -m app.workers.ingestion`
(docker compose does this) or embedded in the API process (EMBEDDED_WORKER).

Failure policy:
* `IngestionError` (bad/empty/scanned file...) -> permanent: status=failed
  with the user-facing message.
* Transient provider errors (rate limit, timeout, outage) -> job released and
  retried with back-off, up to WORKER_MAX_ATTEMPTS.
* Anything else -> failed with a generic message; details go to the logs.
"""

from __future__ import annotations

import asyncio
import signal
import time
import uuid
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.db.models import DocumentStage
from app.db.session import session_scope
from app.embeddings.base import get_embedding_provider
from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import IngestionPipeline, IngestionResult
from app.llm.errors import ProviderError
from app.repositories import chunks as chunk_repo
from app.repositories import documents as doc_repo
from app.repositories import usage as usage_repo

logger = get_logger("app.worker")


class IngestionWorker:
    def __init__(self, settings: Settings, pipeline: IngestionPipeline) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        logger.info("worker.started", extra={"concurrency": self.settings.worker_concurrency})
        await asyncio.gather(*(self._loop(i) for i in range(self.settings.worker_concurrency)))
        logger.info("worker.stopped")

    async def _loop(self, slot: int) -> None:
        while not self._stop.is_set():
            try:
                processed = await self.process_next()
            except Exception:  # never let one bad job kill the loop
                logger.exception("worker.loop_error", extra={"slot": slot})
                processed = False
            if not processed:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.settings.worker_poll_interval_seconds
                    )
                except TimeoutError:
                    pass

    async def process_next(self) -> bool:
        async with session_scope() as session:
            doc = await doc_repo.claim_next(session, self.settings.worker_stale_lock_seconds)
            if doc is None:
                return False
            job = (doc.id, doc.agent_id, Path(doc.storage_path), doc.file_type, doc.filename)
            attempts = doc.attempts

        document_id, agent_id, path, file_type, filename = job
        if attempts > self.settings.worker_max_attempts:
            await self._fail(
                document_id,
                "max_attempts_exceeded",
                f"Processing failed {attempts - 1} times; giving up. Use 'Retry' to try again.",
            )
            return True
        await self._process(document_id, agent_id, path, file_type, filename, attempts)
        return True

    async def _process(
        self,
        document_id: uuid.UUID,
        agent_id: uuid.UUID,
        path: Path,
        file_type: str,
        filename: str,
        attempt: int,
    ) -> None:
        # A single stage can be long (OCR of a large scan). Keep the job lock
        # fresh meanwhile so no other worker reclaims it as stale and processes
        # the same document twice.
        heartbeat = asyncio.create_task(self._heartbeat(document_id))
        try:
            await self._run_job(document_id, agent_id, path, file_type, filename, attempt)
        finally:
            heartbeat.cancel()

    async def _heartbeat(self, document_id: uuid.UUID) -> None:
        interval = max(self.settings.worker_stale_lock_seconds / 4, 0.2)
        while True:
            await asyncio.sleep(interval)
            try:
                async with session_scope() as session:
                    await doc_repo.heartbeat(session, document_id)
            except Exception:  # a missed beat is not fatal; the next one retries
                logger.warning(
                    "ingestion.heartbeat_failed", extra={"document_id": str(document_id)}
                )

    async def _run_job(
        self,
        document_id: uuid.UUID,
        agent_id: uuid.UUID,
        path: Path,
        file_type: str,
        filename: str,
        attempt: int,
    ) -> None:
        log = {"document_id": str(document_id), "agent_id": str(agent_id), "attempt": attempt}
        logger.info("ingestion.started", extra={**log, "doc_filename": filename})
        started = time.perf_counter()

        async def on_stage(stage: DocumentStage) -> None:
            async with session_scope() as s:
                await doc_repo.set_stage(s, document_id, stage)

        try:
            if not path.exists():
                raise IngestionError("The uploaded file is missing from storage. Upload it again.")
            result = await self.pipeline.run(path, file_type, filename, on_stage)
        except IngestionError as exc:
            logger.warning("ingestion.rejected", extra={**log, "error_code": exc.code})
            await self._fail(document_id, exc.code, exc.message)
            return
        except ProviderError as exc:
            if exc.retryable and attempt < self.settings.worker_max_attempts:
                delay = int(min(30 * 2 ** (attempt - 1), 300))
                logger.warning(
                    "ingestion.retry_scheduled",
                    extra={
                        **log,
                        "error_code": exc.code,
                        "cause": repr(exc.__cause__),
                        "retry_in_s": delay,
                    },
                )
                async with session_scope() as s:
                    await doc_repo.release(
                        s, document_id, delay, self.settings.worker_stale_lock_seconds
                    )
                return
            logger.error(
                "ingestion.embedding_failed",
                extra={**log, "error_code": exc.code, "cause": repr(exc.__cause__)},
            )
            await self._fail(document_id, exc.code, f"Embedding failed: {exc.message}")
            return
        except AppError as exc:
            logger.error("ingestion.failed", extra={**log, "error_code": exc.code})
            await self._fail(document_id, exc.code, exc.message)
            return
        except Exception:
            logger.exception("ingestion.crashed", extra=log)
            await self._fail(
                document_id,
                "processing_error",
                "An unexpected error occurred while processing this file. "
                "It may be malformed; try re-saving it, or check the worker logs.",
            )
            return

        try:
            stored = await self._store(document_id, agent_id, result)
        except Exception:
            logger.exception("ingestion.indexing_failed", extra=log)
            await self._fail(
                document_id,
                "indexing_failed",
                "The document was processed but could not be saved to the index. "
                "Use 'Retry'; if it keeps failing, check the worker logs.",
            )
            return
        if not stored:
            logger.info("ingestion.document_deleted_midway", extra=log)
            return
        logger.info(
            "ingestion.completed",
            extra={
                **log,
                "chunks": len(result.chunks),
                "pages": result.page_count,
                "ocr_pages": result.ocr_page_count,
                "embedding_tokens": result.embedding_tokens,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    async def _store(
        self, document_id: uuid.UUID, agent_id: uuid.UUID, result: IngestionResult
    ) -> bool:
        """Write chunks and mark the document ready, atomically."""
        async with session_scope() as session:
            # The user may have deleted the document while it was processing.
            if not await doc_repo.exists(session, document_id):
                return False
            await doc_repo.set_stage(session, document_id, DocumentStage.INDEXING)
            # Idempotent: never leave duplicates if the job ran more than once.
            await doc_repo.delete_chunks(session, document_id)
            await chunk_repo.insert_chunks(session, agent_id, document_id, result.chunks)
            await doc_repo.mark_ready(
                session,
                document_id,
                page_count=result.page_count,
                ocr_page_count=result.ocr_page_count,
                chunk_count=len(result.chunks),
                char_count=result.char_count,
                embedding_model=result.embedding_model,
            )
            await usage_repo.record_usage(
                session,
                agent_id=agent_id,
                kind="embedding_ingest",
                provider=result.embedding_provider,
                model=result.embedding_model,
                prompt_tokens=result.embedding_tokens,
            )
        return True

    async def _fail(self, document_id: uuid.UUID, code: str, message: str) -> None:
        async with session_scope() as session:
            await doc_repo.mark_failed(session, document_id, code, message)


def build_worker(settings: Settings | None = None) -> IngestionWorker:
    settings = settings or get_settings()
    pipeline = IngestionPipeline.from_settings(settings, get_embedding_provider())
    return IngestionWorker(settings, pipeline)


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    await chunk_repo.verify_text_search_configs(settings.fts_languages)
    worker = build_worker(settings)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, worker.stop)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(_main())
