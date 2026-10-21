"""Original-file storage on a local (Docker volume) filesystem.

Layout: <UPLOAD_DIR>/<agent_id>/<document_id>.<ext>. Keeping one directory
per agent makes "delete agent" a single directory removal. Swapping this for
S3/GCS means implementing the same four methods.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


class FileStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, agent_id: uuid.UUID, document_id: uuid.UUID, file_type: str) -> Path:
        return self.root / str(agent_id) / f"{document_id}.{file_type}"

    def save(
        self, agent_id: uuid.UUID, document_id: uuid.UUID, file_type: str, data: bytes
    ) -> Path:
        path = self.path_for(agent_id, document_id, file_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic rename: the worker never sees a partial file
        return path

    def delete_file(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)

    def delete_agent(self, agent_id: uuid.UUID) -> None:
        shutil.rmtree(self.root / str(agent_id), ignore_errors=True)
