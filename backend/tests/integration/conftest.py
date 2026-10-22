"""Integration fixtures: a real PostgreSQL + pgvector database.

Set TEST_DATABASE_URL to a dedicated, disposable database, e.g.

    TEST_DATABASE_URL=postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot_test

Tables are created before and dropped after the test session. Tests are skipped
when the variable is unset.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

TEST_DB = os.getenv("TEST_DATABASE_URL")

if TEST_DB:
    # Must happen before app modules read settings (models size the vector column).
    os.environ["DATABASE_URL"] = TEST_DB
    os.environ["EMBEDDING_DIMENSIONS"] = "1536"
    os.environ["EMBEDDED_WORKER"] = "false"
    os.environ["FTS_LANGUAGES"] = "english,indonesian"  # independent of the local .env


def pytest_collection_modifyitems(config, items):
    if TEST_DB:
        return
    skip = pytest.mark.skip(reason="TEST_DATABASE_URL not set")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip)


@pytest.fixture
async def db(tmp_path, monkeypatch) -> AsyncIterator[None]:
    from app.core.config import get_settings
    from app.db import session as session_module
    from app.db.models import Base

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    # Engines are bound to an event loop; build a fresh one per test.
    session_module.get_engine.cache_clear()
    session_module.get_session_factory.cache_clear()
    engine = session_module.get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    session_module.get_engine.cache_clear()
    session_module.get_session_factory.cache_clear()
    get_settings.cache_clear()
