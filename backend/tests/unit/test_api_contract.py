"""Structural guarantees of the HTTP layer."""

from app.api.routes import agents, conversations, documents, health, providers, usage
from app.db.session import get_db


def _walk(dependant):
    for dep in dependant.dependencies:
        yield dep
        yield from _walk(dep)


def test_db_session_commits_before_the_response_is_sent():
    """Regression: with FastAPI's default "request" scope the commit of `get_db`
    runs after the response is sent, so a client could receive "201 Created"
    and get a 404 for the same row on its very next request."""
    usages = [
        (route.path, dep.scope)
        for module in (agents, conversations, documents, health, providers, usage)
        for route in module.router.routes
        for dep in _walk(route.dependant)
        if dep.call is get_db
    ]
    assert usages, "expected routes that use the database session"
    wrong = [path for path, scope in usages if scope != "function"]
    assert not wrong, f'use Depends(get_db, scope="function") in: {wrong}'
