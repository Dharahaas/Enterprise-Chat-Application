"""Pytest configuration — ensure backend is on sys.path and seed DB once."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make app/ importable when running from tests/
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# Absolute path of the real vector store, captured before any test runs.
# ``guard_real_vector_store`` fails any test that selects this path again.
_PROTECTED_VECTOR_PATH: Path | None = None


@pytest.fixture(autouse=True)
def guard_real_vector_store():
    """Fail fast if the real vector store becomes the selected path.

    Runs around every test: before it (a test fixture could have switched
    the path) and after it (a test's monkeypatch teardown could have
    restored the real path mid-session). Any selection of the protected
    path aborts the run instead of silently rewriting production data.
    """
    from app.config import settings

    def assert_isolated() -> None:
        if _PROTECTED_VECTOR_PATH is None:
            pytest.fail("protected vector path was never captured by the session fixture")
        selected = Path(settings.vector_db_path).resolve()
        if selected == _PROTECTED_VECTOR_PATH:
            pytest.fail(
                f"test selected the real vector store at {selected}; "
                "all vector access must stay on isolated temporary storage"
            )

    assert_isolated()
    yield
    assert_isolated()


@pytest.fixture(scope="session", autouse=True)
def isolated_vector_store(tmp_path_factory):
    """Point the whole test session at a seeded temporary vector store.

    The real ``backend/data/chroma_db`` must never be opened by the suite:
    chromadb 1.5.x re-persists index files whenever it opens an existing
    store, so even a read-only endpoint (``GET /api/documents``) rewrites
    ``chroma.sqlite3`` and the HNSW index on disk. The temporary store is
    seeded with one synthetic chunk per agent so readiness checks (which
    require a non-zero chunk count) pass without triggering the app's
    sample-document auto-seed path against real data.
    """
    import chromadb

    from app.config import agent_collections, settings

    global _PROTECTED_VECTOR_PATH
    original_path = settings.vector_db_path
    _PROTECTED_VECTOR_PATH = Path(original_path).resolve()
    vector_dir = tmp_path_factory.mktemp("eka-vectors") / "chroma_db"
    settings.vector_db_path = str(vector_dir)

    client = chromadb.PersistentClient(path=str(vector_dir))
    for agent_id, collection_name in agent_collections().items():
        collection = client.get_or_create_collection(collection_name)
        collection.add(
            ids=[f"{agent_id}-test-seed-0"],
            documents=[f"Synthetic test chunk for the {agent_id} collection."],
            embeddings=[[0.0] * 384],
            metadatas=[
                {
                    "agent": agent_id,
                    "source": f"{agent_id.upper()}_Test_Seed.txt",
                    "uploaded_at": "2026-01-01T00:00:00Z",
                }
            ],
        )

    yield str(vector_dir)

    settings.vector_db_path = original_path


@pytest.fixture(scope="session", autouse=True)
def isolated_auth_database(tmp_path_factory):
    from app.auth import create_user, initialize_auth_store
    from app.config import settings

    original_path = settings.auth_database_path
    original_admin_username = settings.admin_username
    original_admin_password = settings.admin_password
    settings.auth_database_path = str(tmp_path_factory.mktemp("eka-auth") / "auth.sqlite3")
    settings.admin_username = ""
    settings.admin_password = ""
    initialize_auth_store()
    create_user("admin@example.test", "test-admin-password-123", "admin")
    create_user("employee@example.test", "test-employee-password-123", "employee")
    yield
    settings.auth_database_path = original_path
    settings.admin_username = original_admin_username
    settings.admin_password = original_admin_password


@pytest.fixture(autouse=True)
def default_access_context():
    """Run direct pipeline calls as the explicit service principal.

    Request-path tests set their own session-derived context on top of this.
    Fail-closed behaviour is asserted by tests that install a null context.
    """
    from app.authz import reset_access_context, service_context, set_access_context

    token = set_access_context(service_context())
    try:
        yield token
    finally:
        reset_access_context(token)


def authenticate_client(client, role="admin"):
    credentials = {
        "admin": ("admin@example.test", "test-admin-password-123"),
        "employee": ("employee@example.test", "test-employee-password-123"),
    }
    username, password = credentials[role]
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    client.headers.update({"X-CSRF-Token": response.json()["csrf_token"]})
    return response.json()["user"]
