from __future__ import annotations

import hashlib
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from tests.conftest import authenticate_client


def test_login_roles_cookie_and_invalid_credentials():
    client = TestClient(app)
    failed = client.post(
        "/api/auth/login",
        json={"username": "employee@example.test", "password": "wrong-password"},
    )
    assert failed.status_code == 401
    assert "eka_session" not in failed.cookies

    user = authenticate_client(client, "employee")
    assert user == {"username": "employee@example.test", "role": "employee"}
    cookie = client.cookies.get("eka_session")
    assert cookie
    session_cookie = next(cookie for cookie in client.cookies.jar if cookie.name == "eka_session")
    assert session_cookie.has_nonstandard_attr("HttpOnly")
    assert session_cookie.get_nonstandard_attr("SameSite") == "lax"
    session = client.get("/api/auth/session")
    assert session.status_code == 200
    assert session.json()["user"] == user
    assert session.json()["csrf_token"]

    connection = sqlite3.connect(settings.auth_database_path)
    stored_hash = connection.execute(
        "SELECT password_hash FROM users WHERE username = ?",
        ("employee@example.test",),
    ).fetchone()[0]
    connection.close()
    assert stored_hash.startswith("pbkdf2_sha256$")
    assert "test-employee-password-123" not in stored_hash


def test_demo_login_uses_selected_role_and_keeps_admin_apis_protected(monkeypatch):
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr("app.api.auth_routes._demo_login_enabled_for", lambda request: True)
    monkeypatch.setattr("app.api.routes.list_documents_in_collection", lambda agent: [])

    employee = TestClient(app)
    employee_login = employee.post(
        "/api/auth/login",
        json={"username": "anything", "password": "anything", "role": "employee"},
    )
    assert employee_login.status_code == 200
    assert employee_login.json()["user"]["role"] == "employee"
    employee.headers["X-CSRF-Token"] = employee_login.json()["csrf_token"]
    assert employee.get("/api/documents").status_code == 403

    admin = TestClient(app)
    admin_login = admin.post(
        "/api/auth/login",
        json={"username": "another-dummy-name", "password": "any-value", "role": "admin"},
    )
    assert admin_login.status_code == 200
    assert admin_login.json()["user"]["role"] == "admin"
    admin.headers["X-CSRF-Token"] = admin_login.json()["csrf_token"]
    assert admin.get("/api/documents").status_code == 200


def test_demo_login_is_disabled_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr("app.api.auth_routes._demo_login_enabled_for", lambda request: False)
    client = TestClient(app)

    options = client.get("/api/auth/options")
    assert options.status_code == 200
    assert options.json()["demo_login_enabled"] is False
    login = client.post(
        "/api/auth/login",
        json={"username": "anything", "password": "anything", "role": "admin"},
    )
    assert login.status_code == 401


def test_public_demo_login_is_employee_only(monkeypatch):
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr(settings, "demo_auth_public", True)
    client = TestClient(app)

    options = client.get("/api/auth/options")
    assert options.status_code == 200
    assert options.json()["demo_login_enabled"] is True

    employee = client.post(
        "/api/auth/login",
        json={"username": "visitor", "password": "demo", "role": "employee"},
    )
    assert employee.status_code == 200
    assert employee.json()["user"]["role"] == "employee"
    assert client.get(
        "/api/documents",
        headers={"Cookie": f"eka_session={employee.cookies['eka_session']}"},
    ).status_code == 403

    admin = client.post(
        "/api/auth/login",
        json={"username": "visitor", "password": "demo", "role": "admin"},
    )
    assert admin.status_code == 401

    provisioned_admin = client.post(
        "/api/auth/login",
        json={
            "username": "admin@example.test",
            "password": "test-admin-password-123",
            "role": "admin",
        },
    )
    assert provisioned_admin.status_code == 200
    assert provisioned_admin.json()["user"]["role"] == "admin"


def test_logout_invalidates_session():
    client = TestClient(app)
    authenticate_client(client, "employee")
    response = client.post("/api/auth/logout")
    assert response.status_code == 204
    assert client.get("/api/auth/session").status_code == 401


def test_expired_session_is_rejected():
    client = TestClient(app)
    authenticate_client(client, "employee")
    token = client.cookies.get("eka_session")
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with sqlite3.connect(settings.auth_database_path) as connection:
        connection.execute(
            "UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
            (time.time() - 1, token_hash),
        )
    assert client.get("/api/auth/session").status_code == 401


def test_employee_cannot_access_admin_apis():
    client = TestClient(app)
    authenticate_client(client, "employee")
    for path in (
        "/api/documents",
        "/api/analytics",
        "/api/eval/summary",
        "/api/eval/latest",
        "/api/metrics",
    ):
        assert client.get(path).status_code == 403
    assert client.post("/api/seed").status_code == 403
    assert client.post(
        "/api/documents/upload",
        data={"agent": "hr"},
        files={"file": ("example.txt", b"sample", "text/plain")},
    ).status_code == 403
    assert client.delete("/api/documents/hr/example.txt").status_code == 403


def test_static_pages_redirect_and_private_files_are_not_served():
    client = TestClient(app)
    response = client.get("/app/chat.html", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/app/index.html"
    assert client.get("/app/backend/app/auth.py").status_code == 404

    authenticate_client(client, "employee")
    denied = client.get("/app/documents.html", follow_redirects=False)
    assert denied.status_code == 303
    assert denied.headers["location"] == "/app/portal.html?access=denied"

    admin_client = TestClient(app)
    authenticate_client(admin_client)
    assert admin_client.get("/app/documents.html").status_code == 200


def test_admin_can_access_existing_admin_routes(monkeypatch):
    monkeypatch.setattr("app.api.routes.list_documents_in_collection", lambda agent: [])
    monkeypatch.setattr("app.ingestion.ingest.seed_sample_docs", lambda force=False: [])
    client = TestClient(app)
    authenticate_client(client)
    assert client.get("/api/documents").status_code == 200
    assert client.get("/api/analytics").status_code == 200
    assert client.get("/api/eval/summary").status_code == 200
    assert client.get("/api/eval/latest").status_code == 200
    assert client.get("/api/metrics").status_code == 200
    assert client.post("/api/seed").status_code == 200


def test_chat_requires_csrf_and_preserves_agent_selection(monkeypatch):
    from app.rag.pipeline import RagResult

    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda agent, question: RagResult(
            answer=f"Response from {agent}",
            agent=agent,
            sources=[],
            latency_ms=1,
            llm_provider="test",
        ),
    )
    client = TestClient(app)
    authenticate_client(client, "employee")
    client.headers.pop("X-CSRF-Token", None)
    assert client.post("/api/chat", json={"question": "policy question", "agent": "policy"}).status_code == 403
    from tests.conftest import authenticate_client as login
    client = TestClient(app)
    login(client, "employee")
    for agent in ("hr", "it", "policy"):
        response = client.post("/api/chat", json={"question": "question", "agent": agent})
        assert response.status_code == 200
        assert response.json()["agent"] == agent
        assert response.json()["answer"] == f"Response from {agent}"


def test_admin_document_upload_and_delete_still_work(tmp_path, monkeypatch):
    from app.ingestion import ingest

    monkeypatch.setattr(settings, "vector_db_path", str(tmp_path / "vectors"))
    monkeypatch.setattr(
        ingest,
        "ingest_file",
        lambda path, agent: {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "chunks": 1,
            "uploaded_at": "2025-01-01T00:00:00+00:00",
        },
    )
    monkeypatch.setattr("app.api.routes.delete_document", lambda agent, name: 1)
    client = TestClient(app)
    authenticate_client(client)
    upload = client.post(
        "/api/documents/upload",
        data={"agent": "hr"},
        files={"file": ("policy.txt", b"sample policy", "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["document"]["agent"] == "hr"
    saved_original = (
        tmp_path
        / "uploads"
        / "hr"
        / next((tmp_path / "uploads" / "hr").iterdir()).name
        / "policy.txt"
    )
    assert saved_original.read_bytes() == b"sample policy"
    assert client.delete("/api/documents/hr/policy.txt").json()["deleted_chunks"] == 1
    assert not saved_original.exists()


def test_upload_rejects_unknown_agent_and_sanitizes_original_filename(monkeypatch, tmp_path):
    from app.ingestion import ingest

    monkeypatch.setattr(settings, "vector_db_path", str(tmp_path / "vectors"))
    monkeypatch.setattr(
        ingest,
        "ingest_file",
        lambda path, agent: {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "chunks": 1,
            "uploaded_at": "2025-01-01T00:00:00+00:00",
        },
    )
    monkeypatch.setattr("app.api.routes.delete_document", lambda agent, name: 1)
    client = TestClient(app)
    authenticate_client(client)

    invalid_agent = client.post(
        "/api/documents/upload",
        data={"agent": "../private"},
        files={"file": ("policy.txt", b"sample", "text/plain")},
    )
    assert invalid_agent.status_code == 400

    upload = client.post(
        "/api/documents/upload",
        data={"agent": "hr"},
        files={"file": ("..\\policy.txt", b"safe content", "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json()["document"]["name"] == "policy.txt"
    stored_files = list((tmp_path / "uploads" / "hr").glob("*/policy.txt"))
    assert len(stored_files) == 1


def test_admin_secure_cookie_flag_is_enabled_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "environment", "prod")
    client = TestClient(app)
    response = client.post(
        "/api/auth/login",
        json={"username": "admin@example.test", "password": "test-admin-password-123"},
    )
    assert response.status_code == 200
    assert "Secure" in response.headers["set-cookie"]


def test_config_reads_secret_from_file_without_exposing_it_in_environment(
    tmp_path,
    monkeypatch,
):
    from app.config import Settings, _read_secret_file

    secret_path = tmp_path / "gemini_api_key"
    secret_path.write_text("test-secret-value\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY_FILE", str(secret_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert _read_secret_file("GEMINI_API_KEY") == "test-secret-value"
    assert Settings(_env_file=None).gemini_api_key == "test-secret-value"


def test_config_reports_missing_secret_file(monkeypatch, tmp_path):
    from app.config import _read_secret_file

    monkeypatch.setenv(
        "GEMINI_API_KEY_FILE",
        str(tmp_path / "missing-gemini-key"),
    )

    with pytest.raises(FileNotFoundError):
        _read_secret_file("GEMINI_API_KEY")
