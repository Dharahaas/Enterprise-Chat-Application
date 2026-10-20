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


def test_admin_settings_are_read_from_environment_without_repr_exposure(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("ADMIN_USERNAME", "Runtime.Admin@example.test")
    monkeypatch.setenv("ADMIN_PASSWORD", "runtime-admin-password-123")
    config = Settings(_env_file=None)

    assert config.admin_username == "Runtime.Admin@example.test"
    assert config.admin_password == "runtime-admin-password-123"
    assert "runtime-admin-password-123" not in repr(config)


def test_startup_bootstraps_configured_real_admin_and_keeps_existing_password(
    monkeypatch,
    tmp_path,
):
    import app.auth as auth

    monkeypatch.setattr(settings, "auth_database_path", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(settings, "admin_username", "  Bootstrap.Admin@example.test ")
    monkeypatch.setattr(settings, "admin_password", "bootstrap-admin-password-123")

    auth.initialize_auth_store()
    first_login = auth.authenticate("bootstrap.admin@example.test", "bootstrap-admin-password-123")
    assert first_login is not None
    assert first_login.role == "admin"
    with sqlite3.connect(settings.auth_database_path) as connection:
        first_hash = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            ("bootstrap.admin@example.test",),
        ).fetchone()[0]

    auth.initialize_auth_store()
    with sqlite3.connect(settings.auth_database_path) as connection:
        second_hash = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            ("bootstrap.admin@example.test",),
        ).fetchone()[0]
    assert second_hash == first_hash


def test_admin_bootstrap_does_not_reset_existing_password(monkeypatch, tmp_path):
    import app.auth as auth

    monkeypatch.setattr(settings, "auth_database_path", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(settings, "admin_username", "")
    monkeypatch.setattr(settings, "admin_password", "")
    auth.initialize_auth_store()
    auth.create_user("existing.admin@example.test", "saved-admin-password-123", "admin")
    monkeypatch.setattr(settings, "admin_username", "existing.admin@example.test")
    monkeypatch.setattr(settings, "admin_password", "current-admin-password-123")
    with sqlite3.connect(settings.auth_database_path) as connection:
        original_hash = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            ("existing.admin@example.test",),
        ).fetchone()[0]

    with pytest.raises(RuntimeError, match="different password"):
        auth.initialize_auth_store()

    with sqlite3.connect(settings.auth_database_path) as connection:
        unchanged_hash = connection.execute(
            "SELECT password_hash FROM users WHERE username = ?",
            ("existing.admin@example.test",),
        ).fetchone()[0]
    assert unchanged_hash == original_hash
    assert auth.authenticate("existing.admin@example.test", "saved-admin-password-123")
    assert auth.authenticate("existing.admin@example.test", "current-admin-password-123") is None


def test_admin_bootstrap_rejects_username_collision_and_invalid_configuration(
    monkeypatch,
    tmp_path,
):
    import app.auth as auth

    monkeypatch.setattr(settings, "auth_database_path", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(settings, "admin_username", "")
    monkeypatch.setattr(settings, "admin_password", "")
    auth.initialize_auth_store()
    auth.create_user("existing.employee@example.test", "employee-password-123", "employee")
    monkeypatch.setattr(settings, "admin_username", "existing.employee@example.test")
    monkeypatch.setattr(settings, "admin_password", "bootstrap-admin-password-123")

    with pytest.raises(RuntimeError, match="non-administrator"):
        auth.initialize_auth_store()
    assert auth.authenticate("existing.employee@example.test", "employee-password-123").role == "employee"

    monkeypatch.setattr(settings, "admin_username", "")
    with pytest.raises(RuntimeError, match="both be configured"):
        auth.initialize_auth_store()

    monkeypatch.setattr(settings, "admin_username", "new.admin@example.test")
    monkeypatch.setattr(settings, "admin_password", "too-short")
    with pytest.raises(RuntimeError, match="at least 12 characters"):
        auth.initialize_auth_store()


def test_invalid_optional_admin_bootstrap_does_not_block_public_demo(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(settings, "auth_database_path", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setattr(settings, "admin_username", "real.admin@example.test")
    monkeypatch.setattr(settings, "admin_password", "too-short")
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr(settings, "demo_auth_public", True)
    monkeypatch.setattr("app.main.is_llm_enabled", lambda: True)
    monkeypatch.setattr("app.rag.vectorstore.count_chunks", lambda agent: 1)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        employee_demo = client.post(
            "/api/auth/login",
            json={"role": "employee"},
        )
        admin_demo = client.post(
            "/api/auth/login",
            json={"role": "admin"},
        )

    assert employee_demo.status_code == 200
    assert employee_demo.json()["user"]["role"] == "employee"
    assert admin_demo.status_code == 200
    assert admin_demo.json()["user"]["username"] == "demo-admin@demo.invalid"


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
    assert admin_login.json()["user"] == {
        "username": "demo-admin@demo.invalid",
        "role": "admin",
    }
    admin.headers["Cookie"] = f"eka_session={admin.cookies['eka_session']}"
    admin.headers["X-CSRF-Token"] = admin_login.json()["csrf_token"]
    assert admin.get("/api/documents").status_code == 403


def test_demo_login_is_disabled_outside_development(monkeypatch):
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr("app.api.auth_routes._demo_login_enabled_for", lambda request: False)
    client = TestClient(app)

    options = client.get("/api/auth/options")
    assert options.status_code == 200
    assert options.json()["demo_login_enabled"] is False
    assert options.json()["demo_admin_login_enabled"] is False
    login = client.post(
        "/api/auth/login",
        json={"username": "anything", "password": "anything", "role": "admin"},
    )
    assert login.status_code == 401


def test_public_demo_admin_is_disabled_when_demo_auth_is_off(monkeypatch):
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr(settings, "demo_auth_enabled", False)
    monkeypatch.setattr(settings, "demo_auth_public", True)
    client = TestClient(app)

    options = client.get("/api/auth/options")
    assert options.status_code == 200
    assert options.json() == {
        "demo_login_enabled": False,
        "demo_admin_login_enabled": False,
    }
    login = client.post(
        "/api/auth/login",
        json={"username": "", "password": "", "role": "admin"},
    )
    assert login.status_code == 401


def test_public_demo_admin_is_limited_and_real_admin_remains_secure(monkeypatch):
    monkeypatch.setattr(settings, "environment", "prod")
    monkeypatch.setattr(settings, "demo_auth_enabled", True)
    monkeypatch.setattr(settings, "demo_auth_public", True)
    client = TestClient(app)

    options = client.get("/api/auth/options")
    assert options.status_code == 200
    assert options.json()["demo_login_enabled"] is True
    assert options.json()["demo_admin_login_enabled"] is True

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

    invalid_real_admin = client.post(
        "/api/auth/login",
        json={
            "username": "admin@example.test",
            "password": "incorrect-password",
            "role": "admin",
        },
    )
    assert invalid_real_admin.status_code == 401

    demo_admin = TestClient(app)
    admin = demo_admin.post(
        "/api/auth/login",
        json={"username": "", "password": "", "role": "admin"},
    )
    assert admin.status_code == 200
    assert admin.json()["user"] == {
        "username": "demo-admin@demo.invalid",
        "role": "admin",
    }
    demo_admin.headers["Cookie"] = f"eka_session={demo_admin.cookies['eka_session']}"
    demo_admin.headers["X-CSRF-Token"] = admin.json()["csrf_token"]
    assert demo_admin.get("/app/demo-admin.html").status_code == 200
    assert demo_admin.get("/app/documents.html", follow_redirects=False).status_code == 303
    assert demo_admin.get("/app/analytics.html", follow_redirects=False).status_code == 303
    for path in (
        "/api/documents",
        "/api/analytics",
        "/api/eval/summary",
        "/api/eval/latest",
        "/api/metrics",
    ):
        assert demo_admin.get(path).status_code == 403
    assert demo_admin.post("/api/seed").status_code == 403
    assert demo_admin.post(
        "/api/documents/upload",
        data={"agent": "hr"},
        files={"file": ("example.txt", b"sample", "text/plain")},
    ).status_code == 403
    assert demo_admin.delete("/api/documents/hr/example.txt").status_code == 403

    employee_cannot_open_demo_admin = TestClient(app)
    authenticate_client(employee_cannot_open_demo_admin, "employee")
    denied_demo_page = employee_cannot_open_demo_admin.get(
        "/app/demo-admin.html", follow_redirects=False
    )
    assert denied_demo_page.status_code == 303

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
    assert provisioned_admin.json()["user"]["username"] == "admin@example.test"
    real_admin = TestClient(app)
    real_admin.headers["Cookie"] = f"eka_session={client.cookies['eka_session']}"
    assert real_admin.get("/api/documents").status_code == 200


def test_limited_demo_admin_identity_cannot_be_provisioned():
    from app.auth import create_user

    with pytest.raises(ValueError, match="reserved"):
        create_user("demo-admin@demo.invalid", "a-long-test-password", "admin")


def test_demo_login_does_not_rehash_identity_on_every_login(tmp_path, monkeypatch):
    import app.auth as auth

    monkeypatch.setattr(settings, "auth_database_path", str(tmp_path / "auth.sqlite3"))
    auth.initialize_auth_store()
    auth.authenticate_demo("admin")

    def unexpected_rehash(*args, **kwargs):
        pytest.fail("existing demo identity should not be rehashed per login")

    monkeypatch.setattr(auth, "_hash_password", unexpected_rehash)
    assert auth.authenticate_demo("admin").username == "demo-admin@demo.invalid"


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
