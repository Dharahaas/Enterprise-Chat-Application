"""Pytest configuration — ensure backend is on sys.path and seed DB once."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make app/ importable when running from tests/
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


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
