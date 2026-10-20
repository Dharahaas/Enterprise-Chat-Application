"""Server-side users and expiring cookie sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from time import time

from fastapi import Depends, HTTPException, Request, status

from app.config import settings

logger = logging.getLogger(__name__)

SESSION_COOKIE = "eka_session"
CSRF_HEADER = "X-CSRF-Token"
DEMO_ADMIN_USERNAME = "demo-admin@demo.invalid"
PASSWORD_HASH_ITERATIONS = 600_000
_DUMMY_PASSWORD_HASH = ""


class AdminBootstrapConfigurationError(RuntimeError):
    """Safe configuration error that should not disable the public demo flows."""


@dataclass(frozen=True)
class AuthenticatedUser:
    username: str
    role: str


def _connect() -> sqlite3.Connection:
    database_path = Path(settings.auth_database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _hash_password(
    password: str,
    salt: bytes | None = None,
    iterations: int = PASSWORD_HASH_ITERATIONS,
) -> str:
    actual_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        actual_salt,
        iterations,
    )
    return "$".join((
        "pbkdf2_sha256",
        str(iterations),
        base64.urlsafe_b64encode(actual_salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    ))


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt, expected = stored_hash.split("$", 3)
        rounds = int(iterations)
        if algorithm != "pbkdf2_sha256" or rounds < 100_000 or rounds > 2_000_000:
            return False
        actual = _hash_password(
            password,
            base64.urlsafe_b64decode(salt.encode("ascii")),
            rounds,
        ).rsplit("$", 1)[1]
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def initialize_auth_store() -> None:
    """Create auth tables and optionally bootstrap an env-configured admin."""
    global _DUMMY_PASSWORD_HASH
    _DUMMY_PASSWORD_HASH = _hash_password(secrets.token_urlsafe(32))
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('employee', 'admin')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
                csrf_token TEXT NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);
            """
        )
    configured_username = settings.admin_username.strip().casefold()
    configured_password = settings.admin_password
    if bool(configured_username) != bool(configured_password):
        raise AdminBootstrapConfigurationError(
            "ADMIN_USERNAME and ADMIN_PASSWORD must both be configured for administrator bootstrap."
        )
    if configured_username:
        if len(configured_username) > 254:
            raise AdminBootstrapConfigurationError("ADMIN_USERNAME must contain at most 254 characters.")
        if configured_username == DEMO_ADMIN_USERNAME:
            raise AdminBootstrapConfigurationError("ADMIN_USERNAME is reserved for the limited demo identity.")
        if len(configured_password) < 12:
            raise AdminBootstrapConfigurationError(
                "ADMIN_PASSWORD must be at least 12 characters to bootstrap an administrator."
            )
        with _connect() as connection:
            exists = connection.execute(
                "SELECT username, password_hash, role, active FROM users WHERE username = ?",
                (configured_username,),
            ).fetchone()
        if exists is None:
            create_user(
                configured_username,
                configured_password,
                "admin",
            )
            logger.info("Configured initial administrator was provisioned.")
        elif exists["role"] != "admin":
            raise AdminBootstrapConfigurationError(
                "ADMIN_USERNAME is already assigned to a non-administrator account; no account was changed."
            )
        elif not exists["active"]:
            raise AdminBootstrapConfigurationError(
                "The configured administrator account is inactive; no account was changed."
            )
        elif not _verify_password(configured_password, exists["password_hash"]):
            raise AdminBootstrapConfigurationError(
                "The configured administrator already exists with a different password; "
                "bootstrap does not reset existing credentials."
            )
        else:
            logger.info("Configured initial administrator is ready.")


def create_user(username: str, password: str, role: str) -> None:
    normalized_username = username.strip().casefold()
    if not normalized_username or len(normalized_username) > 254:
        raise ValueError("Username must contain 1 to 254 characters.")
    if role not in {"employee", "admin"}:
        raise ValueError("Role must be 'employee' or 'admin'.")
    if normalized_username == DEMO_ADMIN_USERNAME:
        raise ValueError("That username is reserved for the limited demo administrator.")
    if len(password) < 12:
        raise ValueError("Password must be at least 12 characters.")
    password_hash = _hash_password(password)
    try:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO users (username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                (normalized_username, password_hash, role, time()),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username already exists.") from exc


def authenticate(username: str, password: str) -> AuthenticatedUser | None:
    normalized_username = username.strip().casefold()
    with _connect() as connection:
        row = connection.execute(
            "SELECT username, password_hash, role, active FROM users WHERE username = ?",
            (normalized_username,),
        ).fetchone()
    stored_hash = row["password_hash"] if row else _DUMMY_PASSWORD_HASH
    valid = _verify_password(password, stored_hash) if stored_hash else False
    if not row or not valid or not row["active"]:
        return None
    return AuthenticatedUser(username=row["username"], role=row["role"])


def authenticate_demo(role: str) -> AuthenticatedUser:
    """Return a local demo identity for a selected role, without storing a password."""
    if role not in {"employee", "admin"}:
        raise ValueError("Role must be 'employee' or 'admin'.")
    username = DEMO_ADMIN_USERNAME if role == "admin" else "demo-employee@demo.invalid"
    with _connect() as connection:
        row = connection.execute(
            "SELECT username, role, active FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            password_hash = _hash_password(secrets.token_urlsafe(32))
            connection.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?)",
                (username, password_hash, role, time()),
            )
            row = connection.execute(
                "SELECT username, role, active FROM users WHERE username = ?",
                (username,),
            ).fetchone()
    if row is None or row["role"] != role or not row["active"]:
        raise RuntimeError("The local demo identity could not be initialized.")
    return AuthenticatedUser(username=row["username"], role=row["role"])


def create_session(user: AuthenticatedUser) -> tuple[str, str, int]:
    session_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    now = time()
    expires_at = now + max(60, settings.session_ttl_seconds)
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (token_hash, username, csrf_token, expires_at, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (_token_hash(session_token), user.username, csrf_token, expires_at, now),
        )
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    return session_token, csrf_token, int(expires_at)


def get_session_user(session_token: str | None) -> AuthenticatedUser | None:
    if not session_token:
        return None
    now = time()
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT users.username, users.role
            FROM sessions
            JOIN users ON users.username = sessions.username
            WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.active = 1
            """,
            (_token_hash(session_token), now),
        ).fetchone()
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    if not row:
        return None
    return AuthenticatedUser(username=row["username"], role=row["role"])


def invalidate_session(session_token: str | None) -> None:
    if not session_token:
        return
    with _connect() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (_token_hash(session_token),),
        )


def require_user(request: Request) -> AuthenticatedUser:
    user = get_session_user(request.cookies.get(SESSION_COOKIE))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to continue.",
        )
    return user


def require_admin(user: AuthenticatedUser = Depends(require_user)) -> AuthenticatedUser:
    if user.role != "admin" or user.username == DEMO_ADMIN_USERNAME:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Full administrator access is required.",
        )
    return user


def is_demo_admin(user: AuthenticatedUser) -> bool:
    """Identify the public demo identity, which must never reach admin APIs."""
    return user.role == "admin" and user.username == DEMO_ADMIN_USERNAME


def require_csrf(
    request: Request,
    user: AuthenticatedUser = Depends(require_user),
) -> AuthenticatedUser:
    session_token = request.cookies.get(SESSION_COOKIE)
    csrf_token = request.headers.get(CSRF_HEADER, "")
    if not session_token or not csrf_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="A valid CSRF token is required.",
        )
    with _connect() as connection:
        row = connection.execute(
            "SELECT csrf_token FROM sessions WHERE token_hash = ? AND username = ? AND expires_at > ?",
            (_token_hash(session_token), user.username, time()),
        ).fetchone()
    if not row or not hmac.compare_digest(row["csrf_token"], csrf_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="A valid CSRF token is required.",
        )
    return user


def current_csrf_token(session_token: str | None, username: str) -> str | None:
    if not session_token:
        return None
    with _connect() as connection:
        row = connection.execute(
            "SELECT csrf_token FROM sessions WHERE token_hash = ? AND username = ? AND expires_at > ?",
            (_token_hash(session_token), username, time()),
        ).fetchone()
    return row["csrf_token"] if row else None


def secure_cookie_enabled() -> bool:
    return settings.environment in {"staging", "prod"}
