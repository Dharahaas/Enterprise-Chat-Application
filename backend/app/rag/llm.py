"""Language-model client configuration."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from threading import Lock
from typing import Any

from app.config import is_llm_enabled, settings

logger = logging.getLogger(__name__)
_client_lock = Lock()
_client_fingerprint: tuple[Any, ...] | None = None
_client_instance: Any = None

LLM_NOT_CONFIGURED_MESSAGE = (
    "A language model is not configured for this assistant yet. To generate "
    "grounded answers, set GEMINI_API_KEY in backend/.env "
    "and restart the backend."
)


class LLMNotConfiguredError(RuntimeError):
    """Raised when no supported generation provider is configured."""


class LLMConfigurationError(RuntimeError):
    """Raised when the configured provider cannot be initialized."""


def llm_model_name() -> str:
    """Return the configured model identifier without exposing credentials."""
    if settings.llm_provider == "gemini":
        return settings.gemini_model.strip()
    return settings.openai_model.strip()


def diagnose_llm_error(
    exc: Exception,
) -> tuple[str, int | None, str | None, str]:
    """Extract safe provider diagnostics without logging prompts or credentials."""
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(
        response, "status_code", None
    )
    if not isinstance(status, int):
        status = None

    body = getattr(exc, "body", None)
    error_body = body.get("error", body) if isinstance(body, dict) else {}
    code = error_body.get("code") or error_body.get("status") if isinstance(
        error_body, dict
    ) else None
    code = code or getattr(exc, "code", None)
    message = (
        error_body.get("message")
        if isinstance(error_body, dict)
        else None
    ) or getattr(exc, "message", None) or str(exc)
    message = str(message).replace("\r", " ").replace("\n", " ").strip()

    configured_key = settings.gemini_api_key or settings.openai_api_key
    configured_key = configured_key or os.getenv("GEMINI_API_KEY", "")
    if configured_key:
        message = message.replace(configured_key, "[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"(?i)\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", message)
    message = re.sub(r"(?i)\bsk-[0-9A-Za-z_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key|authorization|token)(\s*[:=]\s*)[^,\s;}]+",
        r"\1\2[REDACTED]",
        message,
    )[:300]

    error_code = str(code) if code is not None else ""
    error_name = type(exc).__name__.casefold()
    code_name = error_code.casefold()
    if status == 429 or "quota" in code_name or "resource_exhausted" in code_name:
        category = "quota"
    elif status in {401, 403} or any(
        value in code_name for value in ("unauthenticated", "permission_denied")
    ):
        category = "authentication"
    elif status == 404 or any(
        value in code_name for value in ("model_not_found", "not_found")
    ):
        category = "model_unavailable"
    elif (
        "connection" in error_name
        or "timeout" in error_name
        or isinstance(exc, (ConnectionError, TimeoutError))
    ):
        category = "network"
    else:
        category = "provider_error"
    return category, status, error_code or None, message


def _cached_chat_model(
    provider: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    client_factory: Any,
) -> Any:
    """Reuse one configured client and its HTTP connection pool per process."""
    global _client_fingerprint, _client_instance
    fingerprint = (
        provider,
        model,
        hashlib.sha256(api_key.encode("utf-8")).digest(),
        temperature,
        max_tokens,
        id(client_factory),
    )
    with _client_lock:
        if fingerprint == _client_fingerprint:
            return _client_instance

        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 60 if provider == "gemini" else 30,
            "max_retries": 1,
        }
        if provider == "gemini":
            kwargs["base_url"] = (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        client = client_factory(**kwargs)
        _client_instance = client
        _client_fingerprint = fingerprint
        return client


def get_llm() -> Any:
    """Return the configured reusable chat client; never substitute a template answer."""
    if not is_llm_enabled():
        raise LLMNotConfiguredError(LLM_NOT_CONFIGURED_MESSAGE)

    try:
        from langchain_openai import ChatOpenAI

        if settings.llm_provider == "gemini":
            model = settings.gemini_model.strip()
            api_key = settings.gemini_api_key.strip() or os.getenv(
                "GEMINI_API_KEY", ""
            ).strip()
            if not model:
                raise ValueError("GEMINI_MODEL must not be empty")
        else:
            model = settings.openai_model.strip()
            api_key = settings.openai_api_key.strip() or os.getenv(
                "OPENAI_API_KEY", ""
            ).strip()
        return _cached_chat_model(
            settings.llm_provider,
            model,
            api_key,
            settings.llm_temperature,
            settings.llm_max_tokens,
            ChatOpenAI,
        )
    except ValueError:
        raise
    except Exception as exc:
        category, status, code, message = diagnose_llm_error(exc)
        logger.error(
            "LLM initialization failed stage=client provider=%s model=%s "
            "category=%s http_status=%s provider_code=%s error_type=%s "
            "provider_message=%s",
            settings.llm_provider,
            llm_model_name(),
            category,
            status,
            code,
            type(exc).__name__,
            message,
        )
        raise LLMConfigurationError from exc


def llm_provider_name() -> str:
    return settings.llm_provider if is_llm_enabled() else "unconfigured"
