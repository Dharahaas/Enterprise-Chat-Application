"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _default_auth_database_path() -> str:
    if os.name == "nt":
        data_root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        data_root = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    return str(data_root / "EnterpriseKnowledgeAssistant" / "auth.sqlite3")


def _read_secret_file(secret_name: str) -> str:
    secret_file = os.getenv(f"{secret_name}_FILE", "").strip()
    if not secret_file:
        return ""
    return Path(secret_file).read_text(encoding="utf-8").strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Enterprise Knowledge Assistant"
    app_version: str = "2.0.0"
    environment: Literal["dev", "staging", "prod"] = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    auth_database_path: str = _default_auth_database_path()
    session_ttl_seconds: int = 28800
    demo_auth_enabled: bool = False
    demo_auth_public: bool = False
    admin_username: str = ""
    admin_password: str = Field(
        default_factory=lambda: _read_secret_file("ADMIN_PASSWORD"),
        repr=False,
    )

    # --- Frontend CORS ---
    # Allow local development origins; add only trusted frontend origins.
    cors_origins: str = (
        "http://localhost:8000,http://localhost:5500,http://127.0.0.1:8000,"
        "http://127.0.0.1:5500"
    )

    # --- Vector DB ---
    vector_db_path: str = str(
        BACKEND_DIR / "data" / "chroma_db"
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    collection_prefix: str = "eka_"  # eka_hr, eka_it, eka_policy

    # --- RAG retrieval ---
    retrieval_top_k: int = 5
    rerank_top_k: int = 3
    chunk_size: int = 512
    chunk_overlap: int = 64

    # --- LLM ---
    # Embeddings remain local; Gemini Flash-Lite is the default hosted model.
    llm_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str = Field(
        default_factory=lambda: _read_secret_file("GEMINI_API_KEY"),
        repr=False,
    )
    gemini_model: str = "gemini-3.5-flash-lite"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512

    # --- Agents ---
    enabled_agents: str = "hr,it,policy"

    # --- Analytics (in-memory counters; replace with DB in Phase 3) ---
    enable_analytics: bool = True

    # --- Phase 3: Tracing + Eval ---
    # LangSmith tracing (optional — leave blank to disable)
    langsmith_api_key: str = ""
    langsmith_project: str = "enterprise-knowledge-assistant"
    # OpenTelemetry endpoint (optional — for Jaeger/Tempo)
    otel_exporter_otlp_endpoint: str = ""

    # --- Phase 3: Eval config ---
    eval_question_count: int = 24  # 8 questions per agent

    @model_validator(mode="after")
    def resolve_storage_paths(self) -> "Settings":
        for field_name in ("auth_database_path", "vector_db_path"):
            path = Path(getattr(self, field_name)).expanduser()
            if not path.is_absolute():
                path = BACKEND_DIR / path
            setattr(self, field_name, str(path.resolve()))
        return self


settings = Settings()


def agent_collections() -> dict[str, str]:
    """Return mapping of agent_id → Chroma collection name."""
    return {
        agent_id: f"{settings.collection_prefix}{agent_id}"
        for agent_id in settings.enabled_agents.split(",")
        if agent_id.strip()
    }


def is_openai_enabled() -> bool:
    return settings.llm_provider == "openai" and bool(
        settings.openai_api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
    )


def is_llm_enabled() -> bool:
    if settings.llm_provider == "gemini":
        return bool(
            settings.gemini_api_key.strip() or os.getenv("GEMINI_API_KEY", "").strip()
        )
    return is_openai_enabled()
