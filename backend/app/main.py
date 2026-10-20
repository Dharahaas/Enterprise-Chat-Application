"""Enterprise Knowledge Assistant — FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.api.auth_routes import router as auth_router
from app.api.routes import router as api_router
from app.auth import (
    SESSION_COOKIE,
    AdminBootstrapConfigurationError,
    get_session_user,
    initialize_auth_store,
    is_demo_admin,
)
from app.config import is_llm_enabled, settings
from app.request_context import reset_request_id, set_request_id


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eka")


def _require_production_knowledge_base(chunk_counts: dict[str, int]) -> None:
    missing_agents = [agent for agent, count in chunk_counts.items() if count == 0]
    if missing_agents and settings.environment == "prod":
        raise RuntimeError(
            "Production knowledge base is missing indexed collections for: "
            f"{', '.join(missing_agents)}. Build and verify the sample knowledge base "
            "before starting the service."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown hooks."""
    startup_start = time.perf_counter()
    startup_issues: list[str] = []
    app.state.readiness = {"ready": False, "issues": ["startup_in_progress"]}
    startup_stage = "auth_store"
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info(
        "Startup configuration environment=%s port_configured=%s render_port=%s "
        "llm_provider=%s model=%s llm_key_configured=%s",
        settings.environment,
        bool(os.getenv("PORT")),
        os.getenv("PORT", "unset"),
        settings.llm_provider,
        settings.gemini_model if settings.llm_provider == "gemini" else settings.openai_model,
        is_llm_enabled(),
    )
    try:
        logger.info("Startup stage=auth_store status=starting")
        try:
            initialize_auth_store()
            logger.info("Startup stage=auth_store status=complete")
        except AdminBootstrapConfigurationError as exc:
            # Invalid optional real-admin bootstrap settings should not disable
            # the separately restricted employee and demo-admin experiences.
            logger.error("Admin account bootstrap was not applied: %s", exc)

        from app.rag.llm import llm_provider_name
        logger.info("Startup configured language-model provider=%s", llm_provider_name())

        from app.rag.vectorstore import count_chunks

        agents = [
            item.strip()
            for item in settings.enabled_agents.split(",")
            if item.strip()
        ]
        startup_stage = "knowledge_base_check"
        logger.info(
            "Startup stage=knowledge_base_check status=starting agents=%s",
            ",".join(agents),
        )
        chunk_counts = {agent: count_chunks(agent) for agent in agents}
        logger.info(
            "Startup knowledge_base_chunks=%s",
            ",".join(f"{agent}:{count}" for agent, count in chunk_counts.items()),
        )
        missing_agents = [agent for agent, count in chunk_counts.items() if count == 0]

        if missing_agents and settings.environment != "prod":
            startup_stage = "sample_seed"
            from app.ingestion.ingest import seed_sample_docs

            logger.info(
                "Startup stage=sample_seed status=starting missing_agents=%s",
                ",".join(missing_agents),
            )
            results = seed_sample_docs()
            seeded = len([item for item in results if "error" not in item])
            logger.info("Startup stage=sample_seed status=complete documents_seeded=%d", seeded)
            chunk_counts = {agent: count_chunks(agent) for agent in agents}
            missing_agents = [agent for agent, count in chunk_counts.items() if count == 0]

        if missing_agents:
            startup_issues.append("knowledge_base_missing:" + ",".join(missing_agents))
            logger.error(
                "Startup readiness failure stage=knowledge_base_check missing_agents=%s",
                ",".join(missing_agents),
            )
            if settings.environment == "prod":
                try:
                    _require_production_knowledge_base(chunk_counts)
                except RuntimeError as exc:
                    logger.error("Production knowledge-base validation failed: %s", exc)
        if not is_llm_enabled():
            startup_issues.append("language_model_not_configured")
            logger.error("Startup readiness failure stage=language_model reason=not_configured")
    except Exception as exc:
        startup_issues.append("startup_initialization_failed:" + startup_stage)
        logger.exception(
            "Startup initialization failed stage=%s error_type=%s",
            startup_stage,
            type(exc).__name__,
        )

    app.state.readiness = {"ready": not startup_issues, "issues": startup_issues}
    logger.info(
        "Startup complete ready=%s issues=%s elapsed_ms=%d",
        not startup_issues,
        ",".join(startup_issues) or "none",
        int((time.perf_counter() - startup_start) * 1000),
    )

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url=None if settings.environment == "prod" else "/docs",
    redoc_url=None if settings.environment == "prod" else "/redoc",
    openapi_url=None if settings.environment == "prod" else "/openapi.json",
    description=(
        "Enterprise Knowledge Assistant backend — RAG + multi-agent. "
        "Phase 2: Python + FastAPI + LangChain + Chroma."
    ),
    lifespan=lifespan,
)


# CORS — allow configured trusted origins and local dev
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if settings.environment == "dev":
    origins.extend(
        ["http://localhost:8081", "http://127.0.0.1:8081"]
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def block_private_frontend_paths(request: Request, call_next):
    """Keep backend files, local data, and dotfiles out of the static frontend mount."""
    if request.url.path.startswith("/app/"):
        components = PurePosixPath(unquote(request.url.path)).parts[2:]
        if any(
            component.startswith(".") or component in {"backend", "data"}
            for component in components
        ):
            return Response(status_code=404)
        page = components[-1] if components else ""
        protected_pages = {
            "portal.html",
            "chat.html",
            "documents.html",
            "analytics.html",
            "demo-admin.html",
        }
        if page in protected_pages:
            user = get_session_user(request.cookies.get(SESSION_COOKIE))
            if user is None:
                return RedirectResponse("/app/index.html", status_code=303)
            if page in {"documents.html", "analytics.html"} and (
                user.role != "admin" or is_demo_admin(user)
            ):
                return RedirectResponse("/app/portal.html?access=denied", status_code=303)
            if page == "demo-admin.html":
                return RedirectResponse("/app/portal.html", status_code=303)
    return await call_next(request)


@app.middleware("http")
async def log_chat_request_timing(request: Request, call_next):
    if request.url.path != "/api/chat":
        return await call_next(request)
    correlation_id = uuid.uuid4().hex[:12]
    token = set_request_id(correlation_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id
        logger.info(
            "HTTP request complete request_id=%s method=POST path=/api/chat "
            "status_code=%d http_total_ms=%d",
            correlation_id,
            response.status_code,
            int((time.perf_counter() - start) * 1000),
        )
        return response
    finally:
        reset_request_id(token)


# Mount API routes under /api
app.include_router(api_router)
app.include_router(auth_router)


# Serve the frontend (Phase 1 static files) at root
# This lets you run the whole stack from one origin during local dev.
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
async def root():
    return RedirectResponse("/app/index.html")
