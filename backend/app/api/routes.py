"""API routes — chat, documents, analytics, health, eval, metrics."""

from __future__ import annotations

import logging
import asyncio
import hmac
import json
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.agents.router import (
    classify_conversation_intent,
    resolve_question_agent,
    conversation_reply,
    normalize_query,
)
from app.auth import (
    SESSION_COOKIE,
    current_csrf_token,
    get_session_user,
    require_admin,
    require_csrf,
)
from app.config import is_llm_enabled, is_openai_enabled, settings
from app.metrics import metrics
from app.models import (
    AnalyticsResponse,
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    HealthResponse,
    SourceChunk,
)
from app.rag.pipeline import RagResult, run_rag, run_rag_stream
from app.rag.llm import (
    LLMConfigurationError,
    LLMNotConfiguredError,
    diagnose_llm_error,
    llm_model_name,
)
from app.rag.vectorstore import count_chunks, delete_document, list_documents_in_collection
from app.request_context import current_request_id, reset_request_id, set_request_id


router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)
_ACTIVE_WS_USERS: set[str] = set()
_ACTIVE_WS_USERS_LOCK = Lock()
_LLM_LAST_SUCCESS_AT: str | None = None


# Bounded process-local analytics history; it resets on restart by design.
_QUERY_LOG: deque[dict] = deque(maxlen=1000)
_QUERY_LOG_LOCK = Lock()


# ---------- Health ----------

@router.get("/health", response_model=HealthResponse)
async def health(request: Request, response: Response) -> HealthResponse:
    from app.config import agent_collections
    from app.rag.llm import llm_provider_name

    readiness = getattr(request.app.state, "readiness", None)
    readiness_issues = list(readiness.get("issues", [])) if readiness else []
    ready = bool(readiness.get("ready", True)) if readiness else True
    if not ready:
        response.status_code = 503
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=settings.app_version,
        openai_enabled=is_openai_enabled(),
        llm_enabled=is_llm_enabled(),
        llm_provider=llm_provider_name(),
        vector_db_path=settings.vector_db_path,
        agents=settings.enabled_agents.split(","),
        collections=agent_collections(),
        readiness_issues=readiness_issues,
    )


@router.get("/llm/status")
def llm_status() -> dict[str, str | bool | None]:
    """Expose safe configuration and last-success metadata, never credentials."""
    return {
        "provider": settings.llm_provider,
        "model_id": llm_model_name(),
        "configured": is_llm_enabled(),
        "last_successful_inference_at": _LLM_LAST_SUCCESS_AT,
    }


# ---------- Chat ----------

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_csrf)])
def chat(req: ChatRequest) -> ChatResponse:
    global _LLM_LAST_SUCCESS_AT
    """Handle simple conversation locally; use RAG for knowledge questions."""
    start = perf_counter()
    question = normalize_query(req.question)
    history = [
        (message.role, message.content, message.agent)
        for message in req.conversation_history
    ]
    resolved_agent_id = resolve_question_agent(
        question, fallback=req.agent or "hr", conversation_history=history
    )
    intent = classify_conversation_intent(question)
    if intent:
        result = RagResult(
            answer=conversation_reply(intent),
            agent=resolved_agent_id,
            sources=[],
            latency_ms=int((perf_counter() - start) * 1000),
            llm_provider="none",
        )
    else:
        if req.conversation_history:
            result = run_rag(
                resolved_agent_id,
                question,
                [
                    (message.role, message.content, message.agent)
                    if message.agent
                    else (message.role, message.content)
                    for message in req.conversation_history
                ],
            )
        else:
            result = run_rag(resolved_agent_id, question)
    agent_id = result.agent
    if result.llm_provider not in {"none", "unconfigured"} and not result.error_code:
        _LLM_LAST_SUCCESS_AT = datetime.now(timezone.utc).isoformat()
    logger.info(
        "Chat request complete request_id=%s selected_agent=%s resolved_agent=%s "
        "responding_agent=%s history_messages=%d request_ms=%d rag_ms=%d",
        current_request_id(),
        req.agent or "auto",
        resolved_agent_id,
        result.agent,
        len(req.conversation_history),
        int((perf_counter() - start) * 1000),
        result.latency_ms,
    )


    # Log for analytics + metrics
    with _QUERY_LOG_LOCK:
        _QUERY_LOG.append(
            {
                "agent": agent_id,
                "latency_ms": result.latency_ms,
                "sources_count": len(result.sources),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    metrics.inc(f"chat_queries_total{{agent=\"{agent_id}\"}}")
    metrics.inc(f"chat_chunks_retrieved_total{{agent=\"{agent_id}\"}}", len(result.sources))
    metrics.observe(f"chat_latency_ms{{agent=\"{agent_id}\"}}", result.latency_ms)

    return ChatResponse(
        answer=result.answer,
        agent=result.agent,
        error_code=result.error_code,
        sources=[
            SourceChunk(
                document=s.document,
                page=s.page,
                section=s.section,
                score=round(s.score, 3),
                snippet=s.snippet,
                agent=s.agent,
            )
            for s in result.sources
        ],
        latency_ms=result.latency_ms,
        llm_provider=result.llm_provider,
        conversation_id=req.conversation_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.websocket("/chat/ws")
async def chat_websocket(websocket: WebSocket) -> None:
    """Version 1: authenticated JSON chat stream over one request per socket."""
    global _LLM_LAST_SUCCESS_AT
    origin = websocket.headers.get("origin", "")
    allowed_origins = {
        value.strip().rstrip("/")
        for value in settings.cors_origins.split(",")
        if value.strip()
    }
    origin_scheme = {"ws": "http", "wss": "https"}.get(
        websocket.url.scheme, websocket.url.scheme
    )
    allowed_origins.add(f"{origin_scheme}://{websocket.url.netloc}".rstrip("/"))
    if not origin or origin.rstrip("/") not in allowed_origins:
        await websocket.close(code=4403)
        return

    session = get_session_user(websocket.cookies.get(SESSION_COOKIE))
    if session is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    try:
        raw = await asyncio.wait_for(websocket.receive(), timeout=10)
        raw_text = raw.get("text")
        if not isinstance(raw_text, str) or len(raw_text.encode("utf-8")) > 20_000:
            await websocket.send_json({"v": 1, "type": "error", "request_id": None,
                                       "error": {"code": "invalid_message", "message": "Invalid request."}})
            await websocket.close(code=1008)
            return
        try:
            envelope = json.loads(raw_text)
            if not isinstance(envelope, dict) or envelope.get("type") != "chat":
                raise ValueError("invalid event")
            csrf_token = envelope.get("csrf_token")
            csrf_expected = current_csrf_token(
                websocket.cookies.get(SESSION_COOKIE), session.username
            )
            if not isinstance(csrf_token, str) or not csrf_expected or not hmac.compare_digest(
                csrf_token, csrf_expected
            ):
                await websocket.send_json({"v": 1, "type": "error", "request_id": None,
                                           "error": {"code": "csrf", "message": "Session validation failed. Sign in again."}})
                await websocket.close(code=1008)
                return
            request_id = envelope.get("request_id")
            if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
                request_id = uuid.uuid4().hex
            request = ChatRequest.model_validate(envelope.get("payload"))
        except Exception:
            await websocket.send_json({"v": 1, "type": "error", "request_id": None,
                                       "error": {"code": "invalid_message", "message": "Invalid chat request."}})
            await websocket.close(code=1008)
            return

        with _ACTIVE_WS_USERS_LOCK:
            if session.username in _ACTIVE_WS_USERS:
                await websocket.send_json({"v": 1, "type": "error", "request_id": request_id,
                                           "error": {"code": "busy", "message": "Another chat request is already active."}})
                await websocket.close(code=1013)
                return
            _ACTIVE_WS_USERS.add(session.username)
        request_context_token = set_request_id(request_id)

        cancel_event = Event()
        event_queue: asyncio.Queue[dict] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        disconnected = False
        worker: asyncio.Task | None = None
        cancel_listener: asyncio.Task | None = None
        timings: dict[str, int | None] = {"retrieval_ms": None, "generation_ms": None}
        try:
            question = normalize_query(request.question)
            history_for_routing = [
                (item.role, item.content, item.agent)
                for item in request.conversation_history
            ]
            resolved_agent = resolve_question_agent(
                question,
                fallback=request.agent or "hr",
                conversation_history=history_for_routing,
            )
            conversation_id = request.conversation_id or str(uuid.uuid4())
            await websocket.send_json({
                "v": 1, "type": "start", "request_id": request_id,
                "payload": {
                    "conversation_id": conversation_id,
                    "provider": settings.llm_provider,
                    "model_id": llm_model_name(),
                },
            })
            await websocket.send_json({
                "v": 1, "type": "status", "request_id": request_id,
                "payload": {"stage": "routing", "agent": resolved_agent},
            })

            async def listen_for_cancel() -> None:
                nonlocal disconnected
                try:
                    while True:
                        incoming = await websocket.receive()
                        if incoming.get("type") == "websocket.disconnect":
                            disconnected = True
                            cancel_event.set()
                            event_queue.put_nowait({"type": "disconnected"})
                            return
                        text = incoming.get("text")
                        if not isinstance(text, str) or len(text.encode("utf-8")) > 20_000:
                            event_queue.put_nowait({"type": "invalid_control"})
                            return
                        try:
                            control = json.loads(text) if isinstance(text, str) else {}
                        except ValueError:
                            control = {}
                        if control.get("type") == "cancel" and control.get("request_id") == request_id:
                            cancel_event.set()
                            event_queue.put_nowait({"type": "cancelled"})
                            return
                        if control.get("type") == "chat":
                            event_queue.put_nowait({"type": "duplicate"})
                            return
                except WebSocketDisconnect:
                    disconnected = True
                    cancel_event.set()
                    event_queue.put_nowait({"type": "disconnected"})

            async def execute() -> None:
                try:
                    if classify_conversation_intent(question):
                        intent = classify_conversation_intent(question)
                        result = RagResult(
                            answer=conversation_reply(intent), agent=resolved_agent,
                            sources=[], latency_ms=0, llm_provider="none",
                        )
                    else:
                        history = [
                            (item.role, item.content, item.agent) if item.agent
                            else (item.role, item.content)
                            for item in request.conversation_history
                        ]
                        def status_callback(stage: str, metadata: dict) -> None:
                            if metadata.get("retrieval_ms") is not None:
                                timings["retrieval_ms"] = metadata["retrieval_ms"]
                            if metadata.get("generation_ms") is not None:
                                timings["generation_ms"] = metadata["generation_ms"]
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "status", "stage": stage, **metadata,
                            })
                        def token_callback(token: str) -> None:
                            loop.call_soon_threadsafe(event_queue.put_nowait, {
                                "type": "token", "text": token,
                            })
                        result = await asyncio.to_thread(
                            run_rag_stream, resolved_agent, question, history,
                            token_callback, status_callback, cancel_event,
                        )
                    event_queue.put_nowait({"type": "worker_done", "result": result})
                except Exception as exc:
                    category, status, code, message = diagnose_llm_error(exc)
                    logger.error(
                        "WebSocket chat failed request_id=%s agent=%s category=%s "
                        "http_status=%s provider_code=%s error_type=%s provider_message=%s",
                        request_id, resolved_agent, category, status, code,
                        type(exc).__name__, message,
                    )
                    event_queue.put_nowait({"type": "worker_error", "error": exc})

            cancel_listener = asyncio.create_task(listen_for_cancel())
            worker = asyncio.create_task(execute())
            first_token_at: float | None = None
            stream_started = perf_counter()
            cancel_ack_sent = False
            while True:
                queued_event = asyncio.create_task(event_queue.get())
                done, _ = await asyncio.wait(
                    {queued_event, cancel_listener},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnected:
                    queued_event.cancel()
                    await asyncio.gather(queued_event, return_exceptions=True)
                    cancel_event.set()
                    break
                if queued_event not in done:
                    queued_event.cancel()
                    await asyncio.gather(queued_event, return_exceptions=True)
                    continue
                item = queued_event.result()
                if item["type"] == "status":
                    await websocket.send_json({
                        "v": 1, "type": "status", "request_id": request_id,
                        "payload": {key: value for key, value in item.items() if key != "type"},
                    })
                elif item["type"] == "token":
                    if first_token_at is None:
                        first_token_at = perf_counter()
                    await websocket.send_json({
                        "v": 1, "type": "token", "request_id": request_id,
                        "payload": {"text": item["text"]},
                    })
                elif item["type"] == "cancelled":
                    cancel_event.set()
                    cancel_ack_sent = True
                    await websocket.send_json({"v": 1, "type": "cancel", "request_id": request_id,
                                               "payload": {"accepted": True}})
                elif item["type"] == "duplicate":
                    await websocket.send_json({"v": 1, "type": "error", "request_id": request_id,
                                               "error": {"code": "duplicate_request", "message": "This connection already has an active request."}})
                elif item["type"] == "invalid_control":
                    await websocket.send_json({"v": 1, "type": "error", "request_id": request_id,
                                               "error": {"code": "invalid_message", "message": "Invalid stream control message."}})
                elif item["type"] == "disconnected":
                    cancel_event.set()
                    break
                elif item["type"] == "worker_error":
                    error = item["error"]
                    category = "not_configured" if isinstance(error, LLMNotConfiguredError) else (
                        "provider_unavailable" if isinstance(error, LLMConfigurationError)
                        else diagnose_llm_error(error)[0]
                    )
                    await websocket.send_json({"v": 1, "type": "error", "request_id": request_id,
                                               "error": {"code": category, "message": "The assistant could not complete this request."}})
                    break
                elif item["type"] == "worker_done":
                    result = item["result"]
                    if result.error_code and result.error_code != "cancelled":
                        await websocket.send_json({"v": 1, "type": "error", "request_id": request_id,
                                                   "error": {"code": result.error_code, "message": result.answer}})
                    elif result.error_code == "cancelled":
                        if not cancel_ack_sent:
                            await websocket.send_json({"v": 1, "type": "cancel", "request_id": request_id,
                                                       "payload": {"accepted": True}})
                    else:
                        if result.llm_provider not in {"none", "unconfigured"} and not result.error_code:
                            _LLM_LAST_SUCCESS_AT = datetime.now(timezone.utc).isoformat()
                        await websocket.send_json({
                            "v": 1, "type": "complete", "request_id": request_id,
                            "payload": {
                                "answer": result.answer, "agent": result.agent,
                                "sources": [
                                    {"document": source.document, "page": source.page,
                                     "section": source.section, "score": round(source.score, 3),
                                     "snippet": source.snippet, "agent": source.agent}
                                    for source in result.sources
                                ],
                                "conversation_id": conversation_id,
                                "provider": result.llm_provider, "model_id": llm_model_name(),
                                "time_to_first_token_ms": int((first_token_at - stream_started) * 1000) if first_token_at else None,
                                "total_ms": int((perf_counter() - stream_started) * 1000),
                                "retrieval_ms": timings["retrieval_ms"],
                                "generation_ms": timings["generation_ms"],
                            },
                        })
                    break
            if worker and not worker.done():
                cancel_event.set()
                await asyncio.wait_for(worker, timeout=65)
        except (WebSocketDisconnect, asyncio.CancelledError):
            disconnected = True
            cancel_event.set()
            if worker:
                try:
                    await asyncio.wait_for(worker, timeout=65)
                except Exception:
                    pass
        finally:
            cancel_event.set()
            if cancel_listener:
                cancel_listener.cancel()
                try:
                    await asyncio.shield(
                        asyncio.gather(cancel_listener, return_exceptions=True)
                    )
                except asyncio.CancelledError:
                    pass
            if not disconnected:
                try:
                    await websocket.close()
                except Exception:
                    pass
            with _ACTIVE_WS_USERS_LOCK:
                _ACTIVE_WS_USERS.discard(session.username)
            reset_request_id(request_context_token)
    except (asyncio.TimeoutError, WebSocketDisconnect):
        try:
            await websocket.close(code=1008)
        except Exception:
            pass

# ---------- Documents ----------

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(require_admin)],
)
def list_documents() -> DocumentListResponse:
    """List all indexed documents across all agents."""
    docs: list[DocumentInfo] = []
    by_agent: dict[str, int] = {}
    for agent_id in settings.enabled_agents.split(","):
        agent_id = agent_id.strip()
        if not agent_id:
            continue
        raw = list_documents_in_collection(agent_id)
        by_agent[agent_id] = len(raw)
        for r in raw:
            docs.append(
                DocumentInfo(
                    id=f"{agent_id}:{r['name']}",
                    name=r["name"],
                    agent=agent_id,  # type: ignore[arg-type]
                    size_bytes=r["size_bytes"],
                    chunks=r["chunks"],
                    status="indexed",  # type: ignore[arg-type]
                    uploaded_at=r["uploaded_at"],
                )
            )
    return DocumentListResponse(
        documents=docs,
        total=len(docs),
        by_agent=by_agent,
    )


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    dependencies=[Depends(require_admin), Depends(require_csrf)],
)
async def upload_document(
    agent: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> DocumentUploadResponse:
    """Upload and ingest a document into the agent's vector namespace."""
    from app.ingestion.ingest import ingest_file

    enabled_agents = {
        item.strip() for item in settings.enabled_agents.split(",") if item.strip()
    }
    if agent not in enabled_agents:
        raise HTTPException(status_code=400, detail="Unknown agent.")

    filename = Path((file.filename or "upload").replace("\\", "/")).name
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A valid filename is required.")

    upload_root = Path(settings.vector_db_path).parent / "uploads" / agent
    saved_path = upload_root / uuid.uuid4().hex / filename
    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(await file.read())

    try:
        result = await run_in_threadpool(ingest_file, saved_path, agent)
    except Exception as e:
        saved_path.unlink(missing_ok=True)
        saved_path.parent.rmdir()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    doc = DocumentInfo(
        id=f"{agent}:{result['name']}",
        name=result["name"],
        agent=agent,  # type: ignore[arg-type]
        size_bytes=result["size_bytes"],
        chunks=result["chunks"],
        status="indexed",  # type: ignore[arg-type]
        uploaded_at=result["uploaded_at"],
    )
    return DocumentUploadResponse(document=doc, message=f"Indexed {result['chunks']} chunks")


@router.delete(
    "/documents/{agent}/{doc_name}",
    dependencies=[Depends(require_admin), Depends(require_csrf)],
)
def remove_document(agent: str, doc_name: str) -> dict:
    """Delete all chunks of a document from the agent's namespace."""
    if (
        agent not in {item.strip() for item in settings.enabled_agents.split(",")}
        or Path(doc_name.replace("\\", "/")).name != doc_name
    ):
        raise HTTPException(status_code=400, detail="Invalid agent or document name.")
    deleted = delete_document(agent, doc_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    upload_root = Path(settings.vector_db_path).parent / "uploads" / agent
    if upload_root.exists():
        for upload_dir in upload_root.iterdir():
            original = upload_dir / doc_name
            if upload_dir.is_dir() and original.is_file():
                original.unlink()
                upload_dir.rmdir()
        if not any(upload_root.iterdir()):
            upload_root.rmdir()
    return {"deleted_chunks": deleted, "agent": agent, "document": doc_name}


# ---------- Analytics ----------

@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    dependencies=[Depends(require_admin)],
)
def analytics() -> AnalyticsResponse:
    """Return measured process-local activity and completed evaluation results."""
    cutoff = datetime.now(timezone.utc).timestamp() - 7 * 24 * 60 * 60
    with _QUERY_LOG_LOCK:
        query_snapshot = tuple(_QUERY_LOG)
    recent_queries = [
        query
        for query in query_snapshot
        if datetime.fromisoformat(query["timestamp"]).timestamp() >= cutoff
    ]
    # Query volume by agent
    by_agent: dict[str, int] = {a: 0 for a in settings.enabled_agents.split(",") if a.strip()}
    by_day: dict[str, int] = {}
    latencies: list[int] = []
    for q in recent_queries:
        by_agent[q["agent"]] = by_agent.get(q["agent"], 0) + 1
        latencies.append(q["latency_ms"])
        day = q["timestamp"][:10]
        by_day[day] = by_day.get(day, 0) + 1

    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else None
    p95_idx = ceil(len(latencies_sorted) * 0.95) - 1
    p95 = latencies_sorted[min(p95_idx, len(latencies_sorted) - 1)] if latencies_sorted else None

    return AnalyticsResponse(
        accuracy=None,
        retrieval_recall_at_5=None,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        queries_this_week=len(recent_queries),
        unique_users=None,
        queries_by_agent=by_agent,
        queries_by_day=by_day,
        ragas_metrics=_get_ragas_metrics(),
    )


def _get_ragas_metrics() -> dict[str, dict[str, float]]:
    """Return measured RAGAS metrics from the last completed evaluation."""
    cached = _LAST_EVAL_RESULT
    if cached:
        out: dict[str, dict[str, float]] = {}
        for agent in ("hr", "it", "policy"):
            agent_results = [r for r in cached["results"] if r["agent"] == agent]
            if agent_results:
                n = len(agent_results)
                out[agent] = {
                    "faithfulness": round(sum(r["faithfulness"] for r in agent_results) / n, 3),
                    "answer_relevancy": round(sum(r["answer_relevancy"] for r in agent_results) / n, 3),
                    "context_precision": round(sum(r["context_precision"] for r in agent_results) / n, 3),
                    "context_recall": round(sum(r["context_recall"] for r in agent_results) / n, 3),
                }
        return out
    return {}


_LAST_EVAL_RESULT: dict | None = None


# ---------- Seeding ----------

@router.post(
    "/seed",
    dependencies=[Depends(require_admin), Depends(require_csrf)],
)
def seed_db(force: bool = False) -> dict:
    """Seed the vector DB with sample documents."""
    from app.ingestion.ingest import seed_sample_docs

    results = seed_sample_docs(force=force)
    return {"results": results, "total_docs_seeded": len([r for r in results if "error" not in r])}


# ---------- Eval (Phase 3) ----------

@router.get("/eval/summary", dependencies=[Depends(require_admin)])
def eval_summary() -> dict:
    """Return a summary of the default eval set without running it."""
    from app.eval import eval_set_summary
    return eval_set_summary()


@router.post(
    "/eval/run",
    dependencies=[Depends(require_admin), Depends(require_csrf)],
)
def eval_run(save: bool = True) -> dict:
    """Run the RAGAS evaluation pipeline against the gold eval set.

    This is a long-running endpoint (~30s for 24 questions on CPU).
    Returns the full report. Cached in-memory for /api/analytics to surface live values.
    """
    import json
    from dataclasses import asdict
    from app.eval import run_eval

    save_path = Path(settings.vector_db_path).parent / "eval_reports" / "latest.json" if save else None
    report = run_eval(save_path=save_path)

    global _LAST_EVAL_RESULT
    _LAST_EVAL_RESULT = asdict(report)
    return _LAST_EVAL_RESULT


@router.get("/eval/latest", dependencies=[Depends(require_admin)])
def eval_latest() -> dict:
    """Return the most recent eval run results (or null if none yet)."""
    return _LAST_EVAL_RESULT or {"message": "No eval run yet. POST /api/eval/run to start."}


# ---------- Prometheus metrics (Phase 3) ----------

@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(require_admin)],
)
def prometheus_metrics() -> str:
    """Expose metrics in Prometheus exposition format."""
    return metrics.expose()
