"""Smoke tests for the Phase 2 backend.

Run with:
    cd backend
    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the backend app importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    """All key modules should import without error."""
    from app.config import settings
    from app.agents import AGENTS, get_agent, classify_question
    from app.rag.embeddings import get_embeddings
    from app.rag.vectorstore import count_chunks
    from app.rag.pipeline import run_rag
    from app.ingestion.ingest import seed_sample_docs
    from app.models import ChatRequest, ChatResponse
    from app.api.routes import router

    assert settings.app_name
    assert "hr" in AGENTS
    assert "it" in AGENTS
    assert "policy" in AGENTS


def test_relative_storage_paths_are_backend_relative():
    from app.config import BACKEND_DIR, Settings

    settings = Settings(
        _env_file=None,
        auth_database_path="data/auth/auth.sqlite3",
        vector_db_path="data/chroma_db",
    )

    assert Path(settings.auth_database_path) == BACKEND_DIR / "data" / "auth" / "auth.sqlite3"
    assert Path(settings.vector_db_path) == BACKEND_DIR / "data" / "chroma_db"


def test_root_redirects_to_sign_in():
    from fastapi.testclient import TestClient
    from app.main import app

    response = TestClient(app).get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/app/index.html"


def test_production_startup_rejects_missing_agent_collection(monkeypatch):
    from app.config import settings
    from app.main import _require_production_knowledge_base

    monkeypatch.setattr(settings, "environment", "prod")

    with pytest.raises(RuntimeError, match="missing indexed collections for: it"):
        _require_production_knowledge_base({"hr": 4, "it": 0, "policy": 3})


def test_agent_specs():
    from app.agents import get_agent

    hr = get_agent("hr")
    assert hr.id == "hr"
    assert "leave" in hr.system_prompt.lower()

    with pytest.raises(ValueError):
        get_agent("nonexistent")


def test_router_classification():
    from app.agents.router import classify_question, classify_conversation_intent

    assert classify_question("how many leaves am I entitled to?") == "hr"
    assert classify_question("tell me about leave polcy") == "hr"
    assert classify_question("how to reset my password") == "it"
    assert classify_question("my vpn is nt wrking") == "it"
    assert classify_question("what are the data classification levels?") == "policy"
    assert classify_question("gift acceptance policy") == "policy"
    assert classify_question("random question with no keywords") == "hr"  # fallback
    assert classify_conversation_intent("helo") == "greeting"


def test_sample_docs_seed(monkeypatch):
    """Seed orchestration covers each sample document without a live embedding model."""
    from app.rag import vectorstore
    from app.ingestion import ingest
    from app.ingestion.ingest import seed_sample_docs

    ingested = []
    monkeypatch.setattr(vectorstore, "count_chunks", lambda agent: 0)

    def fake_ingest_text(text, name, agent_id, section=None):
        ingested.append((name, agent_id))
        return {"name": name, "agent": agent_id}

    monkeypatch.setattr(ingest, "ingest_text", fake_ingest_text)
    results = seed_sample_docs(force=True)
    assert len(results) == 11  # 4 HR + 3 IT + 4 Policy
    assert len(ingested) == 11
    assert {agent for _, agent in ingested} == {"hr", "it", "policy"}


@pytest.mark.parametrize(
    ("agent", "question", "answer"),
    [
        ("hr", "How many casual leaves am I entitled to?", "12 casual leaves"),
        ("it", "How do I reset my corporate password?", "password reset"),
        ("policy", "What are the data classification levels?", "public, internal"),
    ],
)
def test_rag_pipeline_uses_mocked_model(monkeypatch, agent, question, answer):
    """RAG generation can be tested without credentials or a live model."""
    from langchain_core.messages import AIMessage
    from app.rag import pipeline
    from app.rag.pipeline import RetrievedChunk, run_rag

    chunk = RetrievedChunk(
        document="sample-policy.pdf",
        page=None,
        section="1",
        score=0.7,
        snippet="Policy facts for the mocked response.",
    )

    class MockLLM:
        def invoke(self, messages):
            return AIMessage(content=answer)

    monkeypatch.setattr(pipeline, "retrieve", lambda *_: [chunk])
    monkeypatch.setattr(pipeline, "get_llm", MockLLM)
    result = run_rag(agent, question)
    assert result.agent == agent
    assert result.answer == answer
    assert result.sources == [chunk]


def test_chat_endpoint_via_test_client(monkeypatch):
    """Test the FastAPI chat endpoint via TestClient."""
    from fastapi.testclient import TestClient
    from app.rag.pipeline import RagResult, RetrievedChunk
    from app.main import app

    source = RetrievedChunk(
        document="HR_Policy_Manual_2024.pdf",
        page=1,
        section="1",
        score=0.8,
        snippet="12 casual leaves per year",
    )
    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda agent, question: RagResult(
            answer="12 casual leaves per year",
            agent=agent,
            sources=[source],
            latency_ms=10,
            llm_provider="test",
        ),
    )
    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client, "employee")
    response = client.post(
        "/api/chat",
        json={"question": "How many casual leaves am I entitled to?", "agent": "hr"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["agent"] == "hr"
    assert len(data["sources"]) > 0


def test_documents_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client)
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert data["total"] >= 0


def test_analytics_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client)
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "queries_by_agent" in data
    assert "ragas_metrics" in data


def test_health_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "llm_enabled" in data
    assert "llm_provider" in data
    assert "agents" in data
    assert "hr" in data["agents"]


def test_health_endpoint_does_not_initialize_llm_or_embeddings(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.rag import embeddings, llm, vectorstore

    monkeypatch.setattr(vectorstore, "count_chunks", lambda agent: 1)

    def unexpected_initialization():
        raise AssertionError("Liveness must not initialize AI dependencies.")

    monkeypatch.setattr(embeddings, "get_embeddings", unexpected_initialization)
    monkeypatch.setattr(llm, "get_llm", unexpected_initialization)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_handler_runs_outside_the_event_loop():
    import inspect
    from app.api.routes import chat

    assert not inspect.iscoroutinefunction(chat)


def test_startup_does_not_load_embedding_model(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.rag import embeddings, vectorstore

    loaded = []
    monkeypatch.setattr(embeddings, "get_embeddings", lambda: loaded.append(True))
    monkeypatch.setattr(vectorstore, "count_chunks", lambda agent: 1)

    with TestClient(app):
        pass

    assert loaded == []


def test_chroma_onnx_embeddings_preserve_384_dimensions(monkeypatch):
    from app.rag import embeddings

    calls = []
    constructed = []

    class FakeONNXMiniLM:
        def __init__(self, preferred_providers):
            assert preferred_providers == ["CPUExecutionProvider"]
            constructed.append(self)

        def __call__(self, texts):
            calls.append(texts)
            return [[0.25] * 384 for _ in texts]

    monkeypatch.setattr(
        embeddings,
        "ONNXMiniLM_L6_V2",
        FakeONNXMiniLM,
    )

    adapter = embeddings.ChromaONNXEmbeddings()

    assert len(adapter.embed_query("query")) == 384
    assert len(adapter.embed_documents(["first", "second"])) == 2
    document_vectors = adapter.embed_documents(["text"])
    assert all(len(vector) == 384 for vector in document_vectors)
    assert len(constructed) == 1
    assert calls == [["query"], ["first", "second"], ["text"]]


def test_eval_runner(monkeypatch):
    """Phase 3: eval pipeline runs against the gold set."""
    from app.eval import run_eval, eval_set_summary
    from app.rag.pipeline import RagResult
    from app.eval import runner

    summary = eval_set_summary()
    assert summary["total_questions"] == 24
    assert summary["by_agent"]["hr"] == 8
    assert summary["by_agent"]["it"] == 8
    assert summary["by_agent"]["policy"] == 8

    # Run a subset (4 questions) to keep test fast
    from app.eval.runner import DEFAULT_EVAL_SET
    subset = DEFAULT_EVAL_SET[:4]
    monkeypatch.setattr(
        runner,
        "run_rag",
        lambda agent, question: RagResult(
            answer="12 casual leaves are available each calendar year.",
            agent=agent,
            sources=[],
            latency_ms=1,
            llm_provider="test",
        ),
    )
    report = run_eval(eval_set=subset)
    assert report.total_questions == 4
    assert 0 <= report.avg_overall <= 1
    assert report.p50_latency_ms >= 0


def test_eval_endpoints():
    """Phase 3: /api/eval/summary and /api/eval/latest endpoints work."""
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client)

    r = client.get("/api/eval/summary")
    assert r.status_code == 200
    assert r.json()["total_questions"] == 24

    r = client.get("/api/eval/latest")
    assert r.status_code == 200  # may be empty message or actual results


def test_metrics_endpoint(monkeypatch):
    """Phase 3: /api/metrics returns Prometheus format."""
    from fastapi.testclient import TestClient
    from app.rag.pipeline import RagResult
    from app.main import app

    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda agent, question: RagResult(
            answer="No model needed for this metrics test.",
            agent=agent,
            sources=[],
            latency_ms=1,
            llm_provider="test",
        ),
    )
    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client)
    # Trigger a chat to populate metrics
    client.post("/api/chat", json={"question": "test", "agent": "hr"})

    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert "chat_queries_total" in response.text
