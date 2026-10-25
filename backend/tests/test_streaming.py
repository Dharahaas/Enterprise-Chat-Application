from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.api import routes
from app.config import settings
from app.main import app
from app.rag import llm, pipeline
from app.rag.pipeline import RagResult, RetrievedChunk
from starlette.websockets import WebSocketDisconnect


def _login(client: TestClient, role: str = "employee") -> tuple[str, str]:
    username = "admin@example.test" if role == "admin" else "employee@example.test"
    password = "test-admin-password-123" if role == "admin" else "test-employee-password-123"
    login = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login.status_code == 200
    return login.json()["csrf_token"], login.json()["user"]["username"]


def test_huggingface_provider_configuration_and_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "hf_token", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert not llm.is_llm_enabled()

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    assert llm.is_llm_enabled()


def test_huggingface_uses_hosted_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "hf_token", "hf_test_token")
    monkeypatch.setattr(settings, "hf_model_id", "openai/gpt-oss-20b")
    monkeypatch.setattr(settings, "hf_provider", "auto")
    captured: dict[str, Any] = {}

    def capture(provider: str, model: str, api_key: str, *args: Any) -> str:
        captured.update(provider=provider, model=model, api_key=api_key)
        return "mock-client"

    real_cached_client = llm._cached_chat_model
    monkeypatch.setattr(llm, "_cached_chat_model", capture)
    assert llm.get_llm() == "mock-client"
    assert captured == {
        "provider": "huggingface",
        "model": "openai/gpt-oss-20b",
        "api_key": "hf_test_token",
    }
    assert llm.llm_model_name() == "openai/gpt-oss-20b"

    client_options: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> object:
        client_options.update(kwargs)
        return object()

    monkeypatch.setattr(llm, "_client_fingerprint", None)
    monkeypatch.setattr(llm, "_client_instance", None)
    real_cached_client(
        "huggingface", "openai/gpt-oss-20b", "hf_test_token", 0.2, 512, fake_client
    )
    assert client_options["base_url"] == "https://router.huggingface.co/v1"
    assert client_options["api_key"] == "hf_test_token"


def test_llm_status_exposes_configuration_but_never_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "hf_token", "hf_sensitive_test_token")
    monkeypatch.setattr(settings, "hf_model_id", "openai/gpt-oss-20b:fastest")
    monkeypatch.setattr(routes, "_LLM_LAST_SUCCESS_AT", None)
    response = TestClient(app).get("/api/llm/status")
    assert response.status_code == 200
    assert response.json() == {
        "provider": "huggingface",
        "model_id": "openai/gpt-oss-20b:fastest",
        "configured": True,
        "last_successful_inference_at": None,
    }
    assert "hf_sensitive_test_token" not in response.text


def test_rag_stream_emits_model_chunks_and_keeps_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = RetrievedChunk(
        document="Leave_Policy.pdf", page=2, section="Leave", score=0.8,
        snippet="Employees receive 12 casual leave days each year.",
    )

    class MockStreamingModel:
        def stream(self, _prompt: list[Any]):
            yield AIMessageChunk(content="The policy provides ")
            yield AIMessageChunk(content="12 casual leave days each year.")

    monkeypatch.setattr(pipeline, "retrieve", lambda _agent, _question: [chunk])
    monkeypatch.setattr(pipeline, "get_llm", lambda: MockStreamingModel())
    tokens: list[str] = []
    statuses: list[str] = []
    result = pipeline.run_rag_stream(
        "hr", "How many casual leave days?", [], tokens.append,
        lambda stage, _data: statuses.append(stage), threading.Event(),
    )

    assert tokens == ["The policy provides ", "12 casual leave days each year."]
    assert "retrieving" in statuses and "streaming" in statuses
    assert result.answer == "The policy provides 12 casual leave days each year."
    assert result.sources == [chunk]


def test_rag_stream_preserves_company_policy_no_document_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "retrieve", lambda _agent, _question: [])
    monkeypatch.setattr(
        pipeline, "get_llm",
        lambda: pytest.fail("unsupported company-specific question must not call the model"),
    )
    tokens: list[str] = []
    result = pipeline.run_rag_stream(
        "policy", "What is our remote work policy?", [], tokens.append,
        lambda _stage, _metadata: None, threading.Event(),
    )
    assert result.error_code == "insufficient_evidence"
    assert "don't want to guess" in result.answer.lower()
    assert tokens == []


def test_streamed_websocket_sends_tokens_then_authoritative_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(agent, question, history, emit_token, emit_status, cancelled):
        assert agent == "policy"
        assert question == "What is our remote work policy?"
        assert history[-1] == (
            "assistant", "The current hybrid work allowance is two days.", "policy"
        )
        emit_status("retrieving", {})
        emit_status("retrieved", {"retrieval_ms": 4})
        emit_status("streaming", {"retrieval_ms": 4})
        emit_token("Remote work is allowed.")
        return RagResult(
            answer="Remote work is allowed two days per week.",
            agent=agent,
            sources=[RetrievedChunk("Hybrid_Work_Policy.pdf", 1, "Hybrid", 0.8, "Two days per week.")],
            latency_ms=12,
            llm_provider="huggingface",
        )

    monkeypatch.setattr(routes, "run_rag_stream", fake_stream)
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "hf_model_id", "openai/gpt-oss-20b:fastest")
    client = TestClient(app)
    csrf_token, _ = _login(client)

    with client.websocket_connect(
        "/api/chat/ws", headers={"origin": "http://testserver"}
    ) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "stream-test-1",
            "csrf_token": csrf_token,
            "payload": {
                "question": "What is our remote work policy?",
                "agent": "hr",
                "conversation_id": "conversation-1",
                "conversation_history": [
                    {"role": "user", "content": "How do I request leave?", "agent": "hr"},
                    {"role": "assistant", "content": "Use the HR portal.", "agent": "hr"},
                    {"role": "user", "content": "What is the remote work policy?", "agent": "hr"},
                    {"role": "assistant", "content": "The current hybrid work allowance is two days.", "agent": "policy"},
                ],
            },
        })
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {"complete", "error"}:
                break

    assert [event["type"] for event in events if event["type"] in {"start", "token", "complete"}] == [
        "start", "token", "complete"
    ]
    complete = events[-1]
    assert complete["payload"]["answer"] == "Remote work is allowed two days per week."
    assert complete["payload"]["agent"] == "policy"
    assert complete["payload"]["sources"][0]["document"] == "Hybrid_Work_Policy.pdf"
    assert complete["payload"]["conversation_id"] == "conversation-1"
    assert complete["payload"]["provider"] == "huggingface"
    assert complete["payload"]["retrieval_ms"] == 4
    client.close()


def test_websocket_short_follow_up_keeps_the_previous_specialist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, list[tuple[str, str, str | None]]]] = []

    def fake_stream(agent, _question, history, _emit_token, _emit_status, _cancelled):
        captured.append((agent, history))
        return RagResult("A shorter answer grounded in the policy.", agent, [], 3, "test")

    monkeypatch.setattr(routes, "run_rag_stream", fake_stream)
    client = TestClient(app)
    csrf_token, _ = _login(client)
    history = [
        {"role": "user", "content": "What is the remote work policy?", "agent": "hr"},
        {"role": "assistant", "content": "The hybrid policy allows approved remote work.", "agent": "policy"},
    ]

    with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "context-follow-up",
            "csrf_token": csrf_token,
            "payload": {
                "question": "Can you simplify that?",
                "agent": "hr",
                "conversation_id": "same-conversation",
                "conversation_history": history,
            },
        })
        while True:
            event = websocket.receive_json()
            if event["type"] in {"complete", "error"}:
                break

    assert event["type"] == "complete"
    assert event["payload"]["agent"] == "policy"
    assert event["payload"]["conversation_id"] == "same-conversation"
    assert captured == [("policy", [
        ("user", "What is the remote work policy?", "hr"),
        ("assistant", "The hybrid policy allows approved remote work.", "policy"),
    ])]
    client.close()


def test_websocket_rejects_invalid_origin_and_bad_csrf() -> None:
    client = TestClient(app)
    csrf_token, _ = _login(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/chat/ws", headers={"origin": "https://attacker.invalid"}):
            pass

    with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as websocket:
        websocket.send_json({
            "v": 1, "type": "chat", "request_id": "bad-csrf",
            "csrf_token": "invalid", "payload": {"question": "Hello", "agent": "hr"},
        })
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "csrf"
    client.close()


def test_websocket_rejects_unauthenticated_client() -> None:
    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}):
            pass
    client.close()


def test_websocket_rejects_malformed_chat_payload() -> None:
    client = TestClient(app)
    csrf_token, _ = _login(client)
    with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as websocket:
        websocket.send_json({
            "v": 1, "type": "chat", "request_id": "bad-payload",
            "csrf_token": csrf_token,
            "payload": {"question": "", "agent": "unsupported-agent"},
        })
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "invalid_message"
    client.close()


def test_websocket_client_disconnect_signals_stream_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    observed_cancel = threading.Event()

    def fake_stream(agent, question, history, emit_token, emit_status, cancelled):
        started.set()
        while not cancelled.wait(0.01):
            time.sleep(0.01)
        observed_cancel.set()
        return RagResult("", agent, [], 1, "huggingface", "cancelled")

    monkeypatch.setattr(routes, "run_rag_stream", fake_stream)
    client = TestClient(app)
    csrf_token, _ = _login(client)
    with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as websocket:
        websocket.send_json({
            "v": 1, "type": "chat", "request_id": "disconnect-test",
            "csrf_token": csrf_token,
            "payload": {"question": "How many leave days?", "agent": "hr"},
        })
        assert websocket.receive_json()["type"] == "start"
        assert started.wait(2)
    assert observed_cancel.wait(2)
    deadline = time.monotonic() + 2
    while "employee@example.test" in routes._ACTIVE_WS_USERS and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "employee@example.test" not in routes._ACTIVE_WS_USERS
    client.close()


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (type("ProviderRateLimit", (Exception,), {"status_code": 429})("limited"), "quota"),
        (TimeoutError("upstream timeout"), "network"),
    ],
)
def test_websocket_provider_failures_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_code: str,
) -> None:
    def fail_stream(*_args: Any, **_kwargs: Any) -> RagResult:
        raise error

    monkeypatch.setattr(routes, "run_rag_stream", fail_stream)
    client = TestClient(app)
    csrf_token, _ = _login(client, "admin")
    with client.websocket_connect("/api/chat/ws", headers={"origin": "http://testserver"}) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "provider-error",
            "csrf_token": csrf_token,
            "payload": {"question": "What is the remote work policy?", "agent": "hr"},
        })
        while True:
            event = websocket.receive_json()
            if event["type"] == "error":
                break
    assert event["error"]["code"] == expected_code
    assert event["error"]["message"] == "The assistant could not complete this request."
    client.close()
