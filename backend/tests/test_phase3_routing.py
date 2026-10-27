"""Phase 3 regression tests — ambiguity clarification, routing and source states.

These tests never touch the vector store: retrieval and generation are always
mocked or the pipeline returns before any Chroma access.
"""
from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.router import (
    CLARIFICATION_MESSAGE,
    is_ambiguous_question,
    resolve_question_agent,
)
from app.rag.pipeline import RagResult, RetrievalError
from app.main import app
from tests.conftest import authenticate_client


AMBIGUOUS_QUESTION = "Can you help me with something?"


def login_client(role: str = "employee") -> tuple[TestClient, str]:
    client = TestClient(app)
    authenticate_client(client, role)
    return client, client.headers["X-CSRF-Token"]


def fail_retrieval(*_args: Any, **_kwargs: Any) -> Any:
    pytest.fail("a clarification must not retrieve documents or call the model")


# ---------- is_ambiguous_question ----------

def test_domain_questions_are_not_ambiguous() -> None:
    assert not is_ambiguous_question("How many leave days do I have?")
    assert not is_ambiguous_question("My VPN is not connecting.")
    assert not is_ambiguous_question("What is the work-from-home policy?")


def test_recognized_social_turns_are_not_ambiguous() -> None:
    assert not is_ambiguous_question("hello")
    assert not is_ambiguous_question("thanks")


def test_contextual_follow_up_with_history_is_not_ambiguous() -> None:
    history = [
        ("user", "How do I set up VPN on Windows?", "it"),
        ("assistant", "Install AnyConnect and restart the laptop.", "it"),
    ]
    assert not is_ambiguous_question("What about Windows?", history)


def test_genuinely_ambiguous_questions_are_flagged() -> None:
    assert is_ambiguous_question(AMBIGUOUS_QUESTION)
    # A follow-up-shaped question with no conversation to refer back to.
    assert is_ambiguous_question("What about Windows?")


def test_specialist_retention_still_wins_for_follow_ups() -> None:
    history = [
        ("user", "How do I set up VPN on Windows?", "it"),
        ("assistant", "Install AnyConnect and restart the laptop.", "it"),
    ]
    assert (
        resolve_question_agent("What about Windows?", conversation_history=history)
        == "it"
    )


# ---------- POST /api/chat ----------

def test_ambiguous_question_without_an_agent_hint_asks_for_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.routes.run_rag", fail_retrieval)
    from app.api.routes import _QUERY_LOG

    client, _ = login_client()
    before = len(_QUERY_LOG)
    response = client.post("/api/chat", json={"question": AMBIGUOUS_QUESTION})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == CLARIFICATION_MESSAGE
    assert payload["agent"] == ""
    assert payload["sources"] == []
    assert payload["error_code"] is None
    # A clarification is not a knowledge query and stays out of analytics.
    assert len(_QUERY_LOG) == before


def test_clarification_skips_retrieval_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.routes.run_rag", fail_retrieval)

    client, _ = login_client()
    response = client.post("/api/chat", json={"question": AMBIGUOUS_QUESTION})

    assert response.status_code == 200
    assert response.json()["llm_provider"] == "none"


def test_explicit_topic_hint_keeps_the_existing_fallback_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rag(agent: str, question: str, history: Any = None) -> RagResult:
        captured["agent"] = agent
        return RagResult("Answer.", agent, [], 1, "none")

    monkeypatch.setattr("app.api.routes.run_rag", fake_rag)

    client, _ = login_client()
    response = client.post(
        "/api/chat",
        json={"question": AMBIGUOUS_QUESTION, "agent": "hr"},
    )

    assert response.status_code == 200
    assert response.json()["agent"] == "hr"
    assert captured["agent"] == "hr"


def test_omitted_agent_still_routes_clearly_domain_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rag(agent: str, question: str, history: Any = None) -> RagResult:
        captured["agent"] = agent
        return RagResult("Answer.", agent, [], 1, "none")

    monkeypatch.setattr("app.api.routes.run_rag", fake_rag)

    client, _ = login_client()
    response = client.post(
        "/api/chat", json={"question": "How do I reset my password?"}
    )

    assert response.status_code == 200
    assert response.json()["agent"] == "it"
    assert captured["agent"] == "it"


def test_follow_up_with_history_keeps_the_specialist_without_an_agent_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_rag(agent: str, question: str, history: Any = None) -> RagResult:
        captured["agent"] = agent
        return RagResult("Answer.", agent, [], 1, "none")

    monkeypatch.setattr("app.api.routes.run_rag", fake_rag)

    client, _ = login_client()
    response = client.post(
        "/api/chat",
        json={
            "question": "What about Windows?",
            "conversation_history": [
                {"role": "user", "content": "How do I set up VPN on Windows?", "agent": "it"},
                {"role": "assistant", "content": "Install AnyConnect and restart.", "agent": "it"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["agent"] == "it"
    assert captured["agent"] == "it"


# ---------- POST /api/chat/ws ----------

def test_websocket_ambiguous_question_returns_clarification_without_rag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.api.routes.run_rag_stream", fail_retrieval)
    client, csrf_token = login_client()

    with client.websocket_connect(
        "/api/chat/ws", headers={"origin": "http://testserver"}
    ) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "phase3-clarify",
            "csrf_token": csrf_token,
            "payload": {"question": AMBIGUOUS_QUESTION},
        })
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {"complete", "error"}:
                break

    kinds = [event["type"] for event in events]
    assert kinds == ["start", "status", "complete"]
    assert events[1]["payload"] == {"stage": "routing", "agent": ""}
    payload = events[-1]["payload"]
    assert payload["answer"] == CLARIFICATION_MESSAGE
    assert payload["agent"] == ""
    assert payload["sources"] == []
    client.close()


def test_websocket_keeps_routing_status_for_domain_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stream(agent: str, question: str, history: Any, emit_token: Any,
                    emit_status: Any, cancelled: Any) -> RagResult:
        emit_status("retrieving", {})
        return RagResult("Answer.", agent, [], 3, "test")

    monkeypatch.setattr("app.api.routes.run_rag_stream", fake_stream)
    client, csrf_token = login_client()

    with client.websocket_connect(
        "/api/chat/ws", headers={"origin": "http://testserver"}
    ) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "phase3-routing",
            "csrf_token": csrf_token,
            "payload": {"question": "How do I reset my password?"},
        })
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {"complete", "error"}:
                break

    status = next(event for event in events if event["type"] == "status")
    assert status["payload"]["agent"] == "it"
    assert events[-1]["payload"]["agent"] == "it"
    client.close()


# ---------- Source states: no evidence vs retrieval failure ----------

def test_missing_evidence_message_states_the_required_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag import pipeline

    monkeypatch.setattr(pipeline, "retrieve", lambda _agent, _question: [])

    result = pipeline.run_rag("hr", "What is the policy on sabbaticals?")

    assert result.answer.startswith(
        "I couldn't find this information in the company documents."
    )
    assert result.error_code == "insufficient_evidence"
    assert result.sources == []
    assert result.llm_provider == "none"


def test_retrieval_failure_is_distinct_from_missing_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag import pipeline

    def boom(_agent: str, _question: str) -> list[Any]:
        raise RetrievalError("vector store unavailable")

    monkeypatch.setattr(pipeline, "retrieve", boom)

    result = pipeline.run_rag("hr", "How many leave days do I have?")

    assert result.error_code == "retrieval_error"
    assert result.answer == pipeline.RETRIEVAL_ERROR_MESSAGE
    # The knowledge base was never searched, so nothing was proven absent.
    assert "couldn't find this information" not in result.answer
    assert result.sources == []


def test_streamed_retrieval_failure_carries_the_retrieval_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag import pipeline

    def boom(_agent: str, _question: str) -> list[Any]:
        raise RetrievalError("vector store unavailable")

    monkeypatch.setattr(pipeline, "retrieve", boom)

    result = pipeline.run_rag_stream(
        "hr",
        "How many leave days do I have?",
        [],
        lambda _token: None,
        lambda _stage, _data: None,
        threading.Event(),
    )

    assert result.error_code == "retrieval_error"
    assert result.answer == pipeline.RETRIEVAL_ERROR_MESSAGE
    assert result.sources == []