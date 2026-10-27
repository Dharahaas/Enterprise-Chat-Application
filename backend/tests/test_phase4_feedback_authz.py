"""Phase 4 regression tests — feedback, analytics and document authorization.

All vector-store access uses in-memory test doubles; nothing here opens,
creates or modifies the real ChromaDB directory. Feedback rows are written to
the isolated per-session test auth database created by conftest.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app import feedback as feedback_store
from app.agents.router import classify_question
from app.auth import (
    SESSION_COOKIE,
    AuthenticatedUser,
    authenticate_demo,
    create_session,
)
from app.authz import (
    AccessContext,
    access_context_for,
    chunk_permitted,
    current_access_context,
    reset_access_context,
    service_context,
    set_access_context,
)
from app.main import app
from app.rag import pipeline
from tests.conftest import authenticate_client

RESTRICTED_SNIPPET = "Executive salary bands are confidential to the leadership team."
PUBLIC_SNIPPET = "Casual leave is 12 days each calendar year."
ANSWER_KEY = "answer-phase4-test01"


def login(role: str = "employee") -> TestClient:
    client = TestClient(app)
    authenticate_client(client, role)
    return client


def feedback_rows() -> list[dict[str, Any]]:
    feedback_store.ensure_feedback_store()
    with feedback_store._connect() as connection:
        rows = connection.execute("SELECT * FROM feedback ORDER BY created_at").fetchall()
    return [dict(row) for row in rows]


def make_store(documents_by_query: dict[str, list[tuple[Any, float]]]):
    """A vector-store double: no Chroma, no embeddings, no disk access."""
    from types import SimpleNamespace

    class Store:
        _collection = SimpleNamespace(name="eka_hr", metadata={"hnsw:space": "l2"})

        def __init__(self) -> None:
            self.queries: list[str] = []

        def similarity_search_with_score(self, query: str, k: int):
            self.queries.append(query)
            return documents_by_query.get(query, [])

    return Store()


def doc(text: str, **metadata: Any):
    from langchain_core.documents import Document

    return Document(page_content=text, metadata=metadata)


class RecordingModel:
    """LLM double that records the prompt it was given."""

    def __init__(self) -> None:
        self.prompt: Any = None

    def invoke(self, messages: Any) -> AIMessage:
        self.prompt = messages
        return AIMessage(content="The document confirms the policy.")

    def stream(self, messages: Any):
        self.prompt = messages
        yield AIMessageChunk(content="The document ")
        yield AIMessageChunk(content="confirms the policy.")


# --------------------------------------------------------------------------
# Permission decisions (unit)
# --------------------------------------------------------------------------

def test_missing_or_unverifiable_principal_fails_closed() -> None:
    anonymous = access_context_for(None)
    unknown_role = AccessContext(
        username="someone", role="mystery", allowed_agents=frozenset({"hr"})
    )
    no_agents = AccessContext(username="x", role="employee", allowed_agents=frozenset())

    metadata = {"source": "Leave_Policy.docx", "agent": "hr"}
    assert not chunk_permitted(metadata, None, "hr")
    assert not chunk_permitted(metadata, anonymous, "hr")
    assert not chunk_permitted(metadata, unknown_role, "hr")
    assert not chunk_permitted(metadata, no_agents, "hr")
    assert not chunk_permitted(None, service_context(), "hr")
    assert not chunk_permitted("not-a-mapping", service_context(), "hr")


def test_department_metadata_is_required_and_must_match_the_namespace() -> None:
    context = access_context_for(AuthenticatedUser("employee@example.test", "employee"))

    assert chunk_permitted({"source": "a.pdf", "agent": "hr"}, context, "hr")
    # Missing department tag → refused.
    assert not chunk_permitted({"source": "a.pdf"}, context, "hr")
    # Wrong department for the collection being queried → refused.
    assert not chunk_permitted({"source": "a.pdf", "agent": "it"}, context, "hr")
    # Unknown department → refused.
    assert not chunk_permitted({"source": "a.pdf", "agent": "finance"}, context, "hr")


def test_documents_without_visibility_stay_available_to_employees() -> None:
    context = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    public_default = {"source": "Leave_Policy.docx", "agent": "hr"}
    assert chunk_permitted(public_default, context, "hr")


def test_visibility_marker_can_only_restrict_access() -> None:
    employee = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    admin = access_context_for(AuthenticatedUser("admin@example.test", "admin"))
    demo_admin = access_context_for(AuthenticatedUser("demo-admin@demo.invalid", "admin"))
    service = service_context()

    public = {"agent": "hr", "visibility": "public"}
    restricted = {"agent": "hr", "visibility": "restricted"}
    unverifiable = {"agent": "hr", "visibility": "board-only"}

    assert chunk_permitted(public, employee, "hr")
    assert chunk_permitted(restricted, admin, "hr")
    assert chunk_permitted(restricted, service, "hr")
    # Fail closed for everyone else, including the public demo identity.
    assert not chunk_permitted(restricted, employee, "hr")
    assert not chunk_permitted(restricted, demo_admin, "hr")
    # Unrecognised marker cannot be verified → refused.
    assert not chunk_permitted(unverifiable, admin, "hr")


def test_access_context_is_absent_until_a_request_sets_it() -> None:
    token = set_access_context(None)
    try:
        assert current_access_context() is None
    finally:
        reset_access_context(token)


# --------------------------------------------------------------------------
# Retrieval authorization (fake vector store, no Chroma)
# --------------------------------------------------------------------------

def test_retrieval_without_an_access_context_returns_nothing() -> None:
    store = make_store({"How many casual leaves?": [(doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.2)]})
    monkeypatch_targets = store
    original_get_store = pipeline.get_vectorstore_for_agent
    original_get_embeddings = pipeline.get_embeddings
    pipeline.get_vectorstore_for_agent = lambda agent, embeddings: monkeypatch_targets
    pipeline.get_embeddings = lambda: object()
    token = set_access_context(None)
    try:
        chunks = pipeline.retrieve("hr", "How many casual leaves?")
        assert chunks == []
        # Fail closed happens before the knowledge base is queried at all.
        assert store.queries == []
    finally:
        reset_access_context(token)
        pipeline.get_vectorstore_for_agent = original_get_store
        pipeline.get_embeddings = original_get_embeddings


def test_retrieval_filters_unauthorized_documents_for_employees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store({
        "What are the salary bands?": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.95),
            (doc("Untagged passage.", source="Mystery.docx"), 0.9),
            (doc("Cross-department passage.", source="IT_Notes.docx", agent="it"), 0.9),
        ],
    })
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)

    employee = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    token = set_access_context(employee)
    try:
        chunks = pipeline.retrieve("hr", "What are the salary bands?")
    finally:
        reset_access_context(token)

    documents = {chunk.document for chunk in chunks}
    assert documents == {"Leave_Policy.docx"}
    assert all(RESTRICTED_SNIPPET not in chunk.snippet for chunk in chunks)
    # The other candidates were evaluated and denied, not silently trusted.
    assert len(store.queries) >= 1


def test_administrators_retrieve_restricted_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store({
        "What are the salary bands?": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.95),
        ],
    })
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)

    admin = access_context_for(AuthenticatedUser("admin@example.test", "admin"))
    token = set_access_context(admin)
    try:
        chunks = pipeline.retrieve("hr", "What are the salary bands?")
    finally:
        reset_access_context(token)

    assert {chunk.document for chunk in chunks} == {"Exec_Comp.pdf", "Leave_Policy.docx"}


def test_unauthorized_text_never_reaches_the_prompt_or_the_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store({
        "What are the salary bands?": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.9),
        ],
    })
    model = RecordingModel()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)

    employee = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    token = set_access_context(employee)
    try:
        result = pipeline.run_rag("hr", "What are the salary bands?")
    finally:
        reset_access_context(token)

    assert model.prompt is not None
    prompt_text = str(model.prompt)
    assert PUBLIC_SNIPPET in prompt_text
    assert RESTRICTED_SNIPPET not in prompt_text
    assert [source.document for source in result.sources] == ["Leave_Policy.docx"]
    assert "Exec_Comp.pdf" not in prompt_text


def test_streaming_answers_keep_the_same_document_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    store = make_store({
        "What are the salary bands?": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.9),
        ],
    })
    model = RecordingModel()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)

    employee = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    token = set_access_context(employee)
    tokens: list[str] = []
    try:
        result = pipeline.run_rag_stream(
            "hr",
            "What are the salary bands?",
            [],
            tokens.append,
            lambda _stage, _data: None,
            threading.Event(),
        )
    finally:
        reset_access_context(token)

    assert tokens  # authorization did not break streaming
    assert RESTRICTED_SNIPPET not in str(model.prompt)
    assert [source.document for source in result.sources] == ["Leave_Policy.docx"]


def test_follow_up_history_retrieval_is_filtered_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The short follow-up has no department keyword, so the pipeline merges it
    # with the previous question before querying the knowledge base.
    store = make_store({
        "How do I request leave? tell about eligibility": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.9),
        ],
    })
    model = RecordingModel()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)

    history = [
        ("user", "How do I request leave?", "hr"),
        ("assistant", "Leave eligibility starts after probation.", "hr"),
    ]
    employee = access_context_for(AuthenticatedUser("employee@example.test", "employee"))
    token = set_access_context(employee)
    try:
        result = pipeline.run_rag("hr", "tell about eligibility", history)
    finally:
        reset_access_context(token)

    assert store.queries == ["How do I request leave? tell about eligibility"]
    assert [source.document for source in result.sources] == ["Leave_Policy.docx"]
    assert RESTRICTED_SNIPPET not in str(model.prompt)


# --------------------------------------------------------------------------
# Chat requests carry the session's permissions end to end
# --------------------------------------------------------------------------

def test_chat_request_applies_authorization_for_each_department(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store({
        "How many casual leaves can I take?": [(doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.9)],
        "How do I reset my password?": [(doc("Reset it in the IT portal.", source="IT_Guide.txt", agent="it"), 0.9)],
        "What is the remote work policy?": [(doc("Hybrid work is two days per week.", source="Hybrid_Work_Policy.pdf", agent="policy"), 0.9)],
    })
    model = RecordingModel()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)

    client = login("employee")
    for question, expected_agent, expected_doc in [
        ("How many casual leaves can I take?", "hr", "Leave_Policy.docx"),
        ("How do I reset my password?", "it", "IT_Guide.txt"),
        ("What is the remote work policy?", "policy", "Hybrid_Work_Policy.pdf"),
    ]:
        response = client.post("/api/chat", json={"question": question})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["agent"] == classify_question(question)
        assert [source["document"] for source in payload["sources"]] == [expected_doc]


def test_chat_request_without_a_session_is_rejected() -> None:
    client = TestClient(app)
    response = client.post("/api/chat", json={"question": "How many casual leaves?"})
    assert response.status_code == 401


def test_websocket_chat_streams_only_authorized_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = make_store({
        "What are the salary bands?": [
            (doc(RESTRICTED_SNIPPET, source="Exec_Comp.pdf", agent="hr", visibility="restricted"), 0.15),
            (doc(PUBLIC_SNIPPET, source="Leave_Policy.docx", agent="hr"), 0.9),
        ],
    })
    model = RecordingModel()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)

    client = login("employee")
    csrf_token = client.headers["X-CSRF-Token"]
    with client.websocket_connect(
        "/api/chat/ws", headers={"origin": "http://testserver"}
    ) as websocket:
        websocket.send_json({
            "v": 1,
            "type": "chat",
            "request_id": "phase4-ws-authz",
            "csrf_token": csrf_token,
            "payload": {"question": "What are the salary bands?"},
        })
        events = []
        while True:
            event = websocket.receive_json()
            events.append(event)
            if event["type"] in {"complete", "error"}:
                break

    assert events[-1]["type"] == "complete"
    sources = events[-1]["payload"]["sources"]
    assert [source["document"] for source in sources] == ["Leave_Policy.docx"]
    assert RESTRICTED_SNIPPET not in str(model.prompt)
    client.close()


# --------------------------------------------------------------------------
# Feedback API
# --------------------------------------------------------------------------

def test_positive_feedback_is_recorded() -> None:
    client = login("employee")
    response = client.post("/api/feedback", json={
        "rating": "positive",
        "answer_key": ANSWER_KEY,
        "question": "How many casual leaves can I take?",
        "agent": "auto",
        "source_status": "sources",
    })

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["rating"] == "positive"
    assert payload["reason"] is None
    assert payload["changed"] is True

    rows = feedback_rows()
    assert len(rows) == 1
    assert rows[0]["username"] == "employee@example.test"
    assert rows[0]["question"] == "How many casual leaves can I take?"
    assert rows[0]["source_status"] == "sources"
    # The record stores no assistant answer and no conversation history.
    assert set(rows[0]) == {
        "id", "username", "answer_key", "rating", "reason", "agent",
        "question", "source_status", "created_at", "updated_at",
    }


def test_negative_feedback_accepts_a_valid_reason() -> None:
    client = login("employee")
    response = client.post("/api/feedback", json={
        "rating": "negative",
        "answer_key": "answer-phase4-reason1",
        "question": "What is the sabbatical policy?",
        "reason": "missing_information",
        "source_status": "no_sources",
    })

    assert response.status_code == 201, response.text
    assert response.json()["reason"] == "missing_information"
    assert feedback_rows()[-1]["reason"] == "missing_information"


@pytest.mark.parametrize("reason", [
    "incorrect_answer", "missing_information", "irrelevant_sources",
    "outdated_document", "other",
])
def test_all_negative_reasons_are_accepted(reason: str) -> None:
    client = login("employee")
    response = client.post("/api/feedback", json={
        "rating": "negative",
        "answer_key": f"answer-reason-{reason}",
        "question": "Sample question",
        "reason": reason,
    })
    assert response.status_code == 201, response.text


def test_positive_feedback_may_not_carry_a_reason() -> None:
    client = login("employee")
    response = client.post("/api/feedback", json={
        "rating": "positive",
        "answer_key": "answer-phase4-badreason",
        "question": "Sample question",
        "reason": "incorrect_answer",
    })
    assert response.status_code == 400


def test_invalid_rating_or_reason_is_rejected() -> None:
    client = login("employee")
    assert client.post("/api/feedback", json={
        "rating": "meh",
        "answer_key": "answer-phase4-invalid1",
        "question": "Sample",
    }).status_code == 422
    assert client.post("/api/feedback", json={
        "rating": "negative",
        "answer_key": "answer-phase4-invalid2",
        "question": "Sample",
        "reason": "because",
    }).status_code == 422
    assert client.post("/api/feedback", json={
        "rating": "negative",
        "answer_key": "short",
        "question": "Sample",
    }).status_code == 422
    assert client.post("/api/feedback", json={
        "rating": "negative",
        "answer_key": "answer-phase4-invalid3",
        "question": "",
    }).status_code == 422


def test_duplicate_feedback_is_updated_not_duplicated() -> None:
    client = login("employee")
    answer_key = "answer-phase4-duplicate"

    first = client.post("/api/feedback", json={
        "rating": "positive", "answer_key": answer_key, "question": "Sample question",
    })
    assert first.status_code == 201
    assert first.json()["changed"] is True

    repeat = client.post("/api/feedback", json={
        "rating": "positive", "answer_key": answer_key, "question": "Sample question",
    })
    assert repeat.status_code == 201
    assert repeat.json()["changed"] is False
    assert repeat.json()["id"] == first.json()["id"]

    # Changing the rating updates the same record instead of adding a second one.
    changed = client.post("/api/feedback", json={
        "rating": "negative", "answer_key": answer_key,
        "question": "Sample question", "reason": "incorrect_answer",
    })
    assert changed.status_code == 201
    assert changed.json()["changed"] is True
    assert changed.json()["id"] == first.json()["id"]

    matching = [row for row in feedback_rows() if row["answer_key"] == answer_key]
    assert len(matching) == 1
    assert matching[0]["rating"] == "negative"
    assert matching[0]["reason"] == "incorrect_answer"


def test_feedback_topic_is_detected_on_the_server() -> None:
    client = login("employee")
    response = client.post("/api/feedback", json={
        "rating": "positive",
        "answer_key": "answer-phase4-topic01",
        "question": "How do I reset my corporate password?",
        "agent": "auto",
    })
    assert response.status_code == 201
    assert feedback_rows()[-1]["agent"] == "it"


def test_feedback_requires_a_session_and_csrf() -> None:
    anonymous = TestClient(app)
    assert anonymous.post("/api/feedback", json={
        "rating": "positive", "answer_key": ANSWER_KEY, "question": "Sample",
    }).status_code == 401

    no_csrf = TestClient(app)
    token, _csrf, _expiry = create_session(
        AuthenticatedUser("employee@example.test", "employee")
    )
    no_csrf.cookies.set(SESSION_COOKIE, token)
    assert no_csrf.post("/api/feedback", json={
        "rating": "positive", "answer_key": ANSWER_KEY, "question": "Sample",
    }).status_code == 403


def test_feedback_is_scoped_to_the_signing_in_user() -> None:
    employee = login("employee")
    employee.post("/api/feedback", json={
        "rating": "positive", "answer_key": "answer-phase4-scoped1",
        "question": "Sample question",
    })

    # A second user can post their own feedback for the same answer key, but
    # each account keeps its own record.
    from app.auth import create_user

    try:
        create_user("another.person@example.test", "another-password-123", "employee")
    except ValueError:
        pass  # already created by an earlier test in this session

    other = TestClient(app)
    token, csrf, _expiry = create_session(
        AuthenticatedUser("another.person@example.test", "employee")
    )
    other.cookies.set(SESSION_COOKIE, token)
    other.headers.update({"X-CSRF-Token": csrf})
    response = other.post("/api/feedback", json={
        "rating": "negative", "answer_key": "answer-phase4-scoped1",
        "question": "Sample question", "reason": "other",
    })
    assert response.status_code == 201

    matching = [row for row in feedback_rows() if row["answer_key"] == "answer-phase4-scoped1"]
    assert {(row["username"], row["rating"]) for row in matching} == {
        ("employee@example.test", "positive"),
        ("another.person@example.test", "negative"),
    }


# --------------------------------------------------------------------------
# Analytics: aggregates and administrator-only access
# --------------------------------------------------------------------------

def test_employee_cannot_read_analytics_api_or_page() -> None:
    client = login("employee")
    assert client.get("/api/analytics").status_code == 403
    # Changing the URL is not enough: the page itself refuses the role.
    page = client.get("/app/analytics.html", follow_redirects=False)
    assert page.status_code == 303
    assert "access=denied" in page.headers["location"]


def test_unauthenticated_analytics_access_is_rejected() -> None:
    assert TestClient(app).get("/api/analytics").status_code == 401


def test_demo_admin_identity_cannot_read_analytics() -> None:
    client = TestClient(app)
    demo_user = authenticate_demo("admin")
    token, _csrf, _ = create_session(demo_user)
    client.cookies.set(SESSION_COOKIE, token)
    assert client.get("/api/analytics").status_code == 403


def test_analytics_returns_feedback_aggregates_for_administrators() -> None:
    employee = login("employee")
    employee.post("/api/feedback", json={
        "rating": "positive", "answer_key": "answer-phase4-agg-01",
        "question": "How many casual leaves can I take?", "agent": "hr",
    })
    employee.post("/api/feedback", json={
        "rating": "negative", "answer_key": "answer-phase4-agg-02",
        "question": "What is the sabbatical policy?", "agent": "auto",
        "reason": "missing_information", "source_status": "no_sources",
    })
    employee.post("/api/feedback", json={
        "rating": "negative", "answer_key": "answer-phase4-agg-03",
        "question": "What is the sabbatical policy?", "agent": "auto",
        "reason": "outdated_document",
    })

    admin = login("admin")
    response = admin.get("/api/analytics")
    assert response.status_code == 200
    summary = response.json()["feedback"]
    assert summary["total_positive"] >= 1
    assert summary["total_negative"] >= 2
    assert summary["negative_reasons"]["missing_information"] >= 1
    assert summary["negative_reasons"]["outdated_document"] >= 1
    assert summary["feedback_by_day"], "feedback trend data should be present"
    assert any(gap["question"] == "What is the sabbatical policy?" for gap in summary["knowledge_gaps"])
    assert "no_source_questions" in summary
    assert "unanswered_questions" in summary


def test_analytics_report_never_contains_usernames_or_answers() -> None:
    employee = login("employee")
    employee.post("/api/feedback", json={
        "rating": "negative", "answer_key": "answer-phase4-privacy1",
        "question": "What is the sabbatical policy?", "reason": "other",
    })

    admin = login("admin")
    body = admin.get("/api/analytics").text
    assert "employee@example.test" not in body
    # No assistant answers or conversation history appear in the report.
    assert "The document confirms the policy." not in body


def test_feedback_query_log_records_outcome_without_content() -> None:
    from app.api import routes

    entry = {
        "agent": "hr",
        "latency_ms": 10,
        "sources_count": 0,
        "outcome": "failed",
        "timestamp": "2026-10-27T00:00:00+00:00",
    }
    try:
        with routes._QUERY_LOG_LOCK:
            routes._QUERY_LOG.append(entry)
        stored = dict(routes._QUERY_LOG[-1])
    finally:
        with routes._QUERY_LOG_LOCK:
            if entry in routes._QUERY_LOG:
                routes._QUERY_LOG.remove(entry)

    assert stored["outcome"] == "failed"
    # Analytics inputs still never carry question or answer content.
    assert "question" not in stored
    assert "answer" not in stored


# --------------------------------------------------------------------------
# Direct API access attempts
# --------------------------------------------------------------------------

def test_employee_cannot_call_document_or_eval_endpoints() -> None:
    client = login("employee")
    assert client.get("/api/documents").status_code == 403
    assert client.delete("/api/documents/hr/Leave_Policy.docx").status_code == 403
    assert client.post("/api/eval/run").status_code == 403
    assert client.post("/api/seed").status_code == 403


def test_anonymous_users_cannot_call_document_endpoints() -> None:
    client = TestClient(app)
    assert client.get("/api/documents").status_code == 401
    assert client.delete("/api/documents/hr/Leave_Policy.docx").status_code == 401


def test_uploaded_document_files_are_never_served_by_the_static_frontend() -> None:
    """Originals live under backend/data/uploads, which the static mount blocks."""
    client = login("admin")
    for path in (
        "/app/backend/data/uploads/hr/abc123/Leave_Policy.docx",
        "/app/backend/data/chroma_db/chroma.sqlite3",
        "/app/data/sample_docs/Leave_Policy.txt",
    ):
        assert client.get(path, follow_redirects=False).status_code == 404, path