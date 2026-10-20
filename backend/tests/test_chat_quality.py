from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.rag import llm, pipeline
from app.rag.pipeline import RetrievedChunk


class MockLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.messages: list[Any] | None = None
        self.invocation_count = 0

    def invoke(self, messages: list[Any]) -> Any:
        self.invocation_count += 1
        self.messages = messages
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def sample_chunks(score: float = 0.7) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            document="Leave_Policy.docx",
            page=None,
            section="3.1",
            score=score,
            snippet="Casual leave: 12 days per calendar year.",
        )
    ]


def mock_generation(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[RetrievedChunk],
    response: Any,
) -> MockLLM:
    model = MockLLM(response)
    monkeypatch.setattr(pipeline, "retrieve", lambda agent, question: chunks)
    monkeypatch.setattr(pipeline, "get_llm", lambda: model)
    return model


def post_chat(
    question: str,
    agent: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    from fastapi.testclient import TestClient
    from app.main import app

    body: dict[str, Any] = {"question": question}
    if agent is not None:
        body["agent"] = agent
    if conversation_history is not None:
        body["conversation_history"] = conversation_history
    client = TestClient(app)
    from tests.conftest import authenticate_client
    authenticate_client(client, "employee")
    response = client.post(
        "/api/chat",
        json=body,
    )
    assert response.status_code == 200
    return response.json()


def test_hello_is_conversational_and_skips_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_retrieval(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("greetings must not retrieve documents")

    monkeypatch.setattr(pipeline, "retrieve", unexpected_retrieval)
    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda *args, **kwargs: pytest.fail("greetings must not run RAG"),
    )

    result = post_chat("hello")

    assert result["answer"] == (
        "Hello! How can I help you today? You can ask me about HR policies, "
        "IT support, or company guidelines."
    )
    assert result["sources"] == []


def test_typo_greeting_is_understood(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda *args, **kwargs: pytest.fail("corrected greeting must not run RAG"),
    )

    assert post_chat("helo")["answer"].startswith("Hello!")


@pytest.mark.parametrize(
    ("question", "expected_intent", "expected_reply"),
    [
        ("thanks", "thanks", "You're welcome!"),
        ("thank you", "thanks", "You're welcome!"),
        ("ThankYou!!!", "thanks", "You're welcome!"),
        ("thank you   ", "thanks", "You're welcome!"),
        ("ok thank you", "thanks", "You're welcome!"),
        ("OK, THANK YOU!", "thanks", "You're welcome!"),
        ("thankyou", "thanks", "You're welcome!"),
        ("ok thanks", "thanks", "You're welcome!"),
        ("ok", "acknowledgement", "Okay!"),
        ("okay.", "acknowledgement", "Okay!"),
        ("got it", "acknowledgement", "Okay!"),
        ("ok, got it", "acknowledgement", "Okay!"),
        ("okay thanks", "thanks", "You're welcome!"),
        ("good", "positive_acknowledgement", "Glad to hear it!"),
        ("great!", "positive_acknowledgement", "Glad to hear it!"),
        ("bye", "farewell", "Goodbye!"),
        ("what can you do?", "capabilities", "I can help answer questions"),
        ("how can you help me?", "capabilities", "I can help answer questions"),
        ("how can u help me?", "capabilities", "I can help answer questions"),
    ],
)
def test_short_conversation_turns_skip_rag_and_keep_selected_agent(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    expected_intent: str,
    expected_reply: str,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda *args, **kwargs: pytest.fail("social questions must not run RAG"),
    )

    result = post_chat(question, agent="it")

    assert result["answer"].startswith(expected_reply)
    assert result["agent"] == "it"
    assert result["sources"] == []


def test_good_in_a_knowledge_question_is_not_acknowledgement() -> None:
    from app.agents.router import classify_conversation_intent

    assert classify_conversation_intent("that sounds like a good leave policy") is None


@pytest.mark.parametrize("agent", ["hr", "it", "policy"])
def test_conversation_response_preserves_each_selected_agent(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    monkeypatch.setattr(
        "app.api.routes.run_rag",
        lambda *args, **kwargs: pytest.fail("acknowledgements must not run RAG"),
    )

    result = post_chat("ok", agent=agent)

    assert result["agent"] == agent
    assert result["llm_provider"] == "none"


def test_misspelled_leave_policy_query_is_normalized_and_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="The policy provides 12 casual-leave days per calendar year."),
    )

    result = post_chat("tell me about leave polcy", agent="hr")

    assert result["agent"] == "hr"
    assert result["answer"] == "The policy provides 12 casual-leave days per calendar year."
    assert result["sources"][0]["document"] == "Leave_Policy.docx"
    assert model.messages is not None
    assert "tell me about leave policy" in str(model.messages).lower()


def test_vpm_typo_routes_to_it_and_retrieves_vpn_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved: list[tuple[str, str]] = []
    model = mock_generation(
        monkeypatch,
        [
            RetrievedChunk(
                document="IT_Network_Guide_v2.pdf",
                page=None,
                section="2",
                score=0.37,
                snippet="Section 2: VPN Configuration. Connect using Cisco AnyConnect.",
            )
        ],
        AIMessage(content="The IT guide covers VPN configuration using Cisco AnyConnect."),
    )
    original_retrieve = pipeline.retrieve

    def capture_retrieval(agent_id: str, question: str) -> list[RetrievedChunk]:
        retrieved.append((agent_id, question))
        return original_retrieve(agent_id, question)

    monkeypatch.setattr(pipeline, "retrieve", capture_retrieval)

    result = post_chat("tell me how to setup vpm")

    assert result["agent"] == "it"
    assert result["answer"] == (
        "The IT guide covers VPN configuration using Cisco AnyConnect."
    )
    assert retrieved == [("it", "tell me how to setup vpn")]
    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert model.messages is not None
    assert "tell me how to setup vpn" in str(model.messages).lower()


@pytest.mark.parametrize(
    ("question", "agent", "normalized_question"),
    [
        ("my vpn is nt wrking", "it", "my vpn is not working"),
        ("wht is the code of conduct", "policy", "what is the code of conduct"),
    ],
)
def test_typo_normalization_preserves_domain_routing(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    agent: str,
    normalized_question: str,
) -> None:
    model = mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="The retrieved document contains the relevant guidance."),
    )

    result = post_chat(question)

    assert result["agent"] == agent
    assert model.messages is not None
    assert normalized_question in str(model.messages).lower()


def test_relevant_hr_context_is_passed_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="Employees receive 12 casual-leave days each calendar year."),
    )

    result = pipeline.run_rag("hr", "How many casual leaves can I take?")

    assert result.answer == "Employees receive 12 casual-leave days each calendar year."
    assert result.sources[0].document == "Leave_Policy.docx"
    assert model.messages is not None
    assert "12 days per calendar year" in str(model.messages)
    assert "Chunk 1" not in str(model.messages)


def test_hr_prompt_separates_leave_extensions_from_other_entitlements() -> None:
    prompt = pipeline.build_prompt(
        "hr",
        "Can I extend sick leave beyond 12 days?",
        "12 sick leave days per year. 15 earned leave days per year.",
    )

    assert "first state whether the documents describe an extension" in prompt[0].content
    assert "promptly highlight any useful, documented alternative" in prompt[0].content
    assert "not an extension or substitute for sick leave" in prompt[0].content


def test_it_prompt_allows_plain_language_definition_but_grounds_company_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vpn_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.7,
        snippet="VPN Configuration: Connect using Cisco AnyConnect at vpn.company.com.",
    )
    model = mock_generation(
        monkeypatch,
        [vpn_context],
        AIMessage(
            content=(
                "A VPN (Virtual Private Network) creates a protected connection "
                "between your device and your company's network over the internet. "
                "Think of it like a secure tunnel. The company guide says to connect "
                "using Cisco AnyConnect at vpn.company.com."
            )
        ),
    )

    result = post_chat("I don't understand what this VPN is. Help me understand it.", agent="it")

    assert "Think of it like a secure tunnel" in result["answer"]
    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert model.messages is not None
    system_message = model.messages[0].content
    assert "plain-language general explanation" in system_message
    assert "Ground company-specific configuration, access, and support procedures" in system_message
    assert "do not invent support procedures" in system_message.casefold()


@pytest.mark.parametrize(
    ("question", "previous_question", "expected_query"),
    [
        ("Any simpler way?", "How do I set up VPN?", "How do I set up VPN? Any simpler way?"),
        ("What if that fails?", "What if my MFA fails?", "What if my MFA fails? What if that fails?"),
        ("What about Windows?", "How do I set up VPN?", "How do I set up VPN? What about Windows?"),
        ("i need for macOS", "tell me how to setup vpm", "tell me how to setup vpn i need for macOS"),
        ("Where do I contact them?", "How do I set up VPN?", "How do I set up VPN? Where do I contact them?"),
    ],
)
def test_it_follow_up_retrieval_keeps_previous_question_context(
    question: str,
    previous_question: str,
    expected_query: str,
) -> None:
    assert pipeline._retrieval_topic_question(
        question,
        [("user", previous_question), ("assistant", "Use the documented procedure.")],
    ) == expected_query


def test_it_vpn_query_prioritizes_os_setup_instructions() -> None:
    assert pipeline._retrieval_queries("it", "How to set up VPN on Windows?") == [
        "How to set up VPN on Windows? Windows Cisco AnyConnect VPN configuration"
    ]


def test_it_direct_vpn_setup_uses_relevant_documented_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vpn_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.74,
        snippet=(
            "Windows: Use Cisco AnyConnect. Install it from the IT Self-Service "
            "Portal, open it, enter vpn.company.com, sign in with corporate "
            "credentials, then enter the Microsoft Authenticator 2FA token."
        ),
    )
    model = mock_generation(
        monkeypatch,
        [vpn_context],
        AIMessage(
            content=(
                "A VPN creates a protected connection to the company network.\n\n"
                "Prerequisite: use the documented Cisco AnyConnect client.\n\n"
                "1. Install Cisco AnyConnect from the IT Self-Service Portal.\n"
                "2. Open it and enter vpn.company.com.\n"
                "3. Sign in with your corporate credentials and enter your "
                "Microsoft Authenticator 2FA token."
            )
        ),
    )

    result = post_chat("How do I set up VPN on Windows?", agent="it")

    answer = result["answer"]
    assert answer.index("A VPN creates") < answer.index("Prerequisite:")
    assert answer.index("Prerequisite:") < answer.index("1. Install Cisco AnyConnect")
    assert "2. Open it and enter vpn.company.com." in answer
    assert "Microsoft Authenticator 2FA token" in answer
    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert model.messages is not None
    prompt = str(model.messages).casefold()
    assert "brief plain-language explanation" in prompt
    assert "documented prerequisites" in prompt
    assert "requested operating system" in prompt
    assert "vpn.company.com" in str(model.messages)


def test_it_windows_vpn_question_followed_by_macos_uses_history_and_cites_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved_queries: list[str] = []
    mac_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.625,
        snippet=(
            "macOS:\n"
            "1. Install Cisco AnyConnect Secure Mobility Client from the IT "
            "Self-Service Portal\n"
            "2. Open AnyConnect\n"
            "3. Enter vpn.company.com as the host\n"
            "4. Sign in with corporate credentials (employee_id@company.com)\n"
            "5. Enter 2FA token from Microsoft Authenticator\n"
            "6. Once connected, you will receive a 10.x.x.x internal IP"
        ),
    )
    model = mock_generation(
        monkeypatch,
        [mac_context],
        AIMessage(
            content=(
                "A VPN creates a protected connection to the company network.\n\n"
                "Prerequisite: use the documented Cisco AnyConnect client.\n\n"
                "1. Install Cisco AnyConnect from the IT Self-Service Portal.\n"
                "2. Open AnyConnect and enter vpn.company.com.\n"
                "3. Sign in with your corporate credentials and enter the "
                "Microsoft Authenticator 2FA token."
            )
        ),
    )
    actual_retrieve = pipeline.retrieve

    def capture_retrieve(agent_id: str, query: str) -> list[RetrievedChunk]:
        retrieved_queries.append(query)
        assert agent_id == "it"
        return actual_retrieve(agent_id, query)

    monkeypatch.setattr(pipeline, "retrieve", capture_retrieve)

    result = post_chat(
        "i need for macOS",
        agent="it",
        conversation_history=[
            {"role": "user", "content": "tell me how to setup vpm"},
            {
                "role": "assistant",
                "content": "Windows VPN instructions were provided.",
            },
        ],
    )

    assert result["agent"] == "it"
    answer = result["answer"]
    assert answer.index("A VPN creates") < answer.index("Prerequisite:")
    assert answer.index("Prerequisite:") < answer.index("1. Install Cisco AnyConnect")
    assert "2. Open AnyConnect and enter vpn.company.com." in answer
    assert len(retrieved_queries) == 1
    assert "vpn" in retrieved_queries[0].casefold()
    assert "macos" in retrieved_queries[0].casefold()
    assert "Cisco AnyConnect" in result["answer"]
    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert result["sources"][0]["section"] == "2"
    assert model.invocation_count == 1
    assert model.messages is not None
    prompt = str(model.messages)
    assert "macOS" in prompt
    assert "Windows VPN instructions" in prompt
    assert "requested operating system" in prompt
    assert "use another platform's steps only when the source explicitly says they are the same" in prompt


def test_conversation_history_is_request_scoped_across_authenticated_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient
    from app.api import routes
    from app.main import app
    from app.rag.pipeline import RagResult
    from tests.conftest import authenticate_client

    received: list[tuple[str, str, list[tuple[str, str]]]] = []

    def capture_rag(
        agent: str,
        question: str,
        history: list[tuple[str, str]] | None = None,
    ) -> RagResult:
        received.append((agent, question, history or []))
        return RagResult(
            answer="Grounded test response.",
            agent=agent,
            sources=[],
            latency_ms=1,
            llm_provider="test",
        )

    monkeypatch.setattr(routes, "run_rag", capture_rag)
    requests = [
        (
            "employee",
            "Any simpler way?",
            "it",
            [("user", "How do I set up VPN?")],
        ),
        (
            "admin",
            "Where is the reimbursement policy?",
            "hr",
            [("user", "How many days to submit a claim?")],
        ),
    ]

    for role, question, agent, history in requests:
        client = TestClient(app)
        authenticate_client(client, role)
        response = client.post(
            "/api/chat",
            json={
                "question": question,
                "agent": agent,
                "conversation_history": [
                    {"role": turn_role, "content": content}
                    for turn_role, content in history
                ],
            },
        )
        assert response.status_code == 200
        client.close()

    assert received == [
        ("it", "Any simpler way?", [("user", "How do I set up VPN?")]),
        (
            "hr",
            "Where is the reimbursement policy?",
            [("user", "How many days to submit a claim?")],
        ),
    ]


def test_it_mfa_question_uses_documented_steps_and_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mfa_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.72,
        snippet=(
            "If MFA fails: re-register Microsoft Authenticator via IT Portal. "
            "Contact IT desk: ext. 4500 or it@company.com."
        ),
    )
    model = mock_generation(
        monkeypatch,
        [mfa_context],
        AIMessage(
            content=(
                "Re-register Microsoft Authenticator via the IT Portal. "
                "If you still cannot sign in, contact the IT desk at ext. 4500 "
                "or it@company.com."
            )
        ),
    )

    result = post_chat("What if my MFA fails?", agent="it")

    assert "re-register microsoft authenticator" in result["answer"].casefold()
    assert "ext. 4500" in result["answer"]
    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert model.messages is not None
    assert "re-register Microsoft Authenticator via IT Portal" in str(model.messages)


def test_it_otp_timeout_and_escalation_preserve_full_condition_and_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    otp_source = RetrievedChunk(
        document="IT_SOP_PasswordReset.pdf",
        page=2,
        section="1",
        score=0.76,
        snippet=(
            "If you do not receive the OTP within 2 minutes, contact the IT desk "
            "at ext. 4500."
        ),
    )
    model = mock_generation(
        monkeypatch,
        [otp_source],
        AIMessage(content="Contact the IT desk at ext. 4500."),
    )

    result = post_chat("what if i dont get an otp", agent="it")

    assert result["answer"] == (
        "If you do not receive the OTP within 2 minutes, contact the IT desk at ext. 4500."
    )
    assert result["sources"][0]["document"] == "IT_SOP_PasswordReset.pdf"
    assert result["sources"][0]["page"] == 2
    assert result["sources"][0]["section"] == "1"
    assert model.invocation_count == 1


def test_it_simpler_follow_up_preserves_source_conditions() -> None:
    source = RetrievedChunk(
        document="IT_SOP_PasswordReset.pdf",
        page=None,
        section="1",
        score=0.76,
        snippet=(
            "If you do not receive the OTP within 2 minutes, contact the IT desk "
            "at ext. 4500."
        ),
    )

    answer = pipeline._preserve_procedural_details(
        "it",
        "What if I do not get an OTP? Any simpler way?",
        "Contact the IT desk at ext. 4500.",
        [source],
    )

    assert answer == (
        "If you do not receive the OTP within 2 minutes, contact the IT desk at ext. 4500."
    )


def test_it_mfa_failure_preserves_documented_contact_when_generation_omits_it() -> None:
    source = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.72,
        snippet=(
            "If MFA fails: re-register Microsoft Authenticator via IT Portal.\n"
            "- Contact IT desk: ext. 4500 or it@company.com"
        ),
    )

    answer = pipeline._preserve_procedural_details(
        "it",
        "What if my MFA fails?",
        "Re-register Microsoft Authenticator via IT Portal.",
        [source],
    )

    assert "Re-register Microsoft Authenticator" in answer
    assert "ext. 4500" in answer
    assert "it@company.com" in answer


def test_it_helpdesk_follow_up_prioritizes_escalation_retrieval() -> None:
    query = pipeline._retrieval_topic_question(
        "Where do I contact them?",
        [("user", "What if my MFA fails?")],
    )

    assert pipeline._retrieval_queries("it", query) == [
        "What if my MFA fails? Where do I contact them? IT desk support contact escalation"
    ]


def test_it_missing_procedure_falls_back_to_documented_support_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_core.documents import Document

    class FakeCollection:
        name = "eka_it"
        metadata = {"hnsw:space": "l2"}

    class FakeStore:
        _collection = FakeCollection()

        def __init__(self) -> None:
            self.queries: list[str] = []

        def similarity_search_with_score(self, query: str, k: int):
            self.queries.append(query)
            if query == "IT help desk support contact escalation":
                return [
                    (
                        Document(
                            page_content="Contact the IT desk at ext. 4500.",
                            metadata={"source": "IT_Support_Guide.txt"},
                        ),
                        0.2,
                    )
                ]
            return [
                (
                    Document(
                        page_content="Unrelated software inventory.",
                        metadata={"source": "Asset_Policy.docx"},
                    ),
                    1.7,
                )
            ]

    store = FakeStore()
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "get_vectorstore_for_agent",
        lambda agent, embeddings: store,
    )

    chunks = pipeline.retrieve("it", "How do I configure the unsupported tool?")

    assert store.queries[-1] == "IT help desk support contact escalation"
    assert len(chunks) == 1
    assert chunks[0].document == "IT_Support_Guide.txt"
    assert "ext. 4500" in chunks[0].snippet


def test_it_follow_up_context_is_sent_to_the_model_and_sources_remain_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vpn_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.7,
        snippet="Windows uses Cisco AnyConnect with the same VPN setup flow.",
    )
    model = mock_generation(
        monkeypatch,
        [vpn_context],
        AIMessage(content="1. Open Cisco AnyConnect and follow the same setup flow."),
    )

    result = post_chat(
        "What about Windows?",
        agent="it",
        conversation_history=[
            {"role": "user", "content": "How do I set up VPN?"},
            {"role": "assistant", "content": "Here are the setup steps."},
        ],
    )

    assert result["sources"][0]["document"] == "IT_Network_Guide_v2.pdf"
    assert model.messages is not None
    prompt_text = str(model.messages)
    assert "How do I set up VPN?" in prompt_text
    assert "Windows uses Cisco AnyConnect" in prompt_text


def test_it_simpler_follow_up_keeps_the_previous_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vpn_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.71,
        snippet="Open AnyConnect, enter vpn.company.com, and sign in with your credentials.",
    )
    model = mock_generation(
        monkeypatch,
        [vpn_context],
        AIMessage(content="1. Open AnyConnect. 2. Enter vpn.company.com and sign in."),
    )

    result = post_chat(
        "Any simpler way?",
        agent="it",
        conversation_history=[
            {"role": "user", "content": "How do I set up VPN?"},
            {"role": "assistant", "content": "Here are all six detailed steps."},
        ],
    )

    assert result["answer"].startswith("1. Open AnyConnect")
    assert model.messages is not None
    assert "How do I set up VPN?" in str(model.messages)


@pytest.mark.parametrize(
    "follow_up",
    [
        "How do I do that?",
        "What happens next?",
        "what happens after that?",
        "What about IT?",
    ],
)
def test_short_follow_up_keeps_the_previous_topic(
    monkeypatch: pytest.MonkeyPatch,
    follow_up: str,
) -> None:
    retrieved_queries: list[str] = []
    vpn_context = RetrievedChunk(
        document="IT_Network_Guide_v2.pdf",
        page=None,
        section="2",
        score=0.71,
        snippet="Open AnyConnect, enter vpn.company.com, and sign in with your credentials.",
    )
    model = mock_generation(
        monkeypatch,
        [vpn_context],
        AIMessage(content="Continue with the documented VPN steps."),
    )
    actual_retrieve = pipeline.retrieve

    def capture_retrieve(agent_id: str, query: str) -> list[RetrievedChunk]:
        retrieved_queries.append(query)
        return actual_retrieve(agent_id, query)

    monkeypatch.setattr(pipeline, "retrieve", capture_retrieve)
    result = pipeline.run_rag(
        "it",
        follow_up,
        [("user", "How do I set up VPN?"), ("assistant", "Here are the steps.")],
    )

    assert result.sources
    assert len(retrieved_queries) == 1
    assert "vpn" in retrieved_queries[0].casefold()
    assert model.messages is not None
    assert "How do I set up VPN?" in str(model.messages)


def test_unrelated_low_confidence_matches_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = mock_generation(
        monkeypatch,
        sample_chunks(score=0.01),
        AIMessage(content="This answer must not be used."),
    )

    result = pipeline.run_rag("hr", "What is quantum entanglement?")

    assert "couldn't find enough relevant information" in result.answer.lower()
    assert result.sources == []
    assert model.messages is None


def test_chat_analytics_does_not_retain_raw_question(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import routes

    monkeypatch.setattr(routes, "_QUERY_LOG", [])
    mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="The relevant HR document supports this answer."),
    )

    post_chat("A private question that should not be retained", agent="hr")

    assert len(routes._QUERY_LOG) == 1
    assert "question" not in routes._QUERY_LOG[0]


def test_marginal_retrieval_matches_are_filtered() -> None:
    assert pipeline.rerank(sample_chunks(score=0.349)) == []
    assert len(pipeline.rerank(sample_chunks(score=0.35))) == 1


def test_policy_overview_accepts_section_reference_and_keeps_sources_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = mock_generation(
        monkeypatch,
        [
            RetrievedChunk(
                document="InfoSec_Policy_v4.pdf",
                page=None,
                section="2",
                score=0.7,
                snippet="Section 2 covers information security and data handling.",
            )
        ],
        AIMessage(
            content=(
                "Section 2 describes information-security and data-handling "
                "requirements. The retrieved policies also cover hybrid work "
                "and conduct. Which topic would you like to explore?"
            )
        ),
    )

    result = post_chat("give me info regarding policy", agent="policy")

    assert result["agent"] == "policy"
    assert "Section 2 describes" in result["answer"]
    assert "InfoSec_Policy_v4.pdf" not in result["answer"]
    assert result["sources"][0]["document"] == "InfoSec_Policy_v4.pdf"
    assert model.messages is not None
    assert "broad request for policy information" in str(model.messages).lower()


@pytest.mark.parametrize("agent", ["hr", "policy"])
def test_explicit_selected_agent_is_preserved_for_policy_keyword(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="The retrieved document provides relevant guidance."),
    )

    result = post_chat("give me info regarding policy", agent=agent)

    assert result["agent"] == agent


def test_omitted_agent_uses_intent_based_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="The retrieved document provides relevant guidance."),
    )

    result = post_chat("give me info regarding policy")

    assert result["agent"] == "policy"


def test_no_documents_returns_not_found_without_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = mock_generation(monkeypatch, [], AIMessage(content="fabricated"))

    result = pipeline.run_rag("hr", "What is the policy on sabbaticals?")

    assert "couldn't find enough relevant information" in result.answer.lower()
    assert result.sources == []
    assert model.messages is None


def test_missing_api_key_returns_configuration_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm.settings, "openai_api_key", "")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm, "is_llm_enabled", lambda: False)
    monkeypatch.setattr(pipeline, "retrieve", lambda agent, question: sample_chunks())
    monkeypatch.setattr(pipeline, "get_llm", llm.get_llm)

    result = pipeline.run_rag("hr", "How many casual leaves do I get?")

    assert "language model" in result.answer.lower()
    assert "gemini_api_key" in result.answer.lower()
    assert "template" not in result.answer.lower()
    assert result.sources


def test_configured_environment_key_is_passed_to_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    received: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        received.update(kwargs)
        return object()

    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm.settings, "openai_api_key", "")
    monkeypatch.setenv("OPENAI_API_KEY", "invalid-test-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)

    llm.get_llm()

    assert received["api_key"] == "invalid-test-key"
    assert received["model"] == llm.settings.openai_model


def test_gemini_uses_google_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    received: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        received.update(kwargs)
        return object()

    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(llm, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)

    llm.get_llm()

    assert received["model"] == "gemini-3.5-flash-lite"
    assert received["base_url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert received["api_key"] == "test-gemini-key"


def test_gemini_model_is_read_from_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    received: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        received.update(kwargs)
        return object()

    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-3.8-flash")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(llm, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)

    llm.get_llm()

    assert received["model"] == "gemini-3.8-flash"


@pytest.mark.parametrize("agent", ["hr", "it", "policy"])
def test_model_unavailable_is_safe_and_consistent_for_every_agent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    agent: str,
) -> None:
    class ModelUnavailable(Exception):
        status_code = 404
        body = {
            "error": {
                "code": "NOT_FOUND",
                "message": "Model gemini-3.5-flash-lite unavailable for key sk-test-secret",
            }
        }

    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-3.5-flash-lite")
    monkeypatch.setattr(llm, "llm_provider_name", lambda: "gemini")
    mock_generation(monkeypatch, sample_chunks(), ModelUnavailable())

    with caplog.at_level("WARNING", logger="app.rag.pipeline"):
        result = post_chat("How do I set up VPM?", agent=agent)

    assert result["agent"] == agent
    assert result["error_code"] == "model_unavailable"
    assert "temporarily unavailable" in result["answer"].lower()
    assert "sk-test-secret" not in caplog.text
    assert "gemini-3.5-flash-lite" in caplog.text


def test_gemini_without_key_is_not_reported_as_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")

    assert not llm.is_llm_enabled()
    assert llm.llm_provider_name() == "unconfigured"
    with pytest.raises(llm.LLMNotConfiguredError):
        llm.get_llm()


def test_gemini_environment_key_enables_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")

    assert llm.is_llm_enabled()
    assert llm.llm_provider_name() == "gemini"


def test_google_api_key_name_is_not_a_configured_project_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_FILE", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    configured = Settings(_env_file=None)

    assert configured.gemini_api_key == ""


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            RuntimeError("401 invalid api key sk-test-secret"),
            "unable to generate an answer",
        ),
        (
            TimeoutError("provider request timed out"),
            "could not reach gemini",
        ),
    ],
)
def test_provider_errors_are_safe_and_do_not_leak_exception(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_message: str,
) -> None:
    model = mock_generation(
        monkeypatch,
        sample_chunks(),
        error,
    )

    result = pipeline.run_rag("hr", "How many casual leaves do I get?")

    assert expected_message in result.answer.lower()
    assert "sk-test-secret" not in result.answer
    assert result.sources
    assert model.messages is not None


def test_provider_diagnostics_identify_quota_without_leaking_credentials() -> None:
    class ProviderError(Exception):
        status_code = 429
        body = {
            "error": {
                "code": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for API key sk-test-secret",
            }
        }

    category, status, code, message = llm.diagnose_llm_error(ProviderError())

    assert category == "quota"
    assert status == 429
    assert code == "RESOURCE_EXHAUSTED"
    assert "sk-test-secret" not in message
    assert "[REDACTED]" in message


@pytest.mark.parametrize(
    ("status", "code", "expected_category"),
    [
        (401, "UNAUTHENTICATED", "authentication"),
        (404, "NOT_FOUND", "model_unavailable"),
        (503, "UNAVAILABLE", "provider_error"),
    ],
)
def test_provider_diagnostics_classify_http_failures(
    status: int,
    code: str,
    expected_category: str,
) -> None:
    class ProviderError(Exception):
        pass

    error = ProviderError("provider failure")
    error.status_code = status
    error.body = {"error": {"code": code, "message": "provider failure"}}

    category, actual_status, actual_code, _ = llm.diagnose_llm_error(error)

    assert category == expected_category
    assert actual_status == status
    assert actual_code == code


def test_generation_failure_logs_stage_provider_model_and_redacts_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-3.5-flash-lite")
    model = mock_generation(
        monkeypatch,
        sample_chunks(),
        RuntimeError("provider rejected key sk-test-secret"),
    )

    with caplog.at_level("WARNING", logger="app.rag.pipeline"):
        result = pipeline.run_rag("hr", "How many casual leaves do I get?")

    assert "unable to generate an answer" in result.answer.lower()
    assert "stage=invoke" in caplog.text
    assert "provider=gemini" in caplog.text
    assert "model=gemini-3.5-flash-lite" in caplog.text
    assert "provider rejected key" in caplog.text
    assert "sk-test-secret" not in caplog.text
    assert model.messages is not None


@pytest.mark.parametrize(
    "content",
    [
        "",
        '{"answer":',
        "Chunk 1 [Source: Leave_Policy.docx]",
        'Answer:"',
        "12 casual leaves (Leave_Policy.docx, section 3.1; score: 0.8)",
    ],
)
def test_empty_or_malformed_model_output_is_not_exposed(
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> None:
    mock_generation(monkeypatch, sample_chunks(), AIMessage(content=content))

    result = pipeline.run_rag("hr", "How many casual leaves do I get?")

    assert "malformed response" in result.answer.lower()
    assert "chunk 1" not in result.answer.lower()
    assert 'answer:"' not in result.answer.lower()


@pytest.mark.parametrize(
    "content",
    [
        "System: Follow these instructions.",
        "  **Context:** Retrieved passages go here.",
        "> Question: What did the user ask?",
    ],
)
def test_prompt_headers_are_rejected_at_line_start(content: str) -> None:
    with pytest.raises(pipeline.MalformedModelResponseError):
        pipeline._extract_answer(AIMessage(content=content))


def test_normal_prose_containing_system_label_is_not_rejected() -> None:
    answer = (
        "The VPN system: connect with Cisco AnyConnect and follow the IT guide."
    )

    assert pipeline._extract_answer(AIMessage(content=answer)) == answer


def test_inline_document_filename_is_sanitized_without_losing_answer() -> None:
    answer = "Report suspicious messages as described in InfoSec_Policy_v4.pdf."

    sanitized = pipeline._extract_answer(AIMessage(content=answer))

    assert sanitized == "Report suspicious messages as described in the source document."
    assert "InfoSec_Policy_v4.pdf" not in sanitized


def test_answer_and_citations_are_returned_in_separate_schema_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_generation(
        monkeypatch,
        sample_chunks(),
        AIMessage(content="Employees receive 12 casual-leave days per calendar year."),
    )

    result = post_chat("How many casual leaves can I take?")

    assert set(result) == {
        "answer",
        "agent",
        "sources",
        "latency_ms",
        "llm_provider",
        "error_code",
        "conversation_id",
        "timestamp",
    }
    assert result["error_code"] is None
    assert isinstance(result["llm_provider"], str)
    assert "Leave_Policy.docx" not in result["answer"]
    assert result["sources"][0]["document"] == "Leave_Policy.docx"
    assert result["sources"][0]["section"] == "3.1"


@pytest.mark.parametrize("agent", ["hr", "it", "policy"])
def test_agent_prompts_ask_for_direct_useful_formatting_without_forced_templates(
    agent: str,
) -> None:
    messages = pipeline.build_prompt(agent, "How can I proceed?", "Confirmed passage.")
    system_prompt = messages[0].content

    assert "Answer the user's actual question directly" in system_prompt
    assert "keep greetings and simple questions brief" in system_prompt
    assert "Use short Markdown headings, bullets, or numbered steps only when" in system_prompt
    assert "avoid stock introductions, repeated conclusions, and filler" in system_prompt.casefold()
    assert "source labels, filenames" in system_prompt
    assert "source references are returned separately" in system_prompt


def test_structured_it_answer_keeps_numbered_steps_and_separate_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = RetrievedChunk(
        document="IT_Network_Guide.pdf",
        page=4,
        section="2.1",
        score=0.82,
        snippet="Open the approved VPN client, then select Connect.",
    )
    answer = "VPN setup\n\n1. Open the approved VPN client.\n2. Select Connect."
    mock_generation(monkeypatch, [chunk], AIMessage(content=answer))

    result = post_chat("How do I connect to VPN?", agent="it")

    assert result["answer"] == answer
    assert result["sources"][0]["document"] == "IT_Network_Guide.pdf"
    assert result["sources"][0]["page"] == 4
    assert "IT_Network_Guide.pdf" not in result["answer"]


def test_reusable_model_client_tracks_configuration_without_caching_raw_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import langchain_openai

    created: list[object] = []

    def fake_chat_openai(**kwargs: Any) -> object:
        client = object()
        created.append(client)
        return client

    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-cache-test")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "cache-test-secret")
    monkeypatch.setattr(llm, "is_llm_enabled", lambda: True)
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", fake_chat_openai)

    first = llm.get_llm()
    second = llm.get_llm()
    assert first is second
    assert len(created) == 1
    assert b"cache-test-secret" not in repr(llm._client_fingerprint).encode("utf-8")

    monkeypatch.setattr(llm.settings, "gemini_model", "gemini-cache-test-updated")
    third = llm.get_llm()
    assert third is not first
    assert len(created) == 2


def test_retrieval_distance_is_converted_using_squared_l2_geometry() -> None:
    assert pipeline.distance_to_similarity(0.0, "l2") == 1.0
    assert pipeline.distance_to_similarity(1.0, "l2") == 0.5
    assert pipeline.distance_to_similarity(2.0, "l2") == 0.0
    assert pipeline.distance_to_similarity(0.4, "cosine") == pytest.approx(0.6)
    assert pipeline.distance_to_similarity(2.5, "l2") == 0.0


def test_retrieval_uses_collection_metric_filters_irrelevant_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from langchain_core.documents import Document

    duplicate = "Casual leave is 12 days each calendar year."
    results = [
        (Document(page_content=duplicate, metadata={"source": "Leave_Policy.docx"}), 0.2),
        (Document(page_content=duplicate, metadata={"source": "Leave_Policy.docx"}), 0.3),
        (
            Document(page_content="Irrelevant payroll details.", metadata={"source": "Payroll.docx"}),
            1.8,
        ),
    ]
    store = SimpleNamespace(
        _collection=SimpleNamespace(metadata={"hnsw:space": "l2"}),
        similarity_search_with_score=lambda question, k: results,
    )
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)

    chunks = pipeline.retrieve("hr", "how many casual leaves")

    assert len(chunks) == 1
    assert chunks[0].score == pytest.approx(0.9)
    assert chunks[0].document == "Leave_Policy.docx"


def test_broad_leave_request_expands_query_and_keeps_relevant_sources(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from types import SimpleNamespace
    from langchain_core.documents import Document

    caplog.set_level("INFO")
    queries: list[str] = []
    results = {
        "tell me info regarding leaves": [
            (
                Document(
                    page_content="Overview of employee leave categories.",
                    metadata={"source": "HR_Policy_Manual_2024.pdf", "page": 1},
                ),
                1.4192,
            )
        ],
        "leave policy": [
            (
                Document(
                    page_content="Casual leave: 12 days per calendar year.",
                    metadata={"source": "Leave_Policy.docx", "section": "3.1"},
                ),
                0.954,
            ),
            (
                Document(
                    page_content="Overview of employee leave categories.",
                    metadata={"source": "HR_Policy_Manual_2024.pdf", "page": 1},
                ),
                0.954,
            ),
        ],
    }
    store = SimpleNamespace(
        _collection=SimpleNamespace(
            name="eka_hr",
            metadata={"hnsw:space": "l2"},
        ),
        similarity_search_with_score=lambda query, k: (
            queries.append(query) or results[query]
        ),
    )
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)

    chunks = pipeline.retrieve("hr", "tell me info regarding leaves")

    assert queries == ["tell me info regarding leaves", "leave policy"]
    assert {chunk.document for chunk in chunks} == {
        "HR_Policy_Manual_2024.pdf",
        "Leave_Policy.docx",
    }
    assert all(chunk.score >= pipeline.MIN_RELEVANCE_SCORE for chunk in chunks)
    assert "collection=eka_hr" in caplog.text
    assert "Leave_Policy.docx" in caplog.text
    assert "Casual leave: 12 days" not in caplog.text


def test_sick_leave_extension_retrieves_policy_and_documented_alternatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from langchain_core.documents import Document

    question = "what if i need more than 12 sick leaves, can i extend?"
    queries: list[str] = []
    results = {
        question: [
            (
                Document(
                    page_content="Employees receive 12 sick leaves per calendar year.",
                    metadata={"source": "HR_Policy_Manual_2024.pdf", "section": "1"},
                ),
                0.8,
            )
        ],
        "sick leave policy": [
            (
                Document(
                    page_content="Medical certificate required for 3+ consecutive sick-leave days.",
                    metadata={"source": "Leave_Policy.docx", "section": "3.2"},
                ),
                0.8,
            )
        ],
        "earned leave policy": [
            (
                Document(
                    page_content="15 earned leave days per calendar year.",
                    metadata={"source": "Leave_Policy.docx", "section": "3.3"},
                ),
                0.8,
            )
        ],
    }
    store = SimpleNamespace(
        _collection=SimpleNamespace(
            name="eka_hr",
            metadata={"hnsw:space": "l2"},
        ),
        similarity_search_with_score=lambda query, k: (
            queries.append(query) or results[query]
        ),
    )
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "get_vectorstore_for_agent",
        lambda agent, embeddings: store,
    )

    chunks = pipeline.retrieve("hr", question)

    assert queries == [question, "sick leave policy", "earned leave policy"]
    assert {chunk.document for chunk in chunks} == {
        "HR_Policy_Manual_2024.pdf",
        "Leave_Policy.docx",
    }
    assert any("Medical certificate" in chunk.snippet for chunk in chunks)
    assert any("earned leave days" in chunk.snippet for chunk in chunks)


def test_general_leave_extension_retrieves_earned_leave_alternative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from langchain_core.documents import Document

    question = "if i run out of 12 leaves can i extend more in some way?"
    queries: list[str] = []
    results = {
        question: [
            (
                Document(
                    page_content=(
                        "Employees are entitled to 12 Casual Leaves, 12 Sick Leaves, "
                        "and 15 Earned Leaves per calendar year."
                    ),
                    metadata={"source": "HR_Policy_Manual_2024.pdf", "section": "1"},
                ),
                0.8,
            )
        ],
        "leave entitlement": [
            (
                Document(
                    page_content="Casual and sick leaves cannot be carried forward.",
                    metadata={"source": "HR_Policy_Manual_2024.pdf", "section": "1"},
                ),
                0.8,
            )
        ],
        "earned leave policy": [
            (
                Document(
                    page_content=(
                        "15 earned leave days per year; up to 45 days can be carried forward."
                    ),
                    metadata={"source": "Leave_Policy.docx", "section": "3.3"},
                ),
                0.8,
            )
        ],
    }
    store = SimpleNamespace(
        _collection=SimpleNamespace(
            name="eka_hr",
            metadata={"hnsw:space": "l2"},
        ),
        similarity_search_with_score=lambda query, k: (
            queries.append(query) or results[query]
        ),
    )
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(
        pipeline,
        "get_vectorstore_for_agent",
        lambda agent, embeddings: store,
    )

    chunks = pipeline.retrieve("hr", question)

    assert queries == [question, "leave entitlement", "earned leave policy"]
    assert any("15 earned leave days" in chunk.snippet for chunk in chunks)


def test_unrelated_query_does_not_receive_leave_expansion_or_pass_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from langchain_core.documents import Document

    queries: list[str] = []
    store = SimpleNamespace(
        _collection=SimpleNamespace(
            name="eka_hr",
            metadata={"hnsw:space": "l2"},
        ),
        similarity_search_with_score=lambda query, k: (
            queries.append(query)
            or [
                (
                    Document(
                        page_content="Leave policy details.",
                        metadata={"source": "Leave_Policy.docx"},
                    ),
                    1.39,
                )
            ]
        ),
    )
    monkeypatch.setattr(pipeline, "get_embeddings", lambda: object())
    monkeypatch.setattr(pipeline, "get_vectorstore_for_agent", lambda agent, embeddings: store)

    chunks = pipeline.retrieve("hr", "What is quantum entanglement?")

    assert queries == ["What is quantum entanglement?"]
    assert chunks == []
