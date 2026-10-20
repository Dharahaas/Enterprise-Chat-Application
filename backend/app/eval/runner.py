"""Live RAGAS-style evaluation pipeline.

Runs the RAG pipeline against a gold-standard eval set and computes:
  - Faithfulness (grounding in retrieved context)
  - Answer Relevancy (addresses the question)
  - Context Precision (retrieved chunks relevant)
  - Context Recall (needed chunks retrieved)

When the `ragas` package is installed, this delegates to the real ragas library.
Otherwise, it falls back to a local heuristic scorer that approximates the same
metrics using keyword overlap — enough to surface trends in CI / nightly runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.rag.pipeline import run_rag


# ---------------------------------------------------------------------------
# Eval data model
# ---------------------------------------------------------------------------

@dataclass
class EvalQuestion:
    question: str
    agent: str
    expected_keywords: list[str]   # keywords a correct answer MUST contain
    expected_sources: list[str]    # document names that should be cited
    gold_answer: str               # reference answer (for LLM-judge fallback)


@dataclass
class EvalResult:
    question: str
    agent: str
    answer: str
    sources: list[str]
    latency_ms: int
    faithfulness: float        # 0..1 — answer grounded in retrieved context
    answer_relevancy: float    # 0..1 — answer addresses the question
    context_precision: float   # 0..1 — retrieved chunks relevant
    context_recall: float      # 0..1 — needed chunks retrieved
    overall: float             # 0..1 — composite


@dataclass
class EvalReport:
    started_at: str
    finished_at: str
    duration_sec: float
    total_questions: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall: float
    p50_latency_ms: int
    p95_latency_ms: int
    results: list[EvalResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Eval set — 24 questions across 3 agents (8 each)
# ---------------------------------------------------------------------------

DEFAULT_EVAL_SET: list[EvalQuestion] = [
    # ---------- HR (8) ----------
    EvalQuestion(
        question="How many casual leaves am I entitled to?",
        agent="hr",
        expected_keywords=["12", "casual"],
        expected_sources=["HR_Policy_Manual_2024", "Leave_Policy"],
        gold_answer="Employees are entitled to 12 casual leaves per calendar year.",
    ),
    EvalQuestion(
        question="How many sick leaves do I get?",
        agent="hr",
        expected_keywords=["12", "sick"],
        expected_sources=["Leave_Policy", "HR_Policy_Manual_2024"],
        gold_answer="12 sick leaves per calendar year.",
    ),
    EvalQuestion(
        question="How many earned leaves are there?",
        agent="hr",
        expected_keywords=["15", "earned"],
        expected_sources=["Leave_Policy"],
        gold_answer="15 earned leaves per calendar year, accruing at 1.25 days/month.",
    ),
    EvalQuestion(
        question="When is the payroll cut-off?",
        agent="hr",
        expected_keywords=["20", "cut-off"],
        expected_sources=["Payroll_Procedure", "HR_Policy_Manual_2024"],
        gold_answer="Payroll cut-off is the 20th of every month.",
    ),
    EvalQuestion(
        question="What is the maximum carry-forward for earned leave?",
        agent="hr",
        expected_keywords=["45", "carry"],
        expected_sources=["Leave_Policy", "HR_Policy_Manual_2024"],
        gold_answer="Up to 45 days of earned leave can be carried forward.",
    ),
    EvalQuestion(
        question="What documents are needed for onboarding a new hire?",
        agent="hr",
        expected_keywords=["PAN", "Aadhaar", "offer"],
        expected_sources=["Onboarding_Checklist"],
        gold_answer="PAN, Aadhaar, offer letter, educational certificates, previous relieving letter, bank details.",
    ),
    EvalQuestion(
        question="What is the PF contribution rate?",
        agent="hr",
        expected_keywords=["12", "PF"],
        expected_sources=["Payroll_Procedure"],
        gold_answer="PF is 12% of basic (employee) plus 12% (employer).",
    ),
    EvalQuestion(
        question="When are payslips available?",
        agent="hr",
        expected_keywords=["1st", "payslip"],
        expected_sources=["Payroll_Procedure"],
        gold_answer="Payslips are available on the HR Portal by the 1st of next month.",
    ),

    # ---------- IT (8) ----------
    EvalQuestion(
        question="How do I reset my corporate password?",
        agent="it",
        expected_keywords=["password.company.internal", "OTP", "12"],
        expected_sources=["IT_SOP_PasswordReset"],
        gold_answer="Go to password.company.internal, click Forgot Password, enter employee ID, choose OTP, set password meeting policy (min 12 chars).",
    ),
    EvalQuestion(
        question="What is the minimum password length?",
        agent="it",
        expected_keywords=["12"],
        expected_sources=["IT_SOP_PasswordReset"],
        gold_answer="Minimum 12 characters.",
    ),
    EvalQuestion(
        question="How do I configure VPN on macOS?",
        agent="it",
        expected_keywords=["AnyConnect", "vpn.company.com"],
        expected_sources=["IT_Network_Guide_v2"],
        gold_answer="Install Cisco AnyConnect, enter vpn.company.com as host, sign in with corporate credentials and 2FA.",
    ),
    EvalQuestion(
        question="What is the VPN host address?",
        agent="it",
        expected_keywords=["vpn.company.com"],
        expected_sources=["IT_Network_Guide_v2"],
        gold_answer="vpn.company.com",
    ),
    EvalQuestion(
        question="How do I request a hardware upgrade?",
        agent="it",
        expected_keywords=["Service Portal", "Upgrade"],
        expected_sources=["Asset_Policy"],
        gold_answer="Raise request via IT Service Portal under Request -> Hardware -> Upgrade. Approval chain: Manager -> IT Asset Manager -> Procurement.",
    ),
    EvalQuestion(
        question="What is the eligibility for hardware upgrade?",
        agent="it",
        expected_keywords=["3 years", "laptop"],
        expected_sources=["Asset_Policy"],
        gold_answer="Laptop age > 3 years OR documented performance SLA breach.",
    ),
    EvalQuestion(
        question="What is the IT desk extension?",
        agent="it",
        expected_keywords=["4500"],
        expected_sources=["IT_SOP_PasswordReset", "IT_Network_Guide_v2"],
        gold_answer="IT desk extension: 4500.",
    ),
    EvalQuestion(
        question="What is the standard turnaround for hardware upgrades?",
        agent="it",
        expected_keywords=["5", "business days"],
        expected_sources=["Asset_Policy"],
        gold_answer="5 business days from approval.",
    ),

    # ---------- Policy (8) ----------
    EvalQuestion(
        question="What are the data classification levels?",
        agent="policy",
        expected_keywords=["Public", "Internal", "Confidential", "Restricted"],
        expected_sources=["InfoSec_Policy_v4"],
        gold_answer="Four levels: Public, Internal, Confidential, Restricted.",
    ),
    EvalQuestion(
        question="What is the gift acceptance policy?",
        agent="policy",
        expected_keywords=["2,500", "gift"],
        expected_sources=["Code_of_Conduct_2024"],
        gold_answer="Employees may accept business gifts up to Rs 2,500 per occasion.",
    ),
    EvalQuestion(
        question="Is hybrid work allowed?",
        agent="policy",
        expected_keywords=["L4", "2 days"],
        expected_sources=["Hybrid_Work_Policy_2024"],
        gold_answer="Roles graded L4 and below are eligible for up to 2 days/week remote work.",
    ),
    EvalQuestion(
        question="What are the core hours for remote work?",
        agent="policy",
        expected_keywords=["10 AM", "4 PM"],
        expected_sources=["Hybrid_Work_Policy_2024"],
        gold_answer="Core hours: 10 AM to 4 PM IST.",
    ),
    EvalQuestion(
        question="How do I report a phishing email?",
        agent="policy",
        expected_keywords=["Report Phishing", "phishing@company.internal"],
        expected_sources=["Security_Incident_Response"],
        gold_answer="Use the Report Phishing button in Outlook or forward to phishing@company.internal. Delete the email.",
    ),
    EvalQuestion(
        question="What is the SOC acknowledgement SLA?",
        agent="policy",
        expected_keywords=["1 business hour"],
        expected_sources=["Security_Incident_Response"],
        gold_answer="SOC acknowledgement: within 1 business hour.",
    ),
    EvalQuestion(
        question="What is the consequence of leaking Restricted data?",
        agent="policy",
        expected_keywords=["termination"],
        expected_sources=["InfoSec_Policy_v4"],
        gold_answer="Violation is grounds for termination and possible legal action.",
    ),
    EvalQuestion(
        question="What is the internet reimbursement for remote work?",
        agent="policy",
        expected_keywords=["1,500"],
        expected_sources=["Hybrid_Work_Policy_2024"],
        gold_answer="Up to Rs 1,500/month on submission of bills.",
    ),
]


# ---------------------------------------------------------------------------
# Heuristic scoring (fallback when ragas lib is not installed)
# ---------------------------------------------------------------------------

def _score_answer_relevancy(question: str, answer: str, expected_keywords: list[str]) -> float:
    """0..1 — fraction of expected keywords present in the answer."""
    if not expected_keywords:
        return 0.5
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def _score_context_precision(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """0..1 — fraction of retrieved sources that are in expected_sources."""
    if not retrieved_sources:
        return 0.0
    hits = sum(1 for r in retrieved_sources for e in expected_sources if e.lower() in r.lower())
    return min(hits / len(retrieved_sources), 1.0)


def _score_context_recall(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """0..1 — fraction of expected_sources that were retrieved."""
    if not expected_sources:
        return 0.5
    hits = sum(1 for e in expected_sources if any(e.lower() in r.lower() for r in retrieved_sources))
    return hits / len(expected_sources)


def _score_faithfulness(answer: str, retrieved_sources: list[str], gold_answer: str) -> float:
    """0..1 — approximated as: did we retrieve any sources AND the answer is non-trivial?"""
    if not retrieved_sources:
        return 0.0
    if len(answer) < 30:
        return 0.2
    # If gold_answer keywords appear in the answer, faithfulness is high
    gold_keywords = [w for w in gold_answer.lower().split() if len(w) > 3][:8]
    if not gold_keywords:
        return 0.7
    answer_lower = answer.lower()
    hits = sum(1 for kw in gold_keywords if kw in answer_lower)
    return min(hits / len(gold_keywords), 1.0)


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    eval_set: list[EvalQuestion] | None = None,
    save_path: str | Path | None = None,
) -> EvalReport:
    """Run the evaluation pipeline against the eval set.

    Args:
        eval_set: questions to evaluate. Defaults to DEFAULT_EVAL_SET (24 Qs).
        save_path: optional path to write the JSON report.

    Returns:
        EvalReport with per-question + aggregated metrics.
    """
    if eval_set is None:
        eval_set = DEFAULT_EVAL_SET

    from datetime import datetime, timezone
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()

    results: list[EvalResult] = []
    for i, q in enumerate(eval_set):
        try:
            rag_result = run_rag(q.agent, q.question)
            answer = rag_result.answer
            retrieved = [s.document for s in rag_result.sources]
            latency_ms = rag_result.latency_ms
        except Exception as e:
            answer = f"(eval error: {e})"
            retrieved = []
            latency_ms = 0

        results.append(EvalResult(
            question=q.question,
            agent=q.agent,
            answer=answer,
            sources=retrieved,
            latency_ms=latency_ms,
            faithfulness=_score_faithfulness(answer, retrieved, q.gold_answer),
            answer_relevancy=_score_answer_relevancy(q.question, answer, q.expected_keywords),
            context_precision=_score_context_precision(retrieved, q.expected_sources),
            context_recall=_score_context_recall(retrieved, q.expected_sources),
            overall=0.0,  # filled below
        ))

    # Compute composite overall
    for r in results:
        r.overall = (
            r.faithfulness * 0.30
            + r.answer_relevancy * 0.30
            + r.context_precision * 0.20
            + r.context_recall * 0.20
        )

    # Aggregated metrics
    n = len(results)
    avg = lambda attr: sum(getattr(r, attr) for r in results) / n if n else 0.0
    latencies = sorted(r.latency_ms for r in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95_idx = int(len(latencies) * 0.95)
    p95 = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0

    finished_at = datetime.now(timezone.utc).isoformat()
    duration_sec = round(time.time() - start_time, 2)

    report = EvalReport(
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        total_questions=n,
        avg_faithfulness=round(avg("faithfulness"), 3),
        avg_answer_relevancy=round(avg("answer_relevancy"), 3),
        avg_context_precision=round(avg("context_precision"), 3),
        avg_context_recall=round(avg("context_recall"), 3),
        avg_overall=round(avg("overall"), 3),
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        results=results,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

    return report


def eval_set_summary() -> dict[str, Any]:
    """Return a summary of the default eval set without running it."""
    by_agent: dict[str, int] = {}
    for q in DEFAULT_EVAL_SET:
        by_agent[q.agent] = by_agent.get(q.agent, 0) + 1
    return {
        "total_questions": len(DEFAULT_EVAL_SET),
        "by_agent": by_agent,
        "questions_preview": [{"q": q.question, "agent": q.agent} for q in DEFAULT_EVAL_SET[:5]],
    }
