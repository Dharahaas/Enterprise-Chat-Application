"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------- Chat ----------

class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)
    agent: Literal["hr", "it", "policy"] | None = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    agent: Literal["hr", "it", "policy"] | None = None
    conversation_id: str | None = None
    conversation_history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        max_length=6,
    )


class SourceChunk(BaseModel):
    document: str
    page: int | None = None
    section: str | None = None
    score: float
    snippet: str
    agent: Literal["hr", "it", "policy"] | None = None


class ChatResponse(BaseModel):
    answer: str
    agent: str
    sources: list[SourceChunk]
    latency_ms: int
    llm_provider: str
    error_code: str | None = None
    conversation_id: str
    timestamp: str


# ---------- Documents ----------

class DocumentInfo(BaseModel):
    id: str
    name: str
    agent: Literal["hr", "it", "policy"]
    size_bytes: int
    chunks: int
    status: Literal["indexed", "processing", "failed"]
    uploaded_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int
    by_agent: dict[str, int]


class DocumentUploadResponse(BaseModel):
    document: DocumentInfo
    message: str


# ---------- Feedback (Phase 4) ----------

class FeedbackRequest(BaseModel):
    rating: Literal["positive", "negative"]
    # Stable, client-derived key identifying the answer being rated. It never
    # carries conversation content beyond what is validated below.
    answer_key: str = Field(..., min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    question: str = Field(..., min_length=1, max_length=300)
    agent: Literal["hr", "it", "policy", "auto"] = "auto"
    reason: (
        Literal[
            "incorrect_answer",
            "missing_information",
            "irrelevant_sources",
            "outdated_document",
            "other",
        ]
        | None
    ) = None
    source_status: Literal["sources", "no_sources", "retrieval_error", "error"] = "sources"


class FeedbackResponse(BaseModel):
    id: str
    rating: str
    reason: str | None
    changed: bool
    updated_at: str


class KnowledgeGap(BaseModel):
    question: str
    agent: str
    count: int
    top_reason: str
    reasons: dict[str, int]


class FeedbackSummary(BaseModel):
    total_records: int
    total_positive: int
    total_negative: int
    negative_reasons: dict[str, int]
    feedback_by_agent: dict[str, dict[str, int]]
    feedback_by_day: dict[str, dict[str, int]]
    knowledge_gaps: list[KnowledgeGap]
    no_source_questions: int
    unanswered_questions: int


# ---------- Analytics ----------

class AnalyticsResponse(BaseModel):
    accuracy: float | None
    retrieval_recall_at_5: float | None
    p50_latency_ms: int | None
    p95_latency_ms: int | None
    queries_this_week: int
    unique_users: int | None
    queries_by_agent: dict[str, int]
    queries_by_day: dict[str, int]
    ragas_metrics: dict[str, dict[str, float]]
    feedback: FeedbackSummary | None = None


# ---------- Health ----------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    # Kept for clients of the original API contract; use llm_enabled instead.
    openai_enabled: bool
    llm_enabled: bool
    llm_provider: str = "unconfigured"
    vector_db_path: str
    agents: list[str]
    collections: dict[str, str]
    readiness_issues: list[str] = Field(default_factory=list)
