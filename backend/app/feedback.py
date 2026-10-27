"""Durable answer-feedback storage and aggregation (Phase 4).

Storage reuses the application's existing SQLite database
(``settings.auth_database_path``): an additional ``feedback`` table is created
with ``CREATE TABLE IF NOT EXISTS``. Nothing existing is dropped, rebuilt or
migrated.

One record per (user, answer): resubmitting for the same answer updates the
existing row instead of duplicating it, so a rating can be changed safely.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from app.config import settings

RATINGS = ("positive", "negative")
NEGATIVE_REASONS = (
    "incorrect_answer",
    "missing_information",
    "irrelevant_sources",
    "outdated_document",
    "other",
)
SOURCE_STATUSES = ("sources", "no_sources", "retrieval_error", "error")
TOPICS = ("hr", "it", "policy", "auto")

_MAX_QUESTION_CHARS = 300

_store_lock = Lock()
_ready_for_path: str | None = None


def _connect() -> sqlite3.Connection:
    database_path = Path(settings.auth_database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_feedback_store() -> None:
    """Create the feedback table if it does not exist yet (idempotent)."""
    global _ready_for_path
    path = settings.auth_database_path
    with _store_lock:
        if _ready_for_path == path:
            return
        with _connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    answer_key TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('positive', 'negative')),
                    reason TEXT CHECK (
                        reason IN (
                            'incorrect_answer', 'missing_information',
                            'irrelevant_sources', 'outdated_document', 'other'
                        )
                    ),
                    agent TEXT NOT NULL,
                    question TEXT NOT NULL,
                    source_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (username, answer_key)
                );
                CREATE INDEX IF NOT EXISTS feedback_created_idx ON feedback(created_at);
                CREATE INDEX IF NOT EXISTS feedback_agent_idx ON feedback(agent);
                """
            )
        _ready_for_path = path


def _row_to_record(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "rating": row["rating"],
        "reason": row["reason"],
        "agent": row["agent"],
        "question": row["question"],
        "source_status": row["source_status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def record_feedback(
    *,
    username: str,
    answer_key: str,
    rating: str,
    reason: str | None,
    agent: str,
    question: str,
    source_status: str,
) -> tuple[dict, bool]:
    """Store feedback for one answer.

    Returns ``(record, changed)`` where ``changed`` reports whether the rating
    or reason differs from what was previously stored for this answer.
    """
    ensure_feedback_store()
    question = question.strip()[:_MAX_QUESTION_CHARS]
    if not question:
        raise ValueError("A question is required.")
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        existing = connection.execute(
            "SELECT rating, reason FROM feedback WHERE username = ? AND answer_key = ?",
            (username, answer_key),
        ).fetchone()
        if existing is None:
            feedback_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO feedback (id, username, answer_key, rating, reason, agent, "
                "question, source_status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id, username, answer_key, rating, reason, agent,
                    question, source_status, now, now,
                ),
            )
            changed = True
        else:
            feedback_id = connection.execute(
                "SELECT id FROM feedback WHERE username = ? AND answer_key = ?",
                (username, answer_key),
            ).fetchone()["id"]
            connection.execute(
                "UPDATE feedback SET rating = ?, reason = ?, agent = ?, question = ?, "
                "source_status = ?, updated_at = ? WHERE username = ? AND answer_key = ?",
                (
                    rating, reason, agent, question, source_status, now,
                    username, answer_key,
                ),
            )
            changed = (
                existing["rating"] != rating
                or (existing["reason"] or None) != reason
            )
        row = connection.execute(
            "SELECT * FROM feedback WHERE id = ?", (feedback_id,)
        ).fetchone()
    return _row_to_record(row), changed


def list_feedback_for_user(username: str) -> list[dict]:
    """Return one record per answer for a single user (never used in reports)."""
    ensure_feedback_store()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM feedback WHERE username = ? ORDER BY updated_at DESC",
            (username,),
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def aggregate_feedback() -> dict:
    """Aggregate feedback for the administrator Analytics report.

    Returns counts only plus de-identified knowledge-gap questions: no
    usernames, no assistant answers and no conversation history are included.
    """
    ensure_feedback_store()
    with _connect() as connection:
        rows = connection.execute(
            "SELECT rating, reason, agent, question, source_status, created_at "
            "FROM feedback"
        ).fetchall()

    total_positive = 0
    total_negative = 0
    negative_reasons: dict[str, int] = {}
    by_agent: dict[str, dict[str, int]] = {}
    by_day: dict[str, dict[str, int]] = {}
    gaps: dict[tuple[str, str], dict] = {}

    for row in rows:
        rating = row["rating"]
        agent = row["agent"] if row["agent"] in ("hr", "it", "policy") else "auto"
        day = (row["created_at"] or "")[:10]

        if rating == "positive":
            total_positive += 1
        else:
            total_negative += 1
            reason = row["reason"] or "unspecified"
            negative_reasons[reason] = negative_reasons.get(reason, 0) + 1

        bucket = by_agent.setdefault(agent, {"positive": 0, "negative": 0})
        bucket[rating] = bucket.get(rating, 0) + 1

        day_bucket = by_day.setdefault(day, {"positive": 0, "negative": 0})
        day_bucket[rating] = day_bucket.get(rating, 0) + 1

        if rating == "negative":
            question = (row["question"] or "").strip()
            if question:
                key = (agent, question)
                gap = gaps.setdefault(
                    key,
                    {
                        "question": question,
                        "agent": agent,
                        "count": 0,
                        "reasons": {},
                    },
                )
                gap["count"] += 1
                reason = row["reason"] or "unspecified"
                gap["reasons"][reason] = gap["reasons"].get(reason, 0) + 1

    ranked_gaps = sorted(
        gaps.values(),
        key=lambda item: item["count"],
        reverse=True,
    )[:5]
    for gap in ranked_gaps:
        gap["top_reason"] = max(gap["reasons"], key=gap["reasons"].get)

    return {
        "total_records": len(rows),
        "total_positive": total_positive,
        "total_negative": total_negative,
        "negative_reasons": dict(
            sorted(negative_reasons.items(), key=lambda item: item[1], reverse=True)
        ),
        "feedback_by_agent": by_agent,
        "feedback_by_day": dict(sorted(by_day.items())),
        "knowledge_gaps": ranked_gaps,
    }