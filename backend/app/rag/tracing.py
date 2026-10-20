"""LangSmith tracing integration.

Wraps RAG calls so they're traceable in LangSmith (or Langfuse when configured).
When neither is configured, this becomes a no-op pass-through.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TraceSpan:
    """A simple trace span — also captures data for LangSmith export."""

    name: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        end = self.end_time or time.time()
        return int((end - self.start_time) * 1000)

    def finish(self, output: Any = None, error: str | None = None) -> None:
        self.end_time = time.time()
        if output is not None:
            self.outputs["result"] = output
        if error:
            self.error = error


def is_langsmith_enabled() -> bool:
    """True if LANGSMITH_API_KEY is set in environment."""
    return bool(os.getenv("LANGSMITH_API_KEY") or bool(os.getenv("LANGCHAIN_API_KEY")))


def langsmith_project() -> str:
    return os.getenv("LANGSMITH_PROJECT", "enterprise-knowledge-assistant")


@contextmanager
def trace(name: str, **inputs: Any) -> Iterator[TraceSpan]:
    """Context manager that records a trace span.

    Usage:
        with trace("rag.retrieve", agent_id="hr", question="...") as span:
            result = do_something()
            span.metadata["chunks_found"] = len(result)
    """
    span = TraceSpan(name=name, inputs=inputs)

    # If LangSmith is enabled, set env vars so LangChain auto-traces
    if is_langsmith_enabled():
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", langsmith_project())

    try:
        yield span
    except Exception as e:
        span.finish(error=str(e))
        raise
    else:
        span.finish()


def export_span(span: TraceSpan) -> dict:
    """Export a span as a dict (for logging or LangSmith client upload)."""
    return {
        "name": span.name,
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "duration_ms": span.duration_ms,
        "inputs": span.inputs,
        "outputs": span.outputs,
        "metadata": span.metadata,
        "error": span.error,
    }
