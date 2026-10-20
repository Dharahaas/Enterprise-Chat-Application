"""Evaluation subpackage — RAGAS-style metrics + eval runner."""

from app.eval.runner import (
    DEFAULT_EVAL_SET,
    EvalQuestion,
    EvalReport,
    EvalResult,
    eval_set_summary,
    run_eval,
)

__all__ = [
    "DEFAULT_EVAL_SET",
    "EvalQuestion",
    "EvalReport",
    "EvalResult",
    "run_eval",
    "eval_set_summary",
]
