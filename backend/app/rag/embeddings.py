"""Embeddings — uses HuggingFace sentence-transformers locally (no API key)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.embeddings import Embeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embeddings instance.

    Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2 by default) which:
    - Runs locally, no API key needed
    - Produces 384-dim vectors
    - ~80MB model size, fast on CPU
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fallback for older langchain-community
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return list(get_embeddings().embed_query(query))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    emb = get_embeddings()
    return [list(v) for v in emb.embed_documents(texts)]


def fake_embedding_dim() -> int:
    """Stand-in dimension used if real embeddings fail to load."""
    return settings.embedding_dim
