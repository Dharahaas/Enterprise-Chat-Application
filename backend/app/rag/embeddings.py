"""Embeddings — use Chroma's CPU-friendly ONNX MiniLM implementation."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


class ChromaONNXEmbeddings(Embeddings):
    """Adapt Chroma's default ONNX embedding function to LangChain's interface."""

    def __init__(self) -> None:
        self._embedding_function = ONNXMiniLM_L6_V2(
            preferred_providers=["CPUExecutionProvider"]
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        vectors = [
            [float(value) for value in vector]
            for vector in self._embedding_function(texts)
        ]
        logger.info(
            "Embedding complete provider=chroma_onnx items=%d elapsed_ms=%d",
            len(texts),
            int((time.perf_counter() - start) * 1000),
        )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached, lazily loaded ONNX embedding adapter."""
    if settings.embedding_model not in {
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    }:
        raise ValueError(
            "Chroma ONNX embeddings support only all-MiniLM-L6-v2; "
            f"configured model is {settings.embedding_model!r}"
        )
    return ChromaONNXEmbeddings()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return list(get_embeddings().embed_query(query))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    emb = get_embeddings()
    return [list(v) for v in emb.embed_documents(texts)]


def fake_embedding_dim() -> int:
    """Return the configured embedding dimension."""
    return settings.embedding_dim
