"""Text chunker — wraps LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document

from app.config import settings


def make_splitter():
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    """Split a list of loaded Documents into smaller chunks."""
    splitter = make_splitter()
    chunks = splitter.split_documents(list(docs))
    # Ensure every chunk carries the source name
    for c in chunks:
        if "source" not in (c.metadata or {}):
            c.metadata = c.metadata or {}
            c.metadata["source"] = "unknown"
    return chunks
