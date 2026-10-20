"""Document ingestion — load → chunk → embed → upsert into vector DB."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.ingestion.chunker import chunk_documents
from app.ingestion.loader import load_document
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import get_vectorstore_for_agent


def ingest_file(file_path: Path, agent_id: str) -> dict[str, Any]:
    """Ingest a single file into an agent's vector namespace.

    Returns a dict with metadata about the ingestion.
    """
    start = time.time()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Load
    docs = load_document(file_path)
    if not docs:
        raise ValueError(f"No content extracted from {file_path.name}")

    size_bytes = file_path.stat().st_size
    uploaded_at = datetime.now(timezone.utc).isoformat()

    # 2. Chunk
    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError(f"Chunking produced 0 chunks for {file_path.name}")

    # 3. Enrich metadata
    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.setdefault("source", file_path.name)
        c.metadata["agent"] = agent_id
        c.metadata["size_bytes"] = size_bytes
        c.metadata["uploaded_at"] = uploaded_at
        c.metadata["doc_id"] = str(uuid.uuid4())

    # 4. Embed + upsert
    embeddings = get_embeddings()
    vs = get_vectorstore_for_agent(agent_id, embeddings)
    try:
        vs.add_documents(chunks)
    except Exception:
        # Some versions of langchain-chroma need explicit persist
        try:
            vs.persist()  # type: ignore[attr-defined]
        except Exception:
            pass

    elapsed = time.time() - start
    return {
        "name": file_path.name,
        "agent": agent_id,
        "size_bytes": size_bytes,
        "chunks": len(chunks),
        "status": "indexed",
        "uploaded_at": uploaded_at,
        "elapsed_sec": round(elapsed, 2),
    }


def ingest_text(
    text: str,
    name: str,
    agent_id: str,
    section: str | None = None,
) -> dict[str, Any]:
    """Ingest raw text content (used for seeding sample docs)."""
    docs = [Document(page_content=text, metadata={"source": name, "section": section or "1"})]
    return _ingest_docs(docs, name, agent_id, size_bytes=len(text.encode("utf-8")))


def _ingest_docs(
    docs: list[Document], name: str, agent_id: str, size_bytes: int
) -> dict[str, Any]:
    """Internal helper — chunk + embed + upsert already-loaded Documents."""
    start = time.time()
    chunks = chunk_documents(docs)
    uploaded_at = datetime.now(timezone.utc).isoformat()

    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.setdefault("source", name)
        c.metadata["agent"] = agent_id
        c.metadata["size_bytes"] = size_bytes
        c.metadata["uploaded_at"] = uploaded_at
        c.metadata["doc_id"] = str(uuid.uuid4())

    embeddings = get_embeddings()
    vs = get_vectorstore_for_agent(agent_id, embeddings)
    vs.add_documents(chunks)

    return {
        "name": name,
        "agent": agent_id,
        "size_bytes": size_bytes,
        "chunks": len(chunks),
        "status": "indexed",
        "uploaded_at": uploaded_at,
        "elapsed_sec": round(time.time() - start, 2),
    }


def seed_sample_docs(force: bool = False) -> list[dict[str, Any]]:
    """Seed the vector DB with sample documents if empty.

    Args:
        force: if True, re-seed even if collections already have content.
    """
    from app.rag.vectorstore import count_chunks

    already_seeded = sum(count_chunks(a) for a in ["hr", "it", "policy"]) > 0
    if already_seeded and not force:
        return [{"message": "Already seeded. Pass force=True to re-seed."}]

    from app.data.sample_docs_seed import SAMPLE_DOCS

    results = []
    for doc in SAMPLE_DOCS:
        try:
            r = ingest_text(
                text=doc["content"],
                name=doc["name"],
                agent_id=doc["agent"],
                section=doc.get("section"),
            )
            results.append(r)
        except Exception as e:
            results.append({"name": doc["name"], "error": str(e)})
    return results
