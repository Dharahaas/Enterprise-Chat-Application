"""Vector store — Chroma (local file-based, no signup needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import agent_collections, settings


def _persist_dir() -> Path:
    p = Path(settings.vector_db_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_vectorstore_for_agent(agent_id: str, embeddings: Embeddings) -> Chroma:
    """Return the Chroma vector store for the given agent's namespace."""
    collection_name = agent_collections().get(agent_id)
    if not collection_name:
        raise ValueError(f"No collection mapping for agent: {agent_id}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(_persist_dir()),
    )


def count_chunks(agent_id: str) -> int:
    """Return the number of chunks indexed for an agent."""
    try:
        # Use the underlying client to avoid needing an embeddings instance
        import chromadb

        client = chromadb.PersistentClient(path=str(_persist_dir()))
        col = client.get_collection(agent_collections()[agent_id])
        return col.count()
    except Exception:
        return 0


def list_documents_in_collection(agent_id: str) -> list[dict[str, Any]]:
    """Return metadata for all unique documents in an agent's collection."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_persist_dir()))
        col = client.get_collection(agent_collections()[agent_id])
        result = col.get(include=["metadatas"])
        docs_by_name: dict[str, dict[str, Any]] = {}
        for meta in result.get("metadatas", []):
            if not meta:
                continue
            name = meta.get("source", "unknown")
            if name not in docs_by_name:
                docs_by_name[name] = {
                    "name": name,
                    "agent": agent_id,
                    "size_bytes": int(meta.get("size_bytes", 0)),
                    "chunks": 0,
                    "uploaded_at": meta.get("uploaded_at", "unknown"),
                }
            docs_by_name[name]["chunks"] += 1
        return list(docs_by_name.values())
    except Exception:
        return []


def delete_document(agent_id: str, doc_name: str) -> int:
    """Delete all chunks for a document. Returns count deleted."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(_persist_dir()))
        col = client.get_collection(agent_collections()[agent_id])
        before = col.count()
        col.delete(where={"source": doc_name})
        after = col.count()
        return before - after
    except Exception:
        return 0
