"""RAG subpackage."""

from app.rag.embeddings import embed_query, embed_texts, get_embeddings
from app.rag.llm import get_llm, llm_provider_name
from app.rag.pipeline import RagResult, run_rag
from app.rag.vectorstore import count_chunks, delete_document, list_documents_in_collection

__all__ = [
    "RagResult",
    "run_rag",
    "get_embeddings",
    "embed_query",
    "embed_texts",
    "get_llm",
    "llm_provider_name",
    "count_chunks",
    "delete_document",
    "list_documents_in_collection",
]
