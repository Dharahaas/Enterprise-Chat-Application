"""Document loaders — PDF, DOCX, TXT, MD."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


def load_pdf(path: Path) -> list[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        return _fallback_text(path)
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    # PyPDFLoader already attaches page numbers in metadata
    return docs


def load_docx(path: Path) -> list[Document]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
    except ImportError:
        return _fallback_text(path)
    loader = Docx2txtLoader(str(path))
    return loader.load()


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def load_markdown(path: Path) -> list[Document]:
    # Treat MD as text for simplicity
    return load_text(path)


def load_document(path: Path) -> list[Document]:
    """Auto-detect file type and load accordingly."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".docx":
        return load_docx(path)
    elif ext in (".txt", ".text"):
        return load_text(path)
    elif ext in (".md", ".markdown"):
        return load_markdown(path)
    else:
        # Try as text
        return load_text(path)


def _fallback_text(path: Path) -> list[Document]:
    """Last-resort loader that just reads the file as text."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = f"(unable to read {path.name})"
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]
