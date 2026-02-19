"""
RAG Ingestion Pipeline
======================
Steps:
  1. Parse  – extract raw text from PDF / HTML / TXT
  2. Chunk  – split into overlapping windows with metadata
  3. Embed  – sentence-transformers (local, free)
  4. Index  – store in ChromaDB

Supports: .pdf, .html, .htm, .txt, .md
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# ── Singleton embedding model (lazy load) ──────────────────────────────────
_embed_model: SentenceTransformer | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


# ── ChromaDB client (lazy) ─────────────────────────────────────────────────
_chroma_client: chromadb.PersistentClient | None = None
_collection = None


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Text parsers ──────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages)


def _parse_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _parse_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_file(path: Path) -> str:
    """Route to correct parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    elif suffix in (".html", ".htm"):
        return _parse_html(path)
    else:
        return _parse_text(path)


# ── Chunker ───────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    Tries to respect paragraph boundaries first, then falls back to
    character-level sliding window.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict[str, Any]] = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(_make_chunk(current, source, chunk_idx))
                chunk_idx += 1
                # carry overlap
                words = current.split()
                overlap_words = words[-overlap // 6 :] if overlap else []
                current = " ".join(overlap_words) + ("\n\n" if overlap_words else "") + para
            else:
                # Single paragraph larger than chunk_size → hard split
                for start in range(0, len(para), chunk_size - overlap):
                    snippet = para[start : start + chunk_size]
                    chunks.append(_make_chunk(snippet, source, chunk_idx))
                    chunk_idx += 1
                current = ""

    if current.strip():
        chunks.append(_make_chunk(current, source, chunk_idx))

    return chunks


def _make_chunk(text: str, source: str, idx: int) -> Dict[str, Any]:
    uid = hashlib.md5(f"{source}-{idx}-{text[:40]}".encode()).hexdigest()
    return {
        "id": uid,
        "text": text.strip(),
        "source": source,
        "chunk_index": idx,
    }


# ── Indexer ───────────────────────────────────────────────────────────────

def index_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and store chunks in ChromaDB. Returns number of new chunks added."""
    if not chunks:
        return 0

    model = get_embed_model()
    collection = get_collection()

    texts      = [c["text"]    for c in chunks]
    ids        = [c["id"]      for c in chunks]
    metadatas  = [
        {"source": c["source"], "chunk_index": c["chunk_index"]}
        for c in chunks
    ]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # Upsert (skip duplicates)
    existing = set(collection.get(ids=ids)["ids"])
    new_chunks = [
        (id_, emb, meta, txt)
        for id_, emb, meta, txt in zip(ids, embeddings, metadatas, texts)
        if id_ not in existing
    ]

    if new_chunks:
        n_ids, n_embs, n_metas, n_docs = zip(*new_chunks)
        collection.add(
            ids=list(n_ids),
            embeddings=list(n_embs),
            metadatas=list(n_metas),
            documents=list(n_docs),
        )

    return len(new_chunks)


# ── Public API ────────────────────────────────────────────────────────────

def ingest_file(path: str | Path, original_name: str | None = None) -> Dict[str, Any]:
    """
    Full pipeline: parse → chunk → embed → index.
    Returns a summary dict.
    original_name: use this as the source label (for uploaded temp files).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    source_name = original_name or path.name
    raw_text = parse_file(path)
    chunks   = chunk_text(raw_text, source=source_name)
    added    = index_chunks(chunks)

    return {
        "file":         source_name,
        "total_chars":  len(raw_text),
        "total_chunks": len(chunks),
        "new_chunks":   added,
    }


def list_indexed_sources() -> List[str]:
    """Return unique source filenames currently in the index."""
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    sources = {m["source"] for m in result["metadatas"]} if result["metadatas"] else set()
    return sorted(sources)


def get_chunk_count() -> int:
    return get_collection().count()
