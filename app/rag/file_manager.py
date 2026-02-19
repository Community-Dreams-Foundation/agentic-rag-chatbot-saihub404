"""
File Manager
============
CRUD operations on the RAG knowledge base:
  - list_sources()      list all indexed files with chunk counts
  - inspect_source()    show all chunks for a given file
  - delete_source()     remove all chunks for a file from ChromaDB
  - reindex_source()    delete + re-ingest a file
  - clear_all()         wipe the entire knowledge base
  - chunk_stats()       summary statistics across all indexed content
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from app.rag.ingestion import get_collection, ingest_file


# ── List ─────────────────────────────────────────────────────────────────

def list_sources() -> List[Dict[str, Any]]:
    """
    Return all indexed sources with chunk counts and char totals.
    Sorted alphabetically by filename.
    """
    collection = get_collection()
    data = collection.get(include=["metadatas", "documents"])

    if not data["metadatas"]:
        return []

    source_map: Dict[str, Dict] = {}
    for meta, doc in zip(data["metadatas"], data["documents"]):
        src = meta["source"]
        if src not in source_map:
            source_map[src] = {"source": src, "chunks": 0, "total_chars": 0}
        source_map[src]["chunks"] += 1
        source_map[src]["total_chars"] += len(doc)

    return sorted(source_map.values(), key=lambda x: x["source"])


# ── Inspect ───────────────────────────────────────────────────────────────

def inspect_source(filename: str) -> List[Dict[str, Any]]:
    """
    Return all chunks for a given source file, sorted by chunk_index.
    Each item: {chunk_index, id, text, char_count}
    """
    collection = get_collection()
    data = collection.get(
        where={"source": filename},
        include=["documents", "metadatas", "embeddings"],
    )

    if not data["ids"]:
        return []

    chunks = []
    for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        chunks.append({
            "chunk_index": meta.get("chunk_index", -1),
            "id":          id_,
            "text":        doc,
            "char_count":  len(doc),
        })

    return sorted(chunks, key=lambda x: x["chunk_index"])


# ── Delete ────────────────────────────────────────────────────────────────

def delete_source(filename: str) -> Dict[str, Any]:
    """
    Remove all chunks for a given source file from ChromaDB.
    Returns: {deleted, source}
    """
    collection = get_collection()

    # Find all IDs for this source
    data = collection.get(where={"source": filename}, include=[])
    ids = data["ids"]

    if not ids:
        return {"deleted": 0, "source": filename, "error": "Source not found in index"}

    collection.delete(ids=ids)
    return {"deleted": len(ids), "source": filename}


# ── Re-index ──────────────────────────────────────────────────────────────

def reindex_source(file_path: str | Path) -> Dict[str, Any]:
    """
    Delete existing chunks for a file then re-ingest it fresh.
    Useful after editing a document.
    Returns combined result: {source, deleted, total_chunks, new_chunks}
    """
    path = Path(file_path)
    delete_result = delete_source(path.name)
    ingest_result = ingest_file(path)

    return {
        "source":       path.name,
        "deleted":      delete_result.get("deleted", 0),
        "total_chunks": ingest_result["total_chunks"],
        "new_chunks":   ingest_result["new_chunks"],
        "total_chars":  ingest_result["total_chars"],
    }


# ── Clear all ─────────────────────────────────────────────────────────────

def clear_all() -> Dict[str, Any]:
    """
    Delete every chunk from the knowledge base.
    Returns: {deleted}
    """
    collection = get_collection()
    data = collection.get(include=[])
    ids = data["ids"]

    if ids:
        collection.delete(ids=ids)

    return {"deleted": len(ids)}


# ── Stats ─────────────────────────────────────────────────────────────────

def chunk_stats() -> Dict[str, Any]:
    """
    Summary statistics across the entire knowledge base.
    """
    collection = get_collection()
    data = collection.get(include=["documents", "metadatas"])

    if not data["documents"]:
        return {
            "total_chunks": 0,
            "total_sources": 0,
            "total_chars": 0,
            "avg_chunk_chars": 0,
            "sources": [],
        }

    docs = data["documents"]
    metas = data["metadatas"]

    total_chars = sum(len(d) for d in docs)
    sources = set(m["source"] for m in metas)

    return {
        "total_chunks":   len(docs),
        "total_sources":  len(sources),
        "total_chars":    total_chars,
        "avg_chunk_chars": round(total_chars / len(docs)) if docs else 0,
        "sources":        sorted(sources),
    }
