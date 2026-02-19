"""
Hybrid Retrieval
================
Combines:
  • Dense retrieval  – ChromaDB cosine similarity
  • Sparse retrieval – BM25 keyword matching (rank_bm25)

Results are fused with Reciprocal Rank Fusion (RRF) for better coverage.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import TOP_K
from app.rag.ingestion import get_collection, get_embed_model


# ── Dense retrieval ────────────────────────────────────────────────────────

# Minimum cosine similarity for a dense result to be considered relevant.
# Cosine similarity of 1.0 = identical vectors; 0.0 = completely orthogonal.
# Documents from a completely different domain typically score < 0.25.
_MIN_COSINE_SIM = 0.30

def dense_search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Search ChromaDB by embedding similarity."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embed_model()
    q_emb = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(k * 2, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - dist  # convert cosine distance → similarity
        if similarity < _MIN_COSINE_SIM:
            continue   # skip off-topic chunks early
        hits.append({
            "text":        doc,
            "source":      meta["source"],
            "chunk_index": meta["chunk_index"],
            "score":       similarity,
        })
    return hits


# ── Sparse retrieval (BM25) ────────────────────────────────────────────────

def bm25_search(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Fetch all docs from ChromaDB and run BM25 in-memory."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_data = collection.get(include=["documents", "metadatas"])
    docs      = all_data["documents"]
    metas     = all_data["metadatas"]

    if not docs:
        return []

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    top_indices = np.argsort(scores)[::-1][: k * 2]

    hits = []
    for idx in top_indices:
        if scores[idx] > 0:
            hits.append({
                "text":        docs[idx],
                "source":      metas[idx]["source"],
                "chunk_index": metas[idx]["chunk_index"],
                "score":       float(scores[idx]),
            })
    return hits


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────

def rrf_fuse(
    dense_hits: List[Dict],
    sparse_hits: List[Dict],
    k: int = 60,
    top_n: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Fuse two ranked lists with RRF.
    k=60 is standard; higher k reduces impact of high ranks.
    """
    scores: Dict[str, float] = {}
    chunks: Dict[str, Dict]  = {}

    for rank, hit in enumerate(dense_hits):
        key = f"{hit['source']}::{hit['chunk_index']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks[key] = hit

    for rank, hit in enumerate(sparse_hits):
        key = f"{hit['source']}::{hit['chunk_index']}"
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in chunks:
            chunks[key] = hit

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Attach the RRF fusion score to each chunk for downstream threshold filtering
    result = []
    for key, rrf_score in ranked:
        chunk = dict(chunks[key])
        chunk["rrf_score"] = rrf_score
        result.append(chunk)
    return result


# ── Relevance threshold filter ─────────────────────────────────────────────

# Minimum RRF score to include a chunk in the LLM context.
# RRF score for top-1 of N results ≈ 1/(61) ≈ 0.016.
# Raising this above 0.005 cuts clearly off-topic chunks that happen to rank in the
# top-K only because there is nothing better in the collection.
_MIN_RRF_SCORE = 0.008   # ~= RRF score of a top-3 match; drops weak retrievals


def _filter_by_relevance(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop chunks whose RRF fusion score is below the minimum threshold.
    This prevents completely off-topic chunks from polluting the LLM context
    and causing hallucinated or misleading answers.
    """
    return [c for c in chunks if c.get("rrf_score", 1.0) >= _MIN_RRF_SCORE]


# ── Public API ────────────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Hybrid BM25 + dense retrieval with RRF fusion.
    Results are filtered by a minimum relevance threshold to prevent
    off-topic chunks from polluting the LLM context.
    """
    dense  = dense_search(query, k=top_k)
    sparse = bm25_search(query, k=top_k)
    fused  = rrf_fuse(dense, sparse, top_n=top_k)
    return _filter_by_relevance(fused)


# ── Citation formatter ────────────────────────────────────────────────────

def format_citations(chunks: List[Dict[str, Any]]) -> List[str]:
    """Return human-readable citation strings."""
    seen = set()
    citations = []
    for chunk in chunks:
        label = f"[{chunk['source']}, chunk {chunk['chunk_index'] + 1}]"
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Build the context string injected into the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        citation = f"[Source {i+1}: {chunk['source']}, chunk {chunk['chunk_index']+1}]"
        parts.append(f"{citation}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)
