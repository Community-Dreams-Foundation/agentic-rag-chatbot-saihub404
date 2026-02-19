"""
Citation Grounding Validator
=============================
Problem: LLMs sometimes generate citations like [Source 7: ...] that don't
correspond to any retrieved chunk. This module post-processes the LLM answer
and strips out any citation that cannot be traced back to an actual retrieved chunk.

Strategy:
  1. Parse all [Source N: ...] labels from the LLM's answer
  2. Compare against the set of valid labels from retrieved chunks
  3. Strike-through or remove citations with no evidence
  4. Return cleaned answer + list of hallucinated labels (for logging/UI)
"""
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any


# Matches: [Source 3: file.pdf, chunk 2] or [Source 1: anything]
_CITATION_RE = re.compile(r"\[Source\s+(\d+)[^\]]*\]", re.IGNORECASE)


def _valid_source_numbers(chunks: List[Dict[str, Any]]) -> set:
    """
    Return the set of valid source indices (1-based) from retrieved chunks.
    Source N corresponds to chunk at index N-1 in the list.
    """
    return set(range(1, len(chunks) + 1))


def validate_citations(
    answer: str,
    chunks: List[Dict[str, Any]],
) -> Tuple[str, List[str]]:
    """
    Scan the LLM answer for citation references.
    - Valid citations (matching a retrieved chunk) are kept as-is.
    - Hallucinated citations (no matching chunk) are flagged and removed.

    Returns:
        cleaned_answer  – answer with hallucinated citations removed
        hallucinated    – list of the removed citation strings
    """
    if not chunks:
        # No chunks were retrieved — any citation is hallucinated
        hallucinated = _CITATION_RE.findall(answer)
        if hallucinated:
            cleaned = _CITATION_RE.sub("", answer).strip()
            cleaned += "\n\n_(Note: Citations were removed as no documents are in the knowledge base.)_"
            return cleaned, [f"[Source {n}]" for n in hallucinated]
        return answer, []

    valid_nums  = _valid_source_numbers(chunks)
    hallucinated: List[str] = []
    seen_invalid: set = set()

    def _check_citation(match: re.Match) -> str:
        num = int(match.group(1))
        label = match.group(0)
        if num not in valid_nums:
            if label not in seen_invalid:
                seen_invalid.add(label)
                hallucinated.append(label)
            return ""   # remove from answer
        return label    # keep

    cleaned = _CITATION_RE.sub(_check_citation, answer)

    # Clean up double spaces left after removal
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    return cleaned, hallucinated


def check_answer_grounded(answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Full grounding report for a given answer + retrieved chunks.
    Returns a structured report useful for debugging and the sanity check.
    """
    cleaned, hallucinated = validate_citations(answer, chunks)

    # Extract citation numbers actually used in the cleaned answer
    used_nums = [int(m) for m in _CITATION_RE.findall(cleaned)]
    unique_sources_cited = sorted(set(used_nums))

    return {
        "grounded":             len(hallucinated) == 0,
        "hallucinated_citations": hallucinated,
        "sources_cited":        unique_sources_cited,
        "total_chunks_available": len(chunks),
        "cleaned_answer":       cleaned,
    }
