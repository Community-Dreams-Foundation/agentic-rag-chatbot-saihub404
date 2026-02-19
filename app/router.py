"""
Intent Router
=============
Classifies each user message into one of three pipelines:

  "rag"           → answer from indexed documents with citations
  "weather_tool"  → fetch + analyze weather via Open-Meteo sandbox
  "general"       → answer from general knowledge (no docs in index
                    or clearly off-topic question)

Two-stage routing:
  Stage 1: Fast regex heuristics (zero latency, covers 90% of cases)
  Stage 2: LLM classifier (fallback for ambiguous messages)

Also extracts structured weather parameters when the intent is weather_tool.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import date, timedelta


Intent = Literal["rag", "weather_tool", "general"]


@dataclass
class RouteResult:
    intent:        Intent
    confidence:    float                    # 0.0–1.0
    method:        str                      # "regex" | "llm"
    # Weather params (populated when intent == "weather_tool")
    location:      Optional[str] = None
    start_date:    Optional[str] = None
    end_date:      Optional[str] = None
    reasoning:     str = ""


# ── Stage 1: Regex heuristics ──────────────────────────────────────────────

_WEATHER_PATTERNS = [
    r"\bweather\b",
    r"\btemperature\b",
    r"\bforecast\b",
    r"\bprecipitation\b",
    r"\brainfall\b",
    r"\bwind\s*speed\b",
    r"\bclimate\s+(data|analysis|trend)",
    r"\bhumidity\b",
    r"\banalyze\b.{0,30}\b(weather|climate|temperature)\b",
    r"\b(weather|climate|temperature)\b.{0,30}\banalyze\b",
    r"\bhow\s+(hot|cold|warm|wet|rainy|windy)\b",
    r"\bsnowfall\b",
]
_WEATHER_RE = re.compile("|".join(_WEATHER_PATTERNS), re.IGNORECASE)

_GENERAL_PATTERNS = [
    r"^(hi|hello|hey|thanks|thank you|ok|okay|sure|bye|goodbye)[!?.,]*$",
    r"^what (is|are|does|do) you",
    r"^who (are|is) you\b",
    r"^can you (help|explain|tell me)",
    r"\b(your name|your model|your version)\b",
]
_GENERAL_RE = re.compile("|".join(_GENERAL_PATTERNS), re.IGNORECASE)


def _regex_route(message: str) -> Optional[RouteResult]:
    """Fast heuristic routing. Returns None if ambiguous."""
    msg = message.strip()

    if _WEATHER_RE.search(msg):
        params = _extract_weather_params(msg)
        return RouteResult(
            intent="weather_tool",
            confidence=0.85,
            method="regex",
            reasoning="Matched weather/climate keywords",
            **params,
        )

    if _GENERAL_RE.match(msg) and len(msg) < 80:
        return RouteResult(
            intent="general",
            confidence=0.90,
            method="regex",
            reasoning="Matched greeting/meta pattern",
        )

    return None   # ambiguous — escalate to LLM


# ── Stage 2: LLM classifier ────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are an intent classifier for an AI assistant.

Classify the user message into exactly one intent:

  "rag"           — The user is asking a question that should be answered using
                    uploaded documents (reports, papers, files).
                    Examples: "What does the report say about X?", "Summarize the document",
                              "What are the key findings?", "Cite the section on Y"

  "weather_tool"  — The user wants weather data, temperature analysis, climate trends,
                    or any time-series analysis of meteorological data for a location.
                    Examples: "Analyze weather in Berlin last month",
                              "What was the temperature in Tokyo in June?",
                              "Show me rainfall data for London Q1 2024"

  "general"       — The user is asking a general knowledge question not tied to
                    uploaded documents or weather data.
                    Examples: "What is RAG?", "Explain transformers", "Hello"

Also extract weather parameters if intent is weather_tool:
  - location (city, country)
  - start_date (YYYY-MM-DD, infer from context; default to 30 days ago if unspecified)
  - end_date   (YYYY-MM-DD, infer from context; default to 7 days ago if unspecified)

Respond with ONLY valid JSON — no markdown, no extra text:
{
  "intent": "rag" | "weather_tool" | "general",
  "confidence": 0.0–1.0,
  "location": "city, country" | null,
  "start_date": "YYYY-MM-DD" | null,
  "end_date": "YYYY-MM-DD" | null,
  "reasoning": "one sentence"
}"""


def _llm_route(message: str, llm_client, has_docs: bool) -> RouteResult:
    """LLM-based intent classification with parameter extraction."""
    import json, re as re_
    from datetime import date, timedelta

    today = date.today()
    default_start = str(today - timedelta(days=37))  # archive lag ~7 days
    default_end   = str(today - timedelta(days=7))

    hint = f"[Context: user {'has' if has_docs else 'has not'} uploaded any documents. Today is {today}.]"
    prompt = f"{hint}\n\nUser message: {message}"

    raw = llm_client.complete_with_system(
        system=_ROUTER_SYSTEM,
        user=prompt,
        temperature=0.0,
        max_tokens=200,
    )
    raw = re_.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(raw)
        intent = data.get("intent", "rag")
        if intent not in ("rag", "weather_tool", "general"):
            intent = "rag" if has_docs else "general"

        return RouteResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.7)),
            method="llm",
            location=data.get("location"),
            start_date=data.get("start_date") or default_start,
            end_date=data.get("end_date") or default_end,
            reasoning=data.get("reasoning", ""),
        )
    except Exception:
        # Safe fallback: use RAG if docs exist, else general
        return RouteResult(
            intent="rag" if has_docs else "general",
            confidence=0.5,
            method="llm_fallback",
            reasoning="JSON parse failed — safe default",
        )


# ── Weather parameter extractor ────────────────────────────────────────────

_LOCATION_HINTS = [
    r"(?:in|for|at|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:,\s*[A-Z][a-z]+)?)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+weather",
    r"weather\s+(?:in|for|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
]
_LOCATION_RE = re.compile("|".join(_LOCATION_HINTS))

_DATE_RANGES = {
    r"last\s+week":      (-14, -7),
    r"this\s+week":      (-7,  -1),
    r"last\s+month":     (-37, -7),
    r"last\s+30\s+days": (-37, -7),
    r"yesterday":        (-8,  -7),
    r"last\s+year":      (-372,-7),
    r"january\s+(\d{4})":    None,   # handled specially
    r"february\s+(\d{4})":   None,
    r"march\s+(\d{4})":      None,
    r"april\s+(\d{4})":      None,
    r"may\s+(\d{4})":        None,
    r"june\s+(\d{4})":       None,
    r"july\s+(\d{4})":       None,
    r"august\s+(\d{4})":     None,
    r"september\s+(\d{4})":  None,
    r"october\s+(\d{4})":    None,
    r"november\s+(\d{4})":   None,
    r"december\s+(\d{4})":   None,
}

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def _extract_weather_params(message: str) -> dict:
    """Best-effort extraction of location and date range from a message."""
    today = date.today()
    params = {
        "location":   None,
        "start_date": str(today - timedelta(days=37)),
        "end_date":   str(today - timedelta(days=7)),
    }

    # Location extraction
    m = _LOCATION_RE.search(message)
    if m:
        params["location"] = next((g for g in m.groups() if g), None)

    # Date range extraction
    lower = message.lower()
    for pattern, offsets in _DATE_RANGES.items():
        match = re.search(pattern, lower)
        if match:
            if offsets is not None:
                params["start_date"] = str(today + timedelta(days=offsets[0]))
                params["end_date"]   = str(today + timedelta(days=offsets[1]))
            else:
                # Named month + year
                month_name = pattern.split(r"\s+")[0]
                year = int(match.group(1))
                month = _MONTH_MAP.get(month_name, 1)
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                params["start_date"] = f"{year}-{month:02d}-01"
                params["end_date"]   = f"{year}-{month:02d}-{last_day:02d}"
            break

    return params


# ── Public API ────────────────────────────────────────────────────────────

def route(
    message: str,
    llm_client=None,
    has_docs: bool = True,
) -> RouteResult:
    """
    Route a user message to the appropriate pipeline.

    Args:
        message:    The user's message
        llm_client: LLM client instance (used for Stage 2 if regex is ambiguous)
        has_docs:   Whether any documents are currently indexed

    Returns:
        RouteResult with intent + optional weather params
    """
    # Stage 1: fast regex
    result = _regex_route(message)
    if result is not None:
        # Fill in weather params if not already extracted
        if result.intent == "weather_tool" and not result.location:
            params = _extract_weather_params(message)
            result.location   = params["location"]
            result.start_date = params["start_date"]
            result.end_date   = params["end_date"]
        # If no docs, downgrade rag → general
        if result.intent == "rag" and not has_docs:
            result.intent = "general"
        return result

    # Stage 2: LLM classifier
    if llm_client is not None:
        return _llm_route(message, llm_client, has_docs)

    # Final fallback
    return RouteResult(
        intent="rag" if has_docs else "general",
        confidence=0.5,
        method="fallback",
        reasoning="No LLM client provided",
    )
