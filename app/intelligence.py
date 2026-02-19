"""
SiteWatch — Fusion Intelligence Core
======================================

The product solves one specific problem:
  Construction site managers start each day needing to know what their crew
  can actually do — given current conditions, their specific handbook rules,
  and their site's active work packages. This decision currently takes
  30–45 minutes of manual cross-referencing every morning.

SiteWatch collapses that into one query.

How the three features connect
────────────────────────────────
  Query: "Can we proceed with crane operations and the concrete pour today?"

  Without integration:
    Manager checks weather app (wind: 42 km/h) → opens handbook (limit: 38 km/h) →
    remembers they have a glazing crew on the facade too (limit: 30 km/h) →
    cross-references all three manually → writes up briefing notes

  With SiteWatch:
    Memory   →  "Site Manager. Active works: crane, concrete pour, glazing crew on Lvl 8."
    RAG      →  Retrieves crane limit (38 km/h), concrete temp range, glazing limit (30 km/h)
    Weather  →  42 km/h sustained, 51 km/h gusts, 18°C, 0 mm rain
    Synthesis → Cross-references every active work package against actual conditions.
                Produces a structured morning briefing with per-activity status.

Architecture
─────────────
  FusionResult is an evidence bag. All three sources populate it independently.
  The synthesis step reads ONLY from this bag — it cannot invent evidence.

  RAG + Weather run concurrently (ThreadPoolExecutor) to minimise latency.
  Total latency ≈ max(rag_time, weather_time) + synthesis_time.

  Two modes:
    stream_query()  →  yields status/token/done events (used by web UI)
    query()         →  blocking, returns FusionResult (used by CLI/tests)
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Iterator

from app.llm.client import LLMClient
from app.rag.retrieval import hybrid_search, build_context_block, format_citations
from app.rag.grounding import validate_citations
from app.memory.memory_manager import build_memory_context, maybe_write_memory
from app.weather import get_current_conditions
from app.sandbox.executor import analyze_weather  # kept for extended analysis expander


# ─────────────────────────────────────────────────────────────────────────────
# Evidence bag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    """
    All gathered evidence, plus the final synthesized output.

    Design intent: populate all evidence fields BEFORE any LLM synthesis call.
    The synthesis prompt only reads from this object — it cannot reach outside it.
    This is what makes citations honest.
    """
    question:           str

    # Source 1 — RAG
    rag_chunks:         List[Dict[str, Any]] = field(default_factory=list)
    rag_context:        str = ""
    rag_citations:      List[str] = field(default_factory=list)
    rag_available:      bool = False

    # Source 2 — Weather
    weather_location:   Optional[str] = None
    weather_conditions: str = ""          # condensed bullet list for synthesis
    weather_raw:        str = ""          # full sandbox output for UI display
    weather_available:  bool = False

    # Source 3 — Memory
    user_context:       str = ""
    memory_available:   bool = False

    # Synthesized output
    answer:             str = ""
    recommendation:     str = "INSUFFICIENT DATA"
    risk_level:         str = "UNKNOWN"       # LOW / MEDIUM / HIGH / CRITICAL
    work_items:         List[Dict] = field(default_factory=list)   # per-activity breakdown
    citations:          List[str] = field(default_factory=list)
    hallucinated:       List[str] = field(default_factory=list)
    memory_written:     bool = False
    memory_summary:     str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are SiteWatch, an operations intelligence assistant for construction site managers.

Your job is to cross-reference evidence and produce a clear, actionable briefing or decision answer.

THRESHOLD DECISION RULE (read carefully):
  Each activity has a MAXIMUM SAFE LIMIT — if the measured value EXCEEDS that limit, the activity
  is SUSPENDED. If the measured value is BELOW the limit, the activity is GO.

  Examples:
    Wind is 17 km/h, crane limit is 38 km/h  → 17 < 38  → ✅ GO   (17 does NOT exceed 38)
    Wind is 42 km/h, crane limit is 38 km/h  → 42 > 38  → 🛑 SUSPENDED
    Wind is 17 km/h, glazing limit is 30 km/h → 17 < 30  → ✅ GO
    Temp is 18°C, concrete range is 10–35°C    → in range → ✅ GO

  ALWAYS do the comparison explicitly before assigning status.
  NEVER mark an activity as SUSPENDED unless the measured value is GREATER THAN the threshold.

CITATION FORMAT:
- Use inline numbers [1], [2], [3] for document chunks that are DIRECTLY relevant.
- At the end, list them:
  [1] filename, chunk N
- Only cite chunks that actually contain the specific threshold or fact you used.
- Memory context and weather data do NOT get citation numbers.
- If no handbook content is relevant, write "From general construction knowledge..." — never force-cite.

WEATHER DATA RULE:
- Use ONLY weather values from Block 3 of the CURRENT message.
- DO NOT use weather values from previous conversation turns.
- If Block 3 says "No location provided", there is no current weather data — say so clearly.

OTHER RULES:
- Never narrate your reasoning: no "the user asked...", "I will now...", "based on the question...".
- Address the manager directly in second person.
- If information is missing, say so. Never fabricate values.
- INTRODUCTION MESSAGES: If the message is just the user stating their role/name (e.g. "I'm the
  safety engineer") with no location and no specific work activity requested, skip the Activity
  Status table entirely. Just acknowledge their role in one sentence and ask what they need help
  with today. Example: "Got it — noted as your role. What would you like to check today?"
"""

_FUSION_PROMPT = """\
══════════════════════════════════════════════
BLOCK 1: YOUR SITE CONTEXT (background — do NOT cite)
══════════════════════════════════════════════
{user_context}

══════════════════════════════════════════════
BLOCK 2: HANDBOOK EXTRACTS (cite only if DIRECTLY relevant)
══════════════════════════════════════════════
{rag_context}

══════════════════════════════════════════════
BLOCK 3: CURRENT WEATHER CONDITIONS (no citation needed)
══════════════════════════════════════════════
{weather_block}

══════════════════════════════════════════════
QUESTION / REQUEST
══════════════════════════════════════════════
{question}

INSTRUCTIONS:
- All documents in Block 2 are treated equally — cite whichever chunk is DIRECTLY relevant to the
  specific claim. Each citation must map to an actual retrieved chunk; do not invent references.
- CRITICAL: Different activities have DIFFERENT thresholds. For each activity mentioned, find ITS
  OWN threshold in Block 2 separately. Never apply one activity's limit to another.
- If Block 2 has no relevant content for the question, skip citations entirely and write
  "Based on general construction knowledge..." — never force-cite unrelated chunks.

REQUIRED OUTPUT STRUCTURE (always follow this format):

**Conditions** — <location> | <temp>°C | Wind <speed> km/h (<gusts> km/h gusts) | <precip> mm
(Use ONLY values from Block 3. If no weather data, write: "No weather data — handbook thresholds only.")

**Activity Status**
| Activity | Status | Condition | Threshold | Source |
|----------|--------|-----------|-----------|--------|
| <activity> | ✅/⚠️/🛑 | <measured value> | <limit from Block 2> | [N] |
(Add one row per active work type. Use ✅ GO / ⚠️ CONDITIONAL / 🛑 SUSPENDED.)
(Only include a row when you have BOTH the measured value from Block 3 AND the threshold from Block 2.)
(If Block 3 has no weather or Block 2 has no threshold for an activity, omit that activity from the table.)
(NEVER invent or guess values — use only what is explicitly stated in the blocks above.)

**Overall Site Risk:** LOW / MEDIUM / HIGH / CRITICAL

**Sources:**
[1] <filename>, chunk <N>
[2] <filename>, chunk <N>
(Only list sources actually cited above. Omit this section if no citations were used.)
"""

_CONDITIONS_EXTRACT = """\
From this weather analysis, extract a concise bullet list of KEY OPERATIONAL CONDITIONS.
Include only metrics relevant to construction site decisions:
- Temperature (°C): current average, min, max
- Wind: sustained (km/h), peak gust (km/h)
- Precipitation: intensity (mm/hr), total (mm)
- Any anomalies (flag with ⚠️)
- Apparent temperature if extreme heat/cold detected

Raw data:
{raw}

Reply with ONLY the bullet list. No preamble. Start each line with •"""

_LOCATION_PROMPT = """\
Extract the location (city, or city + state/country) from the CURRENT MESSAGE.
Use the CONVERSATION CONTEXT below to disambiguate ambiguous city names.

Conversation context (recent turns):
{context}

Examples of correct outputs: "Austin, TX", "Sydney, Australia", "London, UK", "Denton, Texas"
Rules:
- If the user mentions just a city name (e.g. "denton"), use context to infer the state/country
  (e.g. prior messages mention Texas cities → reply "Denton, Texas")
- If multiple cities, pick the one the CURRENT message is asking about.
- If no location at all in the current message: reply NONE.
- Reply with ONLY the location string. No explanation.

Current message: {msg}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class SiteWatch:
    """
    One query. Three evidence sources gathered in parallel. One cited answer.

    Two public entry points:
        result = sw.query("Can we start the crane today in Melbourne?")
        for event in sw.stream_query("Morning briefing for Sydney site"):
            ...
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.llm = LLMClient()
        self.history: List[Dict[str, str]] = []
        self._init_namespace()

    def _init_namespace(self):
        """Resolve per-user memory file paths onto self (never mutate global config)."""
        from app.config import BASE_DIR, USER_MEMORY_FILE, COMPANY_MEMORY_FILE
        if self.user_id and self.user_id != "default":
            d = BASE_DIR / "users" / self.user_id
            d.mkdir(parents=True, exist_ok=True)
            self._user_mem_file    = d / "USER_MEMORY.md"
            self._company_mem_file = d / "COMPANY_MEMORY.md"
        else:
            self._user_mem_file    = USER_MEMORY_FILE
            self._company_mem_file = COMPANY_MEMORY_FILE

    # ─── Blocking query ───────────────────────────────────────────────────

    def query(self, question: str) -> FusionResult:
        """Gather all evidence in parallel, synthesize, return result."""
        ev = FusionResult(question=question)
        location = self._extract_location(question)
        self._gather_all(ev, question, location)
        self._synthesize(ev)
        self._post_process(ev, question)
        return ev

    # ─── Streaming query ──────────────────────────────────────────────────

    def stream_query(self, question: str) -> Iterator[Dict[str, Any]]:
        """
        Yields events as evidence arrives, then streams synthesis tokens.

        Event shapes:
          {"type": "status",  "stage": "memory|rag|weather|synthesis", "message": str}
          {"type": "token",   "content": str}
          {"type": "done",    "result": FusionResult}
        """
        ev = FusionResult(question=question)
        location = self._extract_location(question)

        # Memory — fast local read, no need to thread
        yield {"type": "status", "stage": "memory", "message": "🧠 Loading your site context…"}
        ev.user_context    = build_memory_context(
            user_mem_file=self._user_mem_file,
            company_mem_file=self._company_mem_file,
        )
        ev.memory_available = bool(ev.user_context.strip())

        # RAG + Weather in parallel — report each as it completes
        rag_done = weather_done = False
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rag_fut = pool.submit(self._gather_rag, ev, question)
            wx_fut  = pool.submit(self._gather_weather, ev, location) if location else None

            # Emit status events as futures complete (don't block on one before the other)
            pending = {rag_fut: "rag"}
            if wx_fut:
                pending[wx_fut] = "weather"

            for fut in concurrent.futures.as_completed(pending):
                stage = pending[fut]
                if stage == "rag":
                    n = len(ev.rag_chunks)
                    msg = f"📄 Found {n} relevant {'chunk' if n==1 else 'chunks'} in your handbook"
                    if not ev.rag_available:
                        msg = "📄 No matching handbook sections found"
                else:
                    loc = ev.weather_location or location
                    if ev.weather_available:
                        msg = f"🌤 Weather data loaded for {loc}"
                    else:
                        msg = f"🌤 Weather unavailable for {loc}"
                yield {"type": "status", "stage": stage, "message": msg}

            if not wx_fut:
                yield {"type": "status", "stage": "weather",
                       "message": "🌤 No location detected — answering from handbook only"}

        # Synthesis — stream tokens
        yield {"type": "status", "stage": "synthesis", "message": "⚡ Cross-referencing evidence…"}
        messages = self._build_messages(ev)
        full = ""
        for tok in self.llm.stream(messages, temperature=0.1, max_tokens=1400):
            full += tok
            yield {"type": "token", "content": tok}

        ev.answer = full
        self._post_process(ev, question)
        yield {"type": "done", "result": ev}

    # ─── Evidence gathering ───────────────────────────────────────────────

    def _gather_all(self, ev: FusionResult, question: str, location: Optional[str]):
        """Parallel gather for blocking path."""
        ev.user_context    = build_memory_context(
            user_mem_file=self._user_mem_file,
            company_mem_file=self._company_mem_file,
        )
        ev.memory_available = bool(ev.user_context.strip())
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(self._gather_rag, ev, question)]
            if location:
                futs.append(pool.submit(self._gather_weather, ev, location))
            for f in concurrent.futures.as_completed(futs):
                f.result()   # re-raise exceptions

    def _gather_rag(self, ev: FusionResult, question: str):
        # Skip RAG for pure introductions — the handbook doesn't help with
        # "I am the site manager" or "I'm John" type messages.
        q_lower = question.lower().strip()
        _INTRO_PATTERNS = (
            "i am ", "i'm ", "my name", "my role", "hello", "hi,", "hi ",
            "good morning", "just wanted to say", "just to confirm",
            "noted,", "please store", "please remember",
        )
        # Only skip if the ENTIRE message is a short intro (< 12 words, no location/work keywords)
        _WORK_KEYWORDS = (
            "weather", "analysis", "crane", "concrete", "scaffold",
            "glaz", "wind", "rain", "brief", "today", "report", "check",
            "dallas", "sydney", "denton", "houston", "austin",  # city names trigger work analysis
        )
        word_count = len(q_lower.split())
        has_work = any(w in q_lower for w in _WORK_KEYWORDS)
        is_intro = (
            any(q_lower.startswith(p) for p in _INTRO_PATTERNS)
            and word_count <= 12
            and not has_work
        )
        if is_intro:
            ev.rag_context   = "No matching handbook sections (introduction/greeting query)."
            ev.rag_available = False
            return

        try:
            chunks = hybrid_search(question, top_k=4)
            # Secondary quality gate: even after RRF, discard if the best chunk
            # doesn't clear the cosine similarity bar (off-topic collection hit).
            if chunks:
                best_score = max(c.get("score", 0) for c in chunks)
                if best_score < 0.35:
                    chunks = []  # nothing relevant enough
            if chunks:
                ev.rag_chunks    = chunks
                ev.rag_context   = build_context_block(chunks)
                ev.rag_citations = format_citations(chunks)
                ev.rag_available = True
            else:
                ev.rag_context   = "No matching handbook sections found."
                ev.rag_available = False
        except Exception as e:
            ev.rag_context = f"[Search error: {e}]"
            ev.rag_available = False

    def _gather_weather(self, ev: FusionResult, location: str):
        """Fetch current weather conditions using wttr.in (primary) with Open-Meteo fallback."""
        if not location:
            return
        try:
            data = get_current_conditions(location)
            if data["success"]:
                ev.weather_location  = data["location"]
                ev.weather_raw       = data.get("summary_text", "")
                ev.weather_available = True
                ev.weather_conditions = data["summary_text"]
            else:
                ev.weather_conditions = (
                    f"⚠️ Could not retrieve weather for '{location}': "
                    f"{data.get('error', 'Unknown error')}. "
                    "Use handbook thresholds only."
                )
        except Exception as e:
            ev.weather_conditions = f"⚠️ Weather error: {e}"

    # ─── Synthesis ────────────────────────────────────────────────────────

    def _synthesize(self, ev: FusionResult):
        ev.answer = self.llm.chat(self._build_messages(ev),
                                  temperature=0.1, max_tokens=1400)

    def _build_messages(self, ev: FusionResult) -> List[Dict]:
        user_ctx = ev.user_context if ev.memory_available else (
            "No user context saved yet. Ask the user for their role and active works.")

        # ── Lightweight path: pure intro / greeting / no evidence ────────────
        # When there's no weather data AND no relevant handbook content,
        # skip the full structured FUSION_PROMPT (it would produce an empty table).
        # Just respond conversationally.
        if not ev.weather_available and not ev.rag_available:
            simple_prompt = (
                f"Site context (background):\n{user_ctx}\n\n"
                f"User message: {ev.question}\n\n"
                "Respond briefly and directly. If this is a role introduction, "
                "acknowledge it in one sentence and ask what they need help with. "
                "Do NOT produce a table, conditions block, or risk rating."
            )
            return [
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": simple_prompt},
            ]

        # ── Full synthesis path: evidence available ───────────────────────────
        if ev.weather_available:
            weather_block = (f"Location: {ev.weather_location}\n\n"
                             f"Conditions:\n{ev.weather_conditions}")
        elif ev.weather_conditions:
            weather_block = ev.weather_conditions
        else:
            weather_block = "No location provided. Use handbook thresholds only."

        rag_block = ev.rag_context if ev.rag_available else (
            "No matching handbook sections in knowledge base. "
            "Answer from general construction knowledge and note this.")

        user_prompt = _FUSION_PROMPT.format(
            user_context=user_ctx,
            rag_context=rag_block,
            weather_block=weather_block,
            question=ev.question,
        )
        msgs = [{"role": "system", "content": _SYSTEM}]
        # No history: each turn is self-contained.
        # Persistent context comes from Block 1 (memory), not conversation history.
        # This prevents weather values from previous turns leaking into new queries.
        msgs.append({"role": "user", "content": user_prompt})
        return msgs

    # ─── Post-processing ──────────────────────────────────────────────────

    def _post_process(self, ev: FusionResult, question: str):
        ev.answer, ev.hallucinated = validate_citations(ev.answer, ev.rag_chunks)
        ev.citations    = format_citations(ev.rag_chunks)
        ev.recommendation = _parse_recommendation(ev.answer)
        ev.risk_level     = _parse_risk_level(ev.answer)

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant", "content": ev.answer})

        mem = maybe_write_memory(
            question, ev.answer, self.llm,
            user_mem_file=self._user_mem_file,
            company_mem_file=self._company_mem_file,
        )
        ev.memory_written = mem["wrote"]
        ev.memory_summary = mem.get("summary", "")

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _extract_location(self, msg: str) -> Optional[str]:
        # Pass recent history as context for city disambiguation
        ctx_turns = self.history[-4:] if self.history else []
        ctx = " | ".join(
            f"{t['role']}: {t['content'][:120]}" for t in ctx_turns
        ) or "none"
        resp = self.llm.complete(
            _LOCATION_PROMPT.format(msg=msg, context=ctx),
            temperature=0.0, max_tokens=30,
        ).strip()
        if resp.upper() == "NONE" or len(resp) > 80 or "\n" in resp:
            return None
        return resp

    def clear_history(self):
        self.history = []

    def export_history(self) -> str:
        if not self.history:
            return "_No history yet._"
        lines = ["# SiteWatch — Session Log\n"]
        for t in self.history:
            label = "**Site Manager**" if t["role"] == "user" else "**SiteWatch**"
            lines.append(f"{label}:\n{t['content']}\n\n---\n")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_recommendation(answer: str) -> str:
    """Extract the dominant recommendation from the answer."""
    if "🛑" in answer or "SUSPENDED" in answer:
        return "PARTIAL-STOP"
    if "⚠️ CONDITIONAL" in answer or "CONDITIONAL" in answer:
        return "CONDITIONAL"
    if "✅ GO" in answer:
        return "GO"
    return "REVIEW"

def _parse_risk_level(answer: str) -> str:
    """Extract risk rating from the answer."""
    ans = answer.upper()
    if "OVERALL SITE RISK: CRITICAL" in ans or "RISK: CRITICAL" in ans:
        return "CRITICAL"
    if "OVERALL SITE RISK: HIGH" in ans or "RISK: HIGH" in ans:
        return "HIGH"
    if "OVERALL SITE RISK: MEDIUM" in ans or "RISK: MEDIUM" in ans:
        return "MEDIUM"
    if "OVERALL SITE RISK: LOW" in ans or "RISK: LOW" in ans:
        return "LOW"
    return "UNKNOWN"
