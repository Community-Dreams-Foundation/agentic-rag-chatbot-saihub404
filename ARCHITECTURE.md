# ARCHITECTURE.md — SiteWatch AI

## What It Is

SiteWatch AI is an agentic RAG chatbot built for construction site managers. It collapses a 30–45 minute manual morning briefing routine — cross-referencing weather, safety handbooks, and site-specific work packages — into a single cited query. The system fuses three evidence sources **in parallel** before any LLM synthesis occurs, ensuring every claim in the answer traces back to a real source.

---

## 1. Ingestion (Upload → Parse → Chunk → Index)

**Supported input formats:** `.txt`, `.pdf`, `.html`

**Parsing approach:**
- `.txt` files are read directly as UTF-8 text
- `.pdf` files are parsed with `pypdf` (page-level extraction)
- `.html` files are cleaned with `BeautifulSoup` (tags stripped, visible text kept)

**Chunking strategy:**
- Fixed-size character chunks: **500 chars** with **80-char overlap** (`CHUNK_SIZE`, `CHUNK_OVERLAP` in `config.py`)
- Overlap prevents threshold values from being split across chunk boundaries (critical for numeric safety limits like "38 km/h")

**Metadata captured per chunk:**
| Field | Description |
|---|---|
| `source` | Original filename |
| `chunk_id` | Sequential integer within the document |
| `total_chunks` | Total chunks in that document |
| `file_type` | `txt` / `pdf` / `html` |

**Implementation:** `app/rag/ingestion.py` → `ingest_file()`

---

## 2. Indexing / Storage

**Vector store:** [ChromaDB](https://docs.trychroma.com/) (persistent on disk at `chroma_db/`)

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, runs locally — no API call needed)

**Lexical index:** BM25 (`rank-bm25`) — maintained in-memory per session, rebuilt from the ChromaDB collection on startup

**Persistence:** ChromaDB writes to `chroma_db/` automatically; `make clean` removes it for a fresh start

**Implementation:** `app/rag/ingestion.py`, `app/rag/retrieval.py`

---

## 3. Retrieval + Grounded Answering

**Retrieval method:** Hybrid search with Reciprocal Rank Fusion (RRF)

```
Query
  ├── BM25 lexical search   → top-K ranked by keyword overlap
  └── Dense vector search   → top-K ranked by cosine similarity
       ↓
  RRF fusion: score = Σ 1/(rank_i + 60)  for each candidate
       ↓
  Score threshold filter (floor: 0.005) — low-confidence chunks dropped
       ↓
  Top-4 chunks passed to synthesis
```

**Citation construction:**
- Each chunk carries `source` (filename) and `chunk_id`
- `build_context_block()` wraps each chunk with a numbered label
- The synthesis system prompt enforces inline citation format: `[1]`, `[2]` in text, with a `Sources:` block at the end listing `[1] filename, chunk X`
- `validate_citations()` (post-generation) strips any `[Source N]` references exceeding the actual chunk count — preventing hallucinated citations entirely

**Failure behavior:**
- If RRF score falls below threshold → chunk excluded → `NO_DOCS_PROMPT` template used
- Bot responds: *"I cannot find this in the uploaded documents"* — never fabricates

**Implementation:** `app/rag/retrieval.py`, `app/rag/grounding.py`

---

## 4. Memory System (Selective Write)

SiteWatch uses a **two-stage pipeline** before writing any memory, run after every response.

```
Conversation turn ends
    ↓
Stage 1: LLM evaluator
    - Reads the exchange, returns JSON: {worth_storing, confidence, fact, target}
    - confidence must be ≥ 0.65 to proceed
    - target: "user" (role, name, schedule, active works) or "company" (site-wide thresholds, team structure, org rules)
    ↓
Stage 2: Semantic deduplication
    - Embeds the candidate fact
    - Compares cosine similarity against all existing entries in the target memory file
    - If similarity ≥ 0.85 with any existing entry → SKIP (no duplicate written)
    ↓
Write (only if both stages pass)
    - Appended as a timestamped bullet to the appropriate memory file
```

**What counts as high-signal (stored):**
- User's role / name (e.g. *"Site Safety Manager"*, *"John"*)
- Active work packages (e.g. *"crane ops on level 12, glazing facade"*)
- Recurring schedules (e.g. *"daily 8 AM briefing"*)
- Stated preferences (e.g. *"metric units"*)
- Site-specific thresholds / rules (e.g. *"council mandates 25 km/h scaffold limit"*)
- Team structure (e.g. *"3 glazing crews on site"*)

**What is explicitly NOT stored:**
- Raw conversation transcript
- PII (phone numbers, personal addresses)
- API keys or credentials
- Anything below the 0.65 confidence threshold
- Single weather readings (today's data, not a standing fact)
- One-off task completions

**Files written:**
- `users/<manager_id>/USER_MEMORY.md` — per-user; isolated by Manager ID set in the sidebar
- `users/<manager_id>/COMPANY_MEMORY.md` — org-wide for that manager's session
- Falls back to root `USER_MEMORY.md` / `COMPANY_MEMORY.md` when no Manager ID is set
- Directory and files are auto-created on first write (no manual setup needed)

**Implementation:** `app/memory/memory_manager.py`

---

## 5. Safe Tooling — Weather Sandbox (Open-Meteo)

SiteWatch fetches live (or recent archive) weather to cross-reference against handbook thresholds.

**Tool flow:**
```
location string (extracted by fast LLM call)
    ↓
Geocoding: Open-Meteo Geocoding API → (lat, lon)
    ↓
Weather fetch: Open-Meteo Archive/Forecast API → JSON hourly data
    ↓
Sandbox execution:
    - subprocess.run() in a TemporaryDirectory
    - Stripped environment (no secrets passed in)
    - 15-second hard timeout (SANDBOX_TIMEOUT)
    - No network access from within the subprocess
    ↓
Output condensed by LLM: bullet list of operational conditions
    (temperature, wind sustained/gusts, precipitation, anomalies)
    ↓
Passed into FusionResult.weather_conditions
```

**Safety boundaries:**
| Boundary | Implementation |
|---|---|
| Timeout | `subprocess.run(..., timeout=15)` |
| No imported secrets | `env={}` passed to subprocess |
| Temp working dir | `TemporaryDirectory()` — cleaned up automatically |
| Network from subprocess | Disabled (HTTP calls made in parent process only) |
| Injection in location | LLM extraction step sanitises; refuses multi-line or >80 char responses |

**Implementation:** `app/sandbox/executor.py`

---

## 6. The FusionResult Evidence Bag

The core architectural decision is the **evidence bag** (`FusionResult` dataclass in `app/intelligence.py`).

All three sources populate it **before** any LLM synthesis call:

```
RAG gather  ──┐
              ├──► FusionResult ──► Single synthesis LLM call ──► validated answer
Weather     ──┘
Memory (fast local read, sequential)
```

The synthesis prompt can **only reference what is in the bag** — it has no tool-calling capability during synthesis. This is what makes citations verifiable and prevents hallucination at the architecture level.

**Parallelism:** RAG and Weather run concurrently via `ThreadPoolExecutor(max_workers=2)`.  
**Latency:** `max(rag_time, weather_time) + synthesis_time` — not their sum.

---

## 7. Prompt Injection Defense

Every document ingested is treated as **data, not instructions**:

- Document context is wrapped in explicit boundary markers before insertion into the synthesis prompt
- An 11-pattern regex scanner (`app/rag/grounding.py`) fires pre-LLM on retrieved chunks, setting `injection_detected=True` if patterns match (e.g., "ignore prior instructions", "reveal", "pretend")
- The system prompt instructs the LLM to treat document content as evidence only
- Injection flag is surfaced in the API response and `sanity_output.json`

---

## 8. Tradeoffs & Next Steps

**Why this design?**
- The evidence bag eliminates the "answer first, cite later" anti-pattern that causes hallucinated references
- BM25 + dense hybrid search outperforms either alone on threshold-heavy technical docs (exact number matches favour BM25; semantic similarity catches paraphrased rules)
- Subprocess sandbox for weather analysis mirrors production security expectations — no arbitrary code execution in the main process

**What would improve with more time:**
- **Re-ranking:** A cross-encoder reranker (e.g. `ms-marco-MiniLM`) between retrieval and synthesis for higher precision
- **PDF page numbers:** Surface exact page in citations, not just chunk_id
- **Multi-user persistence:** ✅ Implemented — each Manager ID gets isolated memory under `users/<id>/`; production would add auth + per-user ChromaDB namespaces
- **Streaming weather:** Show weather data as it arrives rather than waiting for the full sandbox result