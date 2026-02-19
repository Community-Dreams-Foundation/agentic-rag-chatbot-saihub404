# 🏗️ SiteWatch AI — Construction Site Intelligence

> **The problem:** Construction site managers spend 30–45 minutes every morning manually cross-referencing weather apps, safety handbooks, and their own site plans to decide what their crew can actually do that day.
>
> **SiteWatch:** One question. Three sources looked up in parallel. One cited briefing.

---

## The Core Scenario

```
Site Manager: "Morning briefing for our Sydney site. Active works today:
               tower crane, concrete pour on level 4, glazing crew on facade."

SiteWatch:
  🧠 Profile loaded — Site Manager, active works: crane / concrete / glazing
  📖 Searching handbook... 3 relevant sections found
  🌤 Fetching weather for Sydney, Australia...
  ⚡ Cross-referencing evidence...

  ──────────────────────────────────────────────────────
  MORNING BRIEFING — Sydney Site

  Current conditions: 42 km/h wind, 21°C, 0 mm rain

  🛑 Tower Crane: SUSPENDED
     Wind is 42 km/h. Handbook limit is 38 km/h → exceeded [Source 1, chunk 2]
     Resume when sustained wind drops below 38 km/h.

  ✅ Concrete Pour (Level 4): GO
     Temperature 21°C — within the 10–32°C standard range [Source 1, chunk 4]
     Monitor curing temp every 2 hours.

  🛑 Glazing / Facade: SUSPENDED
     Wind is 42 km/h. Glazing limit is 30 km/h (sail effect) [Source 2, chunk 1]
     Redeploy crew to interior works.

  Overall Site Risk: HIGH — Site Manager + Safety Officer sign-off required
  ──────────────────────────────────────────────────────
```

This answer required three things simultaneously:
- **Memory** → who is the user, what are the active work packages
- **RAG** → exact thresholds from the handbook, with citations
- **Weather** → actual conditions for the specified location right now

None of the three is useful alone. The product value is the synthesis.

---

## How the Three Features Connect

The key architectural piece is the **evidence bag** pattern. Rather than chaining
features sequentially, all three sources are gathered into a `FusionResult` object
**before** any LLM synthesis happens. The synthesis prompt can only reference what's
in the bag — it cannot reach outside it. This is what makes citations honest.

```
query("Morning briefing for Sydney site")
         │
         ├── _extract_location()       →  "Sydney, Australia"
         │                                (fast LLM call, ~0.2s)
         │
         │   [ThreadPoolExecutor — run in parallel]
         ├── _gather_rag()             →  FusionResult.rag_context
         │   hybrid_search → RRF                            (run concurrently)
         │   build_context_block
         │
         └── _gather_weather()         →  FusionResult.weather_conditions
             geocode → archive API                         (run concurrently)
             sandbox analysis
             condense to bullet points

         ↓   [Memory — fast local file read, sequential]
         └── build_memory_context()    →  FusionResult.user_context

         ↓   [Single synthesis LLM call — reads evidence bag only]
         _build_messages(FusionResult)
         llm.stream()                  →  tokens streamed to UI

         ↓   [Post-processing]
         validate_citations()          →  strips hallucinated [Source N] refs
         _parse_risk_level()           →  LOW / MEDIUM / HIGH / CRITICAL
         maybe_write_memory()          →  writes if high-signal fact found
```

Total latency ≈ `max(rag_time, weather_time) + synthesis_time` — not their sum.

---

## Quickstart

```bash
# 1. Set up
cp .env.example .env # add your GROQ_API_KEY (free at console.groq.com)
make install

# 2. Run
make web                       # opens at localhost:8501

# 3. Demo flow (takes ~3 minutes)
#    a. Upload sample_docs/sitewatch_handbook.txt in the sidebar
#    b. Tell SiteWatch your role:
#       "I'm the site manager. Active works: crane, concrete pour on level 4,
#        glazing crew on the facade."
#    c. Ask: "Give me my morning briefing for the Sydney CBD site."
#    d. Watch: three evidence sources appear as they load, then a cited briefing
```

---

## Make Commands

| Command | What it does |
|---|---|
| `make web` | Start the SiteWatch web UI |
| `make cli` | Terminal interface |
| `make sanity` | End-to-end integration test → `artifacts/sanity_output.json` |
| `make eval` | 10-question evaluation harness → `artifacts/eval_report.json` |
| `make install` | Install all dependencies |
| `make clean` | Remove ChromaDB + artifacts |

---

## Safety Guarantees

Every answer from documents is grounded and verifiable:

| Property | How it works |
|---|---|
| **No hallucinated citations** | Post-generation validator strips `[Source N]` refs exceeding the retrieved chunk count |
| **Explicit threshold comparison** | System prompt requires format: *"Wind is X km/h. Limit is Y km/h → exceeded."* |
| **Retrieval failure discipline** | Score floor of 0.005 on RRF fusion; `NO_DOCS_PROMPT` template when no chunks pass |
| **Injection resistance** | Document context wrapped in boundary markers; 11-pattern regex scanner fires pre-LLM |
| **Memory discipline** | Two-stage pipeline: LLM evaluation (confidence ≥ 0.65) + semantic dedup before any write |
| **Sandbox isolation** | subprocess + TemporaryDirectory + stripped env + 15s timeout |

---

## Project Layout

```
app/
├── intelligence.py      ← SiteWatch fusion core (the product)
├── router.py            ← Intent classification
├── chatbot.py           ← General-purpose chatbot
├── config.py            ← All constants
├── llm/client.py        ← Groq wrapper + streaming
├── rag/
│   ├── ingestion.py     ← Parse → Chunk → Embed → Index
│   ├── retrieval.py     ← BM25 + Dense + RRF + threshold filter
│   ├── grounding.py     ← Citation hallucination validator
│   └── file_manager.py  ← List / delete / inspect / reindex
├── memory/
│   └── memory_manager.py ← Evaluate → Dedup → Write
└── sandbox/
    └── executor.py      ← Open-Meteo fetch + subprocess sandbox

web_app.py               ← SiteWatch UI (streaming, risk banners, evidence pills)
cli.py                   ← Terminal interface
scripts/
├── eval_harness.py      ← Automated evaluation (10 test cases, 4 categories)
├── run_sanity.py        ← End-to-end integration test
└── verify_output.py     ← Validates sanity_output.json

sample_docs/
├── sitewatch_handbook.txt  ← Construction safety thresholds (primary demo doc)
├── sample.txt              ← General document for RAG evaluation
└── injection_test.txt      ← Prompt injection resistance test

```
---
## Participant Info

- **Name**: Sai Ganesh Voodi
- **Email**: saiganeshvoodi@gmail.com
- **GitHub Username**: https://github.com/saihub404
- **Video Walkthrough**: https://youtu.be/p-xiS6NN4_c

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design overview including diagrams.

---

## Evaluation Questions

See [EVAL_QUESTIONS.md](EVAL_QUESTIONS.md) for suggested test questions covering all features.