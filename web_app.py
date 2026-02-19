"""
SiteWatch — Construction Site Intelligence
==========================================
One question. Three sources. One briefing.

Run: streamlit run web_app.py
"""
import streamlit as st
from pathlib import Path
from typing import List
import tempfile, os

st.set_page_config(
    page_title="SiteWatch — Site Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── App-level background ── */
section.main > div { background: #0f1117; }

/* ── Risk banners ── */
.risk-low      { padding:12px 20px; border-radius:8px; font-weight:700; font-size:15px;
                 background:#052e16; color:#86efac; border-left:5px solid #22c55e; margin:10px 0; }
.risk-medium   { padding:12px 20px; border-radius:8px; font-weight:700; font-size:15px;
                 background:#451a03; color:#fcd34d; border-left:5px solid #f59e0b; margin:10px 0; }
.risk-high     { padding:12px 20px; border-radius:8px; font-weight:700; font-size:15px;
                 background:#450a0a; color:#fca5a5; border-left:5px solid #ef4444; margin:10px 0; }
.risk-critical { padding:12px 20px; border-radius:8px; font-weight:700; font-size:15px;
                 background:#4c0519; color:#fecdd3; border-left:5px solid #e11d48; margin:10px 0;
                 animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.75; } }
.risk-unknown  { padding:12px 20px; border-radius:8px; font-weight:700; font-size:15px;
                 background:#1e293b; color:#94a3b8; border-left:5px solid #475569; margin:10px 0; }

/* ── Evidence source pills ── */
.src-pill { display:inline-block; padding:4px 12px; border-radius:99px;
            font-size:11px; font-weight:700; margin:2px 3px 2px 0; letter-spacing:0.3px; }
.src-rag  { background:#1e3a5f; color:#93c5fd; border:1px solid #2563eb; }
.src-wx   { background:#422006; color:#fcd34d; border:1px solid #d97706; }
.src-mem  { background:#14532d; color:#86efac; border:1px solid #16a34a; }

/* ── Loading stage indicators ── */
.stage-done   { color:#22c55e; font-weight:600; font-size:13px; }
.stage-active { color:#f59e0b; font-weight:600; font-size:13px; }

/* ── Memory card ── */
.mem-card { background:#1e293b; border:1px solid #334155; border-radius:10px;
            padding:12px 16px; margin-top:4px; }
.mem-item { font-size:12px; color:#cbd5e1; margin:4px 0; padding-left:8px;
            border-left:2px solid #3b82f6; }
.mem-role { font-size:13px; font-weight:700; color:#e2e8f0; margin-bottom:6px; }

/* ── Header strip ── */
.site-header { background: linear-gradient(135deg, #1e3a5f 0%, #0f2027 100%);
               border-radius:12px; padding:18px 24px; margin-bottom:16px;
               border:1px solid #1d4ed8; }
.site-header h2 { margin:0; color:#e2e8f0; font-size:22px; }
.site-header p  { margin:4px 0 0; color:#94a3b8; font-size:13px; }

/* ── Index status badge ── */
.indexed-badge { display:inline-block; background:#14532d; color:#86efac;
                 border:1px solid #16a34a; border-radius:99px;
                 padding:2px 10px; font-size:11px; font-weight:700; margin-left:6px; }
.not-indexed-badge { display:inline-block; background:#450a0a; color:#fca5a5;
                     border:1px solid #dc2626; border-radius:99px;
                     padding:2px 10px; font-size:11px; font-weight:700; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ══ Session init ══════════════════════════════════════════════════════════════
for k, v in {"messages": [], "user_id": "default", "sw": None,
              "indexed_files": set(), "startup_done": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_sw():
    from app.intelligence import SiteWatch
    uid = st.session_state.user_id
    if st.session_state.sw is None or getattr(st.session_state.sw, "user_id", None) != uid:
        st.session_state.sw = SiteWatch(user_id=uid)
    return st.session_state.sw

# ══ Startup: auto-ingest sample handbook if nothing is indexed yet ══════════
if not st.session_state.startup_done:
    st.session_state.startup_done = True
    from app.rag.ingestion import ingest_file, get_chunk_count
    if get_chunk_count() == 0:
        sample_dir = Path(__file__).parent / "sample_docs"
        auto_ingest = ["sitewatch_handbook.txt"]
        for fname in auto_ingest:
            p = sample_dir / fname
            if p.exists():
                try:
                    ingest_file(p)
                except Exception:
                    pass

RISK_STYLE = {
    "LOW":      ("risk-low",      "🟢 Site Risk: LOW — Full operations"),
    "MEDIUM":   ("risk-medium",   "🟡 Site Risk: MEDIUM — Heightened monitoring"),
    "HIGH":     ("risk-high",     "🔴 Site Risk: HIGH — Sign-off required"),
    "CRITICAL": ("risk-critical", "⛔ Site Risk: CRITICAL — Site stand-down"),
    "UNKNOWN":  ("risk-unknown",  "⬜ Risk level not determined"),
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 8px;">
      <span style="font-size:32px;">🏗️</span><br>
      <span style="font-size:18px; font-weight:800; color:#e2e8f0; letter-spacing:1px;">SITEWATCH</span><br>
      <span style="font-size:11px; color:#64748b; letter-spacing:2px;">CONSTRUCTION INTELLIGENCE</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Site Profile ──────────────────────────────────────────────────────
    with st.expander("👷 Site Profile", expanded=True):
        uid = st.text_input("Manager ID", value=st.session_state.user_id,
                            placeholder="e.g. john_smith, site_a",
                            help="Memories are stored per manager ID across sessions.")
        if uid != st.session_state.user_id:
            st.session_state.user_id  = uid
            st.session_state.sw       = None
            st.session_state.messages = []
            st.rerun()

        # Read memory from the ACTIVE user's per-user file paths (via SiteWatch instance)
        sw_instance = get_sw()
        mem_path = sw_instance._user_mem_file
        co_path  = sw_instance._company_mem_file
        mem_text = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
        co_text  = co_path.read_text(encoding="utf-8")  if co_path.exists()  else ""
        has_mem  = bool(mem_text) and "_No memories" not in mem_text
        has_co   = bool(co_text)  and "_No memories" not in co_text

        if has_mem or has_co:
            st.markdown("""
            <div style="font-size:12px; color:#64748b; margin:6px 0 4px;">
            🧠 <b>Persistent memory active</b> — SiteWatch remembers context across sessions.
            </div>""", unsafe_allow_html=True)

            # User memory card — show concise bullet points only
            if has_mem:
                items = [l.strip() for l in mem_text.splitlines()
                         if l.strip().startswith("-")][-5:]  # last 5 facts
                if items:
                    html_items = "".join(
                        f'<div class="mem-item">{"  ".join(i.split("]")[-1].strip().split("(confidence")[0].strip() for _ in [1])}</div>'
                        for i in items
                    )
                    st.markdown(
                        f'<div class="mem-card">'
                        f'<div class="mem-role">👤 Your Profile</div>'
                        f'{html_items}</div>',
                        unsafe_allow_html=True,
                    )
            if has_co:
                co_items = [l.strip() for l in co_text.splitlines()
                            if l.strip().startswith("-")][-3:]
                if co_items:
                    html_items = "".join(
                        f'<div class="mem-item">{"  ".join(i.split("]")[-1].strip().split("(confidence")[0].strip() for _ in [1])}</div>'
                        for i in co_items
                    )
                    st.markdown(
                        f'<div class="mem-card" style="margin-top:8px;">'
                        f'<div class="mem-role">🏢 Site Context</div>'
                        f'{html_items}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("_Tell SiteWatch your role and site details — it remembers across sessions._")
            st.caption("Try: *\"I'm the site manager. Active works: crane level 12, concrete pour level 6, glazing crew on south facade.\"*")

    # ── Operations Handbook ───────────────────────────────────────────────
    with st.expander("📖 Operations Handbook", expanded=True):
        uploaded = st.file_uploader(
            "Drop PDF/TXT/HTML — auto-indexed instantly",
            type=["pdf", "txt", "html", "md"],
            label_visibility="collapsed",
        )
        # Auto-index on drop (no button needed)
        if uploaded and uploaded.name not in st.session_state.indexed_files:
            from app.rag.ingestion import ingest_file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                with st.spinner(f"📥 Indexing {uploaded.name}…"):
                    r = ingest_file(tmp_path, original_name=uploaded.name)
                os.unlink(tmp_path)
                st.session_state.indexed_files.add(uploaded.name)
                st.success(f"✅ **{r['file']}** — {r['total_chunks']} chunks indexed ({r['new_chunks']} new)")
                st.rerun()
            except Exception as e:
                st.error(str(e))
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # Show indexed sources
        from app.rag.file_manager import list_sources, delete_source
        srcs = list_sources()
        if srcs:
            for s in srcs:
                c1, c2 = st.columns([5, 1])
                c1.markdown(
                    f'<span class="indexed-badge">✓ indexed</span> '
                    f'<span style="font-size:12px; color:#cbd5e1;">{s["source"]}</span> '
                    f'<span style="font-size:11px; color:#475569;">({s["chunks"]} chunks)</span>',
                    unsafe_allow_html=True,
                )
                if c2.button("✕", key=f"rm_{s['source']}", help="Remove from index"):
                    delete_source(s["source"])
                    st.rerun()
        else:
            st.markdown(
                '<span class="not-indexed-badge">no handbook indexed</span><br>'
                '<span style="font-size:11px; color:#64748b;">Upload a handbook above or the sample '
                '<code>sitewatch_handbook.txt</code> to enable citation-grounded answers.</span>',
                unsafe_allow_html=True,
            )

    # ── How it works ─────────────────────────────────────────────────────
    with st.expander("ℹ️ How it works"):
        st.markdown("""
| Source | Role |
|--------|------|
| 🧠 **Memory** | Your role, site, active works — remembered across sessions |
| 📖 **Handbook** | Exact thresholds, cited with `[1]`, `[2]` inline + Sources block |
| 🌤 **Weather** | Live conditions fetched for your location |

The engine compares **actual conditions vs handbook thresholds** per activity:
`✅ GO` / `⚠️ CONDITIONAL` / `🛑 SUSPENDED`
        """)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.sw:
                st.session_state.sw.clear_history()
            st.rerun()
    with c2:
        if st.session_state.messages:
            md = get_sw().export_history()
            st.download_button("📥 Export log", md,
                               file_name="sitewatch_log.md",
                               mime="text/markdown",
                               use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="site-header">
  <h2>🏗️ SiteWatch</h2>
  <p>Construction site intelligence — weather · handbook · memory · one cited answer.</p>
</div>
""", unsafe_allow_html=True)

# ── Onboarding (shown only when chat is empty) ────────────────────────────
if not st.session_state.messages:
    col_examples, col_explain = st.columns([5, 4])

    with col_examples:
        st.markdown("**Quick start — try a query:**")
        EXAMPLES = [
            ("☀️ Morning briefing",
             "Give me my morning briefing for the Sydney CBD site. We have crane ops, a concrete pour on level 4, and a glazing crew on the facade."),
            ("🌬️ Wind go/no-go",
             "Wind is 42 km/h in Austin, Texas. I have crane operations and a glazing crew active. Can we proceed?"),
            ("🌡️ Concrete pour check",
             "Is the temperature suitable for a concrete pour in Brisbane this afternoon?"),
            ("👷 Heat stress check",
             "What heat stress protocols apply for our outdoor crew in Darwin today? It's 40°C apparent."),
            ("🌧️ Rain impact assessment",
             "It's been raining all morning in Auckland. What works are affected?"),
            ("💨 Austin weather for site ops",
             "Austin weather for construction — what are current conditions?"),
        ]
        for label, query in EXAMPLES:
            if st.button(label, key=f"ex_{label}", use_container_width=True):
                st.session_state["prefill"] = query
                st.rerun()

    with col_explain:
        st.markdown("**What happens when you ask:**")
        st.code("""
Your question
     │
     ├─ 🧠 Memory   Who are you? Your role, site, works
     │
     ├─ 📖 Handbook  What do the rules say? (cited)
     │   (parallel)  Thresholds, procedures, limits
     │
     └─ 🌤 Weather   What's actually happening?
         (parallel)  Live data for your location

          ↓ Synthesis

  "Wind is 42 km/h. Crane limit 38 km/h → EXCEEDED.
   Glazing limit 30 km/h → EXCEEDED.

   🛑 Crane: SUSPENDED
   🛑 Glazing: SUSPENDED (30 km/h limit)
   ✅ Concrete: GO (temp 21°C, within range)

   Overall Site Risk: HIGH"
""", language="text")

    st.divider()

# ── Render chat history ───────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            risk  = msg.get("risk_level", "UNKNOWN")
            cls, label = RISK_STYLE.get(risk, RISK_STYLE["UNKNOWN"])
            st.markdown(f'<div class="{cls}">{label}</div>', unsafe_allow_html=True)

        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            pills = ""
            if msg.get("rag_available"):
                pills += '<span class="src-pill src-rag">📖 Handbook</span>'
            if msg.get("weather_available"):
                pills += f'<span class="src-pill src-wx">🌤 {msg.get("weather_location","Weather")}</span>'
            if msg.get("memory_used"):
                pills += '<span class="src-pill src-mem">🧠 Profile</span>'
            if pills:
                st.markdown(f'<div style="margin-top:8px">{pills}</div>',
                            unsafe_allow_html=True)
            if msg.get("citations"):
                st.caption("📎 " + " · ".join(msg["citations"]))
            if msg.get("hallucinated"):
                st.warning("⚠️ Removed unverifiable citations: " +
                           ", ".join(msg["hallucinated"]), icon="🔍")
            if msg.get("weather_raw"):
                with st.expander(f"📊 Raw weather data — {msg.get('weather_location','')}"):
                    st.code(msg["weather_raw"], language="text")

# ── Chat input ─────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
prompt  = st.chat_input("Ask a site question or request your morning briefing…")
if not prompt and prefill:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        sw = get_sw()

        status_lines: List[str] = []
        status_box   = st.empty()
        risk_banner  = st.empty()
        answer_ph    = st.empty()

        streamed     = ""
        final_result = None

        for event in sw.stream_query(prompt):
            if event["type"] == "status":
                if status_lines and "⏳" in status_lines[-1]:
                    status_lines[-1] = status_lines[-1].replace("⏳", "✓")
                status_lines.append(f"⏳ {event['message']}")
                status_box.markdown(
                    "  ·  ".join(
                        f'<span class="stage-done">{l}</span>'  if "✓" in l else
                        f'<span class="stage-active">{l}</span>'
                        for l in status_lines
                    ),
                    unsafe_allow_html=True,
                )

            elif event["type"] == "token":
                streamed += event["content"]
                answer_ph.markdown(streamed + "▌")

            elif event["type"] == "done":
                final_result = event["result"]

        status_box.empty()
        answer_ph.markdown(streamed)

        if final_result:
            risk  = final_result.risk_level
            cls, label = RISK_STYLE.get(risk, RISK_STYLE["UNKNOWN"])
            risk_banner.markdown(f'<div class="{cls}">{label}</div>',
                                  unsafe_allow_html=True)

            pills = ""
            if final_result.rag_available:
                pills += '<span class="src-pill src-rag">📖 Handbook</span>'
            if final_result.weather_available:
                pills += f'<span class="src-pill src-wx">🌤 {final_result.weather_location}</span>'
            if final_result.memory_available:
                pills += '<span class="src-pill src-mem">🧠 Profile</span>'
            if pills:
                st.markdown(f'<div style="margin-top:8px">{pills}</div>',
                            unsafe_allow_html=True)

            if final_result.citations:
                st.caption("📎 " + " · ".join(final_result.citations))
            # Memory write is now silent — no banner shown
            if final_result.hallucinated:
                st.warning("⚠️ Removed unverifiable citations: " +
                           ", ".join(final_result.hallucinated), icon="🔍")
            if final_result.weather_available and final_result.weather_raw:
                with st.expander(f"📊 Raw weather data — {final_result.weather_location}"):
                    st.code(final_result.weather_raw, language="text")

    # Persist to session history
    r = final_result
    st.session_state.messages.append({
        "role":             "assistant",
        "content":          streamed,
        "risk_level":       r.risk_level        if r else "UNKNOWN",
        "rag_available":    r.rag_available      if r else False,
        "weather_available": r.weather_available if r else False,
        "weather_location": r.weather_location   if r else None,
        "weather_raw":      r.weather_raw        if r else "",
        "memory_used":      r.memory_available   if r else False,
        "citations":        r.citations          if r else [],
        "hallucinated":     r.hallucinated       if r else [],
    })

# ── Session stats footer ───────────────────────────────────────────────────
if st.session_state.messages:
    turns = [m for m in st.session_state.messages if m["role"] == "user"]
    wx_n  = sum(1 for m in st.session_state.messages if m.get("weather_available"))
    doc_n = sum(1 for m in st.session_state.messages if m.get("rag_available"))
    crit  = sum(1 for m in st.session_state.messages if m.get("risk_level") == "CRITICAL")
    high  = sum(1 for m in st.session_state.messages if m.get("risk_level") == "HIGH")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Queries",       len(turns))
    c2.metric("Live weather",  wx_n)
    c3.metric("Handbook used", doc_n)
    c4.metric("High/Critical", high + crit,
              delta=f"{crit} critical" if crit else None,
              delta_color="inverse")
