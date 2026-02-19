"""
Persistent Memory Manager
==========================
After each conversation turn the bot evaluates whether any high-signal
facts were revealed and, if so, appends them selectively to:
  • USER_MEMORY.md   – user-specific facts
  • COMPANY_MEMORY.md – org-wide reusable learnings

Decision structure:
  {
    "should_write": bool,
    "target": "user" | "company" | "none",
    "summary": "One-sentence fact",
    "confidence": 0.0-1.0
  }

Safeguards:
  • Only writes when confidence >= MEMORY_CONFIDENCE_THRESHOLD
  • Deduplication: LLM checks new fact against existing memories before writing
  • Never dumps transcripts. Never stores secrets.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from app.config import (
    USER_MEMORY_FILE,
    COMPANY_MEMORY_FILE,
    MEMORY_CONFIDENCE_THRESHOLD,
)


# ── File initialisation ───────────────────────────────────────────────────

def _ensure_memory_files():
    for path, title in [
        (USER_MEMORY_FILE, "User Memory"),
        (COMPANY_MEMORY_FILE, "Company / Org Memory"),
    ]:
        if not path.exists():
            path.write_text(
                f"# {title}\n\n"
                "_No memories recorded yet._\n",
                encoding="utf-8",
            )


# ── Memory decision prompt ────────────────────────────────────────────────

MEMORY_SYSTEM_PROMPT = """You are a memory extraction agent for a construction site assistant.

Your job: decide if the USER's message contains a fact worth storing for future sessions.

STORE these types of facts (when explicitly stated by the USER):
  USER memory (personal, role-based):
    - Role / title:        "I'm the site safety manager" → "User is the site safety manager."
    - Name:               "I'm John" → "User's name is John."
    - Site location:      "I manage the Houston site" → "User manages the Houston site."
    - Preferences:        "I prefer metric units" → "User prefers metric units."
    - Recurring schedule: "I have a 8 AM daily briefing" → "User has a daily 8 AM briefing."
    - Active works:       "We have crane ops on level 12" → "Active crane operations on level 12."

  COMPANY memory (org / site-wide):
    - Custom thresholds:  "Council set our scaffold limit to 25 km/h" → "Site scaffold wind limit is 25 km/h (council mandate)."
    - Team structure:     "We have 3 glazing crews" → "Site has 3 glazing crews."
    - Site rules:         "Concrete pours only on weekday mornings" → "Concrete pours scheduled weekday mornings only."
    - Org processes:      "We do daily standups at 9 AM" → "Team has daily standups at 9 AM."

DO NOT STORE:
- Assistant's analysis, advice, or conclusions (only USER statements count)
- Single weather readings ("wind is 42 km/h today" is today's data, not a site fact)
- One-off task completions ("I just finished the pour" — not a standing fact)
- Sensitive data, credentials, or anything already in existing_memories

Target selection:
  "user"    — personal facts about this specific user (role, name, their own schedule)
  "company" — facts about the site/org that any team member would share (thresholds, team, rules)

Respond with ONLY valid JSON (no markdown, no extra text):
{
  "should_write": true | false,
  "target": "user" | "company" | "none",
  "summary": "One-sentence declarative fact (present tense)",
  "confidence": 0.0 to 1.0
}

If nothing worth storing, return:
{"should_write": false, "target": "none", "summary": "", "confidence": 0.0}
"""

DEDUP_CHECK_PROMPT = """You are a memory deduplication agent.

Existing memories:
{existing}

Proposed new memory: "{new_fact}"

Is this new fact already captured (same meaning, even if worded differently) in the existing memories?
Respond with ONLY valid JSON:
{{"is_duplicate": true | false}}
"""


def evaluate_memory(
    user_message: str,
    assistant_response: str,
    llm_client,
    target_hint: str = "",
    user_mem_file: Optional[Path] = None,
    company_mem_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM to evaluate if this exchange contains memorable facts.
    Passes existing memories in context to help avoid duplicates at the decision stage.
    Returns the parsed decision dict.
    """
    # Build existing memory context to guide the LLM
    existing_user    = _extract_bullet_lines(user_mem_file or USER_MEMORY_FILE)
    existing_company = _extract_bullet_lines(company_mem_file or COMPANY_MEMORY_FILE)
    existing_block   = ""
    if existing_user:
        existing_block += "User memory:\n" + "\n".join(existing_user) + "\n"
    if existing_company:
        existing_block += "Company memory:\n" + "\n".join(existing_company) + "\n"

    system = MEMORY_SYSTEM_PROMPT
    if existing_block:
        system += f"\n\nEXISTING MEMORIES (do not duplicate):\n{existing_block}"

    exchange = (
        f"USER said: {user_message}\n\n"
        f"ASSISTANT replied (for context only — do NOT extract facts from this): "
        f"{assistant_response[:400]}"
    )

    raw = llm_client.complete_with_system(
        system=system,
        user=exchange,
        temperature=0.0,
        max_tokens=200,
    )

    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        decision = json.loads(raw)
        if not isinstance(decision.get("should_write"), bool):
            decision["should_write"] = False
        if decision.get("target") not in ("user", "company", "none"):
            decision["target"] = "none"
        if not isinstance(decision.get("confidence"), (int, float)):
            decision["confidence"] = 0.0
        return decision
    except (json.JSONDecodeError, ValueError):
        return {
            "should_write": False,
            "target": "none",
            "summary": "",
            "confidence": 0.0,
        }


def _extract_bullet_lines(path: Path) -> list:
    """Extract bullet-point lines from a memory file."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip().startswith("-")]


def _is_duplicate(
    new_fact: str,
    target: str,
    llm_client,
    user_mem_file: Optional[Path] = None,
    company_mem_file: Optional[Path] = None,
) -> bool:
    path = (user_mem_file or USER_MEMORY_FILE) if target == "user" else (company_mem_file or COMPANY_MEMORY_FILE)
    existing_lines = _extract_bullet_lines(path)

    if not existing_lines:
        return False   # nothing to duplicate against

    existing_str = "\n".join(existing_lines)
    prompt = DEDUP_CHECK_PROMPT.format(
        existing=existing_str,
        new_fact=new_fact,
    )

    raw = llm_client.complete(prompt, temperature=0.0, max_tokens=60)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        result = json.loads(raw)
        return bool(result.get("is_duplicate", False))
    except (json.JSONDecodeError, ValueError):
        return False   # on parse failure, allow the write (conservative)


# ── Memory writer ─────────────────────────────────────────────────────────

def append_memory(
    target: str,
    summary: str,
    confidence: float,
    user_mem_file: Optional[Path] = None,
    company_mem_file: Optional[Path] = None,
) -> Optional[str]:
    """Append a memory entry to the appropriate markdown file."""
    if target == "user":
        path = user_mem_file or USER_MEMORY_FILE
    elif target == "company":
        path = company_mem_file or COMPANY_MEMORY_FILE
    else:
        return None

    # Ensure parent directory and file exist (handles per-user paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        title = "User Memory" if target == "user" else "Company / Org Memory"
        path.write_text(
            f"# {title}\n\n"
            "<!-- SiteWatch auto-managed memory.\n"
            "Do NOT dump raw conversation.\n"
            "Avoid secrets or sensitive information.\n"
            "-->\n\n",
            encoding="utf-8",
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"- [{timestamp}] (confidence: {confidence:.2f}) {summary}\n"

    content = path.read_text(encoding="utf-8")

    # Remove placeholder line if present
    content = content.replace("_No memories recorded yet._\n", "")

    path.write_text(content + entry, encoding="utf-8")
    return str(path)


# ── Combined evaluate + write ─────────────────────────────────────────────

def maybe_write_memory(
    user_message: str,
    assistant_response: str,
    llm_client,
    user_mem_file: Optional[Path] = None,
    company_mem_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: evaluate → dedup check → conditionally write.
    Returns result dict with keys: wrote, target, summary, confidence, file.
    """
    decision = evaluate_memory(
        user_message, assistant_response, llm_client,
        user_mem_file=user_mem_file,
        company_mem_file=company_mem_file,
    )

    if not (
        decision["should_write"]
        and decision["confidence"] >= MEMORY_CONFIDENCE_THRESHOLD
        and decision["summary"].strip()
        and decision["target"] != "none"
    ):
        return {
            "wrote":      False,
            "target":     decision.get("target", "none"),
            "summary":    "",
            "confidence": decision.get("confidence", 0.0),
            "file":       None,
            "skipped_reason": "low_confidence_or_empty",
        }

    # Deduplication check: don't write if this fact is already stored
    if _is_duplicate(
        decision["summary"], decision["target"], llm_client,
        user_mem_file=user_mem_file,
        company_mem_file=company_mem_file,
    ):
        return {
            "wrote":      False,
            "target":     decision["target"],
            "summary":    decision["summary"],
            "confidence": decision["confidence"],
            "file":       None,
            "skipped_reason": "duplicate",
        }

    file_written = append_memory(
        decision["target"],
        decision["summary"],
        decision["confidence"],
        user_mem_file=user_mem_file,
        company_mem_file=company_mem_file,
    )
    return {
        "wrote":          True,
        "target":         decision["target"],
        "summary":        decision["summary"],
        "confidence":     decision["confidence"],
        "file":           file_written,
        "skipped_reason": None,
    }


# ── Read memories for context injection ───────────────────────────────────

def read_memory(target: str) -> str:
    """Return current contents of a memory file."""
    _ensure_memory_files()
    path = USER_MEMORY_FILE if target == "user" else COMPANY_MEMORY_FILE
    return path.read_text(encoding="utf-8")


def build_memory_context(
    user_mem_file: Optional[Path] = None,
    company_mem_file: Optional[Path] = None,
) -> str:
    """Build a short memory context block for LLM prompts."""
    user_mem    = (user_mem_file    or USER_MEMORY_FILE).read_text(encoding="utf-8")    if (user_mem_file    or USER_MEMORY_FILE).exists()    else ""
    company_mem = (company_mem_file or COMPANY_MEMORY_FILE).read_text(encoding="utf-8") if (company_mem_file or COMPANY_MEMORY_FILE).exists() else ""

    lines = []
    if user_mem and "_No memories recorded yet._" not in user_mem:
        lines.append("=== User Memory ===\n" + user_mem)
    if company_mem and "_No memories recorded yet._" not in company_mem:
        lines.append("=== Company Memory ===\n" + company_mem)

    return "\n\n".join(lines) if lines else ""
