"""
Stage A rule-based request classifier — §3.4.

Produces a ClassifierResult with:
  - primary   : highest-confidence class name
  - confidence: score of the winning rule
  - ambiguous : True when confidence < 0.6
  - top2      : [primary, second] for AMBIGUOUS downstream routing (§3.4)
  - distribution: {class: score} for all fired rules

Rules are evaluated in fixed precedence order (highest-confidence first).
The first rule whose conditions match wins — no blending.
"""

from __future__ import annotations

import re

from itol.icr import (
    ClassifierResult,
    ContentType,
    ICR,
    SegmentType,
)
from itol.segmenter import segment_icr


# ---------------------------------------------------------------------------
# Keyword sets (compiled once at import)
# ---------------------------------------------------------------------------

_SUMMARIZE_KW   = re.compile(
    r"\b(summarize|summarise|tl;?dr|summary\s+of)\b", re.I
)
_JSON_SCHEMA_KW = re.compile(
    r'"type"\s*:', re.I
)
_JSON_SCHEMA_FULL = re.compile(
    r'("type"\s*:|"properties"\s*:|\$schema)', re.I
)
_EXTRACT_KW     = re.compile(
    r"\b(extract|parse|find\s+all|list\s+all|what\s+is\s+the)\b", re.I
)
_MATH_KW        = re.compile(
    r"\b(solve|calculate|compute|proof|prove|how\s+many|what\s+is\s+\d)\b", re.I
)
_CODE_KW        = re.compile(
    r"\b(write\s+a\s+function|implement|debug|fix\s+this\s+code)\b", re.I
)
_CLASSIFY_KW    = re.compile(
    r"\b(classify|categorize|categorise|which\s+category|label|is\s+this\s+a|sentiment)\b", re.I
)
_CREATIVE_TRIGGER = re.compile(
    r"\b(write\s+a|draft|compose|create\s+a)\b", re.I
)
_CREATIVE_MARKER = re.compile(
    r"\b(story|poem|blog|email|essay|creative)\b", re.I
)
_FACTUAL_TRIGGER = re.compile(
    r"\b(write\s+a|draft|compose|generate\s+a\s+report|write\s+a\s+report)\b", re.I
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_segment_type(segments: list, seg_type: SegmentType) -> bool:
    return any(s.segment_type == seg_type for s in segments)


def _has_tool_result(icr: ICR) -> bool:
    for msg in icr.messages:
        for block in msg.content:
            if block.type == ContentType.TOOL_RESULT:
                return True
    return False


def _system_instruction_text(segments: list) -> str:
    parts = [s.text for s in segments if s.segment_type == SegmentType.SYSTEM_INSTRUCTION]
    return " ".join(parts)


def _final_user_query(icr: ICR) -> str:
    return icr.final_user_query()


def _token_estimate(text: str) -> int:
    return max(1, int(len(text) / 3.8))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(icr: ICR) -> ClassifierResult:
    """
    Stage A deterministic classifier — §3.4.

    Precedence order (highest-confidence rule wins):
      1. Tools present in ICR          → AGENT_TOOL_LOOP 0.95
      2. Tool results in messages      → AGENT_TOOL_LOOP 0.90
      3. Summarize kw + RETRIEVED_DOC  → SUMMARIZATION   0.90
      4. JSON schema in SYSTEM_INSTR   → EXTRACTION      0.88
      5. Extract kw + RETRIEVED_DOC    → EXTRACTION      0.85
      6. Math/code kw                  → REASONING       0.85
      7. Classify kw + short query     → CLASSIFICATION_SHORT 0.87
      8. Creative trigger + marker     → GENERATION_CREATIVE  0.82
      9. Factual trigger (no creative) → GENERATION_FACTUAL   0.80
     10. System prompt present         → GENERATION_FACTUAL   0.72
     11. Default                       → CHAT_OPEN            0.65
    """
    segments = segment_icr(icr)
    query = _final_user_query(icr)
    query_tokens = _token_estimate(query)

    has_retrieved_doc = _has_segment_type(segments, SegmentType.RETRIEVED_DOC)
    has_system_instr  = _has_segment_type(segments, SegmentType.SYSTEM_INSTRUCTION)
    system_text       = _system_instruction_text(segments)

    # Collect all fired (class, confidence) pairs for distribution + top2
    fired: list[tuple[str, float]] = []

    # Rule 1 — Tools present
    if icr.tools:
        fired.append(("AGENT_TOOL_LOOP", 0.95))

    # Rule 2 — Tool results in messages
    if _has_tool_result(icr):
        fired.append(("AGENT_TOOL_LOOP", 0.90))

    # Rule 3 — Summarize keywords + retrieved doc
    if _SUMMARIZE_KW.search(query) and has_retrieved_doc:
        fired.append(("SUMMARIZATION", 0.90))

    # Rule 4 — JSON schema in SYSTEM_INSTRUCTION
    if system_text and _JSON_SCHEMA_FULL.search(system_text):
        fired.append(("EXTRACTION", 0.88))

    # Rule 5 — Extract keywords + retrieved doc
    if _EXTRACT_KW.search(query) and has_retrieved_doc:
        fired.append(("EXTRACTION", 0.85))

    # Rule 6 — Math or code generation keywords
    if _MATH_KW.search(query) or _CODE_KW.search(query):
        fired.append(("REASONING", 0.85))

    # Rule 7 — Classify keywords + short query (< 200 tokens)
    if _CLASSIFY_KW.search(query) and query_tokens < 200:
        fired.append(("CLASSIFICATION_SHORT", 0.87))

    # Rule 8 — Creative trigger + creative marker
    if _CREATIVE_TRIGGER.search(query) and _CREATIVE_MARKER.search(query):
        fired.append(("GENERATION_CREATIVE", 0.82))

    # Rule 9 — Factual trigger, no creative marker
    if _FACTUAL_TRIGGER.search(query) and not _CREATIVE_MARKER.search(query):
        fired.append(("GENERATION_FACTUAL", 0.80))

    # Rule 10 — System prompt present, no other signals
    if has_system_instr and not fired:
        fired.append(("GENERATION_FACTUAL", 0.72))

    # Default fallback
    if not fired:
        fired.append(("CHAT_OPEN", 0.65))

    # Sort by confidence desc, then class name for determinism
    fired.sort(key=lambda x: (-x[1], x[0]))

    # Build distribution (take highest score per class in case of duplicates)
    distribution: dict[str, float] = {}
    for cls, conf in fired:
        if cls not in distribution or conf > distribution[cls]:
            distribution[cls] = conf

    # Winner = first entry after sort
    primary, confidence = fired[0]

    # top2: two distinct classes with highest confidence
    seen: list[str] = []
    for cls, _ in fired:
        if cls not in seen:
            seen.append(cls)
        if len(seen) == 2:
            break
    if len(seen) < 2:
        seen.append(primary)   # degenerate: only one class fired

    return ClassifierResult(
        primary=primary,
        confidence=confidence,
        distribution=distribution,
        top2=seen,
    )
