"""
Segmenter — Ingestion & Analysis Layer, §3.2

Splits an ICR into typed Segments.  Detection is deterministic (no ML):
role boundaries → grammar sniffing → heuristics.  Each segment gets a
stable sha256 hash over normalised text.

Dependency-free beyond the standard library.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from itol.icr import (
    ContentBlock,
    ContentType,
    ICR,
    Message,
    SegmentType,
    ToolDef,
)


# ---------------------------------------------------------------------------
# Segment — the atomic unit downstream components operate on
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """
    A typed, hashed slice of an ICR.

    `text` is the raw string content.  `segment_hash` is stable across
    requests for identical normalised content — it powers S1 dedupe,
    L2 template lookup, and doc-store addressing.
    """
    segment_type: SegmentType
    text: str
    segment_hash: str                      # sha256 of normalised text
    source_message_index: int | None       # index into ICR.messages; None = system
    source_block_index: int | None         # index within the message's content list
    token_count: int | None = None         # filled by the token-counter (§3.3)
    metadata: dict = field(default_factory=dict)  # e.g. {"tool_name": "search"}


# ---------------------------------------------------------------------------
# Normalisation  (§3.2 — "normalise" definition)
# ---------------------------------------------------------------------------

# Outside of code/data blocks, collapse runs of whitespace to a single space
# and strip leading/trailing whitespace.  Case is preserved (case is semantic).
_WS_RUN = re.compile(r"[ \t]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")


def _normalise(text: str) -> str:
    """Normalise whitespace outside of content used for hashing."""
    text = _WS_RUN.sub(" ", text)
    text = _NEWLINE_RUN.sub("\n\n", text)
    return text.strip()


def _hash(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Repeated doc-delimiter patterns (§3.2 retrieved-doc detection)
_DOC_DELIMITER = re.compile(
    r"(?:^|\n)(?:---+|<doc>|</doc>|Source:|Document \d+|## Document|\[Doc \d+\])",
    re.MULTILINE,
)

# Few-shot: repeated Input/Output or Q/A pair patterns
_FEW_SHOT = re.compile(
    r"(?:(?:Input|Question|Q):\s*.+\n(?:Output|Answer|A):\s*.+\n?){2,}",
    re.IGNORECASE,
)

# Code fence (``` or ~~~, optionally with language tag)
_CODE_FENCE_START = re.compile(r"^(`{3,}|~{3,})\s*\w*\s*$", re.MULTILINE)

# JSON sniff: attempt parse; succeed → STRUCTURED_DATA
def _is_json(text: str) -> bool:
    stripped = text.strip()
    if not (stripped.startswith(("{", "[")) and stripped.endswith(("}", "]"))):
        return False
    try:
        json.loads(stripped)
        return True
    except (json.JSONDecodeError, ValueError):
        return False

# CSV sniff: ≥2 lines, each has the same number of comma-separated fields ≥2
def _is_csv(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return False
    counts = [len(l.split(",")) for l in lines[:5]]
    return len(set(counts)) == 1 and counts[0] >= 2

# Normative verbs that flag a SYSTEM_INSTRUCTION segment
_NORMATIVE_VERBS = re.compile(
    r"\b(?:must|never|always|only|exactly|do not|required|forbidden|"
    r"shall|should not|you are|your (?:task|job|role|goal))\b",
    re.IGNORECASE,
)

# "summarize", "tldr", "tl;dr" etc. in a user turn → SUMMARIZATION hint
_SUMMARIZE_VERBS = re.compile(r"\b(?:summar(?:ise|ize)|tl[;,]?dr|overview)\b", re.IGNORECASE)

# Threshold: a text block is "long" (candidate for RETRIEVED_DOC) if it has
# more than this many words AND delimiter patterns appear.
_RETRIEVED_DOC_WORD_THRESHOLD = 80

# Minimum word count for a block to be considered a retrieved document even
# without delimiters (length-only heuristic).
_RETRIEVED_DOC_LENGTH_ONLY_THRESHOLD = 300


def _classify_text_block(
    text: str,
    role: str,
    is_system_role: bool,
    has_tool_results_in_conversation: bool,
) -> SegmentType:
    """
    Deterministically assign a SegmentType to a text content block.

    Priority order matches §3.2.
    """
    stripped = text.strip()
    word_count = len(stripped.split())

    # 1. Code block (fenced)
    if _CODE_FENCE_START.search(stripped):
        return SegmentType.CODE_BLOCK

    # 2. Structured data (JSON / CSV)
    if _is_json(stripped):
        return SegmentType.STRUCTURED_DATA
    if _is_csv(stripped):
        return SegmentType.STRUCTURED_DATA

    # 3. Few-shot examples (role-independent — can appear in system or user)
    if _FEW_SHOT.search(stripped):
        return SegmentType.FEW_SHOT_EXAMPLE

    # 4. Retrieved document — delimiter pattern OR long non-system block
    if role != "system" and not is_system_role:
        if _DOC_DELIMITER.search(stripped) and word_count >= _RETRIEVED_DOC_WORD_THRESHOLD:
            return SegmentType.RETRIEVED_DOC
        if word_count >= _RETRIEVED_DOC_LENGTH_ONLY_THRESHOLD and role != "assistant":
            return SegmentType.RETRIEVED_DOC

    # 5. System/instruction text
    if is_system_role or role == "system":
        return SegmentType.SYSTEM_INSTRUCTION

    # 6. Assistant turn
    if role == "assistant":
        return SegmentType.ASSISTANT_TURN

    # 7. User query (default for user role)
    return SegmentType.USER_QUERY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_icr(icr: ICR) -> list[Segment]:
    """
    Split an ICR into an ordered list of Segments.

    Order: system blocks first, then messages in conversation order,
    then tool definitions.  Within a message, content blocks are
    processed in list order.
    """
    segments: list[Segment] = []

    # --- System content ---
    for i, block in enumerate(icr.system):
        if block.type is ContentType.TEXT and block.text:
            seg = _segment_from_text_block(
                text=block.text,
                role="system",
                msg_index=None,
                block_index=i,
                has_tool_results=_has_tool_results(icr.messages),
            )
            segments.append(seg)

    # --- Messages ---
    has_tool_results = _has_tool_results(icr.messages)
    for msg_i, msg in enumerate(icr.messages):
        for blk_i, block in enumerate(msg.content):
            seg = _segment_from_block(
                block=block,
                msg=msg,
                msg_index=msg_i,
                block_index=blk_i,
                has_tool_results=has_tool_results,
            )
            if seg is not None:
                segments.append(seg)

    # --- Tool definitions ---
    for i, tool in enumerate(icr.tools):
        text = _tool_def_to_text(tool)
        segments.append(Segment(
            segment_type=SegmentType.TOOL_SCHEMA,
            text=text,
            segment_hash=_hash(text),
            source_message_index=None,
            source_block_index=i,
            metadata={"tool_name": tool.name},
        ))

    return segments


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_tool_results(messages: list[Message]) -> bool:
    for msg in messages:
        for block in msg.content:
            if block.type in (ContentType.TOOL_USE, ContentType.TOOL_RESULT):
                return True
    return False


def _segment_from_text_block(
    text: str,
    role: str,
    msg_index: int | None,
    block_index: int,
    has_tool_results: bool,
) -> Segment:
    is_system_role = role == "system"
    seg_type = _classify_text_block(
        text=text,
        role=role,
        is_system_role=is_system_role,
        has_tool_results_in_conversation=has_tool_results,
    )
    return Segment(
        segment_type=seg_type,
        text=text,
        segment_hash=_hash(text),
        source_message_index=msg_index,
        source_block_index=block_index,
    )


def _segment_from_block(
    block: ContentBlock,
    msg: Message,
    msg_index: int,
    block_index: int,
    has_tool_results: bool,
) -> Segment | None:
    role = msg.role

    if block.type is ContentType.TEXT and block.text is not None:
        return _segment_from_text_block(
            text=block.text,
            role=role,
            msg_index=msg_index,
            block_index=block_index,
            has_tool_results=has_tool_results,
        )

    if block.type is ContentType.TOOL_USE:
        text = json.dumps({
            "tool_use_id": block.tool_use_id,
            "tool_name": block.tool_name,
            "tool_input": block.tool_input,
        }, separators=(",", ":"))
        return Segment(
            segment_type=SegmentType.TOOL_SCHEMA,
            text=text,
            segment_hash=_hash(text),
            source_message_index=msg_index,
            source_block_index=block_index,
            metadata={"tool_name": block.tool_name, "tool_use_id": block.tool_use_id},
        )

    if block.type is ContentType.TOOL_RESULT:
        content = block.text or ""
        return Segment(
            segment_type=SegmentType.TOOL_RESULT,
            text=content,
            segment_hash=_hash(content),
            source_message_index=msg_index,
            source_block_index=block_index,
            metadata={
                "tool_result_for_id": block.tool_result_for_id,
                "is_error": block.is_error,
            },
        )

    # IMAGE and other block types: produce a stub segment so nothing is silently dropped
    if block.type in (ContentType.IMAGE_URL, ContentType.IMAGE_BASE64):
        stub = f"[image block at msg={msg_index} block={block_index}]"
        return Segment(
            segment_type=SegmentType.UNKNOWN,
            text=stub,
            segment_hash=_hash(stub),
            source_message_index=msg_index,
            source_block_index=block_index,
        )

    return None


def _tool_def_to_text(tool: ToolDef) -> str:
    """Stable text representation of a ToolDef for hashing and analysis."""
    return json.dumps(
        {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
        separators=(",", ":"),
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Segment-level utilities used by downstream components
# ---------------------------------------------------------------------------

def segments_full_text(segments: list[Segment]) -> str:
    """Concatenate all segment text in order; used by manifest coverage checks."""
    return "\n".join(s.text for s in segments)


def filter_by_type(segments: list[Segment], *types: SegmentType) -> list[Segment]:
    return [s for s in segments if s.segment_type in types]


def template_signature(segments: list[Segment]) -> str:
    """
    §3.3 template_signature: sha256 of the segment-type sequence concatenated
    with the hashes of SYSTEM_INSTRUCTION segments.

    Stable across requests that share the same template (same system prompt
    + same structural shape) regardless of variable content in user turns.
    """
    type_seq = ",".join(s.segment_type.value for s in segments)
    system_hashes = "".join(
        s.segment_hash
        for s in segments
        if s.segment_type is SegmentType.SYSTEM_INSTRUCTION
    )
    return hashlib.sha256(f"{type_seq}|{system_hashes}".encode()).hexdigest()
