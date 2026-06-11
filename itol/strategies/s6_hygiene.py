"""
S6 Hygiene — LOSSLESS cleanup pass (§4, execution order position 3).

Sub-operations applied in order:
  (a) Whitespace canonicalization — all segments except CODE_BLOCK / STRUCTURED_DATA
  (b) JSON minification — STRUCTURED_DATA and TOOL_RESULT with valid JSON (CR-19)
  (c) Markdown table de-padding — strip interior cell whitespace
  (d) Tool-result expiry — AGENT_TOOL_LOOP only; superseded results → stub
  (e) Tool-schema pruning — AGENT_TOOL_LOOP only; trim description to first sentence

S6 is NEVER rolled back (NO_ROLLBACK set in qps.py).
"""

from __future__ import annotations

import json
import re

from itol.icr import ICR, SegmentType, StrategyReport
from itol.segmenter import Segment
from itol.signals import estimate_token_count, jaccard_estimate, minhash_signature
from itol.strategies.base import OptimizationContext, Strategy, update_segment


# Tool-result expiry: expire results with more than K_TOOL assistant turns after them
_K_TOOL = 3

# Minimum Jaccard for "superseded by later assistant turn" check
_SUPERSEDE_JACCARD = 0.50


def _prefix_safe_for_seg(
    cumulative_bytes: int,
    seg_bytes: int,
    savings_tokens: int,
    ctx: OptimizationContext,
) -> bool:
    """Module-level CR-2 prefix-safety check for use in sub-operations."""
    span_start = cumulative_bytes
    if span_start >= ctx.prefix_cacheable_span_bytes:
        return True
    if ctx.prefix_cacheable_span_bytes == 0 or ctx.provider_cache_value <= 0:
        return True
    savings_usd = savings_tokens * ctx.input_price_per_token
    return savings_usd > ctx.provider_cache_value


class S6HygieneStrategy(Strategy):
    """LOSSLESS hygiene: whitespace, JSON, table de-pad, tool hygiene."""

    strategy_id = "S6"
    risk_class = "LOSSLESS"

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        return True  # always runs

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        snapshot = list(segments)
        tokens_before = sum(
            (s.token_count or estimate_token_count(s.text)) for s in segments
        )

        result, touched, removed_spans, prefix_bytes_mutated = (
            _apply_all(segments, ctx)
        )

        tokens_after = sum(
            (s.token_count or estimate_token_count(s.text)) for s in result
        )

        before_hashes = {s.segment_hash for s in segments}
        new_touched = [s.segment_hash for s in result if s.segment_hash not in before_hashes]

        report = self._make_report(
            segments_before=snapshot,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            touched=new_touched,
            removed_spans=removed_spans,
            prefix_bytes_mutated=prefix_bytes_mutated,
            prefix_savings=0,
            activated=tokens_after < tokens_before or bool(new_touched),
        )
        return result, report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _apply_all(
    segments: list[Segment],
    ctx: OptimizationContext,
) -> tuple[list[Segment], list[str], list[tuple[int, int]], int]:
    result = list(segments)
    touched: list[str] = []
    removed_spans: list[tuple[int, int]] = []
    prefix_bytes_mutated = 0

    result, ws_touched, ws_pmb = _apply_whitespace(result, ctx)
    touched.extend(ws_touched)
    prefix_bytes_mutated += ws_pmb

    result, json_touched = _apply_json_minify(result)
    touched.extend(json_touched)

    result, tbl_touched = _apply_table_depad(result)
    touched.extend(tbl_touched)

    if ctx.request_class == "AGENT_TOOL_LOOP":
        result, exp_touched = _expire_tool_results(result)
        touched.extend(exp_touched)
        result, prn_touched = _prune_tool_schemas(result)
        touched.extend(prn_touched)

    return result, touched, removed_spans, prefix_bytes_mutated


# ---------------------------------------------------------------------------
# (a) Whitespace canonicalization
# ---------------------------------------------------------------------------

def _apply_whitespace(
    segments: list[Segment],
    ctx: OptimizationContext,
) -> tuple[list[Segment], list[str], int]:
    result: list[Segment] = []
    touched: list[str] = []
    prefix_bytes_mutated = 0
    cumulative_bytes = 0

    for seg in segments:
        seg_bytes = len(seg.text.encode())

        if seg.segment_type in (SegmentType.CODE_BLOCK, SegmentType.STRUCTURED_DATA):
            result.append(seg)
            cumulative_bytes += seg_bytes
            continue

        new_text = _canonicalize_whitespace(seg.text)
        if new_text == seg.text:
            result.append(seg)
            cumulative_bytes += seg_bytes
            continue

        savings = estimate_token_count(seg.text) - estimate_token_count(new_text)
        if not _prefix_safe_for_seg(cumulative_bytes, seg_bytes, savings, ctx):
            result.append(seg)
            cumulative_bytes += seg_bytes
            continue

        in_prefix = cumulative_bytes < ctx.prefix_cacheable_span_bytes
        if in_prefix:
            prefix_bytes_mutated += seg_bytes

        new_seg = update_segment(seg, new_text)
        touched.append(new_seg.segment_hash)
        result.append(new_seg)
        cumulative_bytes += seg_bytes

    return result, touched, prefix_bytes_mutated


def _canonicalize_whitespace(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    joined = "\n".join(lines)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# (b) JSON minification — CR-19
# ---------------------------------------------------------------------------

_JSON_TYPES = frozenset({SegmentType.STRUCTURED_DATA, SegmentType.TOOL_RESULT})


def _apply_json_minify(
    segments: list[Segment],
) -> tuple[list[Segment], list[str]]:
    result: list[Segment] = []
    touched: list[str] = []

    for seg in segments:
        if seg.segment_type not in _JSON_TYPES:
            result.append(seg)
            continue
        new_text, ok = _try_minify_json(seg.text)
        if ok:
            new_seg = update_segment(seg, new_text)
            touched.append(new_seg.segment_hash)
            result.append(new_seg)
        else:
            result.append(seg)

    return result, touched


def _try_minify_json(text: str) -> tuple[str, bool]:
    """
    CR-19: minify JSON with round-trip verification.
    Returns (result_text, success).  On any error, returns (text, False).
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return text, False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text, False
    try:
        minified = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return text, False
    # CR-19 round-trip: re-parsed data must equal original
    try:
        if json.loads(minified) != parsed:
            return text, False
    except json.JSONDecodeError:
        return text, False
    if len(minified) >= len(text):
        return text, False
    return minified, True


# ---------------------------------------------------------------------------
# (c) Markdown table de-padding
# ---------------------------------------------------------------------------

def _apply_table_depad(
    segments: list[Segment],
) -> tuple[list[Segment], list[str]]:
    result: list[Segment] = []
    touched: list[str] = []

    for seg in segments:
        if seg.segment_type in (SegmentType.CODE_BLOCK, SegmentType.STRUCTURED_DATA):
            result.append(seg)
            continue
        new_text = _depad_markdown_tables(seg.text)
        if new_text != seg.text:
            new_seg = update_segment(seg, new_text)
            touched.append(new_seg.segment_hash)
            result.append(new_seg)
        else:
            result.append(seg)

    return result, touched


def _depad_line(line: str) -> str:
    if not line.strip().startswith("|"):
        return line
    before_pipes = line.count("|")
    cells = [c.strip() for c in line.split("|")]
    joined = "|".join(cells)
    if joined.count("|") != before_pipes:
        return line
    return joined


def _depad_markdown_tables(text: str) -> str:
    return "\n".join(_depad_line(l) for l in text.split("\n"))


# ---------------------------------------------------------------------------
# (d) Tool-result expiry
# ---------------------------------------------------------------------------

def _expire_tool_results(
    segments: list[Segment],
) -> tuple[list[Segment], list[str]]:
    """
    Replace stale TOOL_RESULT segments with a compact stub.

    Stale = more than K_TOOL ASSISTANT_TURN segments follow this result
    AND its content is superseded (Jaccard ≥ 0.50) by a later assistant turn.
    Always preserves the most-recent TOOL_RESULT.
    """
    tool_indices = [
        i for i, s in enumerate(segments)
        if s.segment_type == SegmentType.TOOL_RESULT
    ]
    if not tool_indices:
        return segments, []

    most_recent = tool_indices[-1]

    asst_sigs = [
        (i, minhash_signature(s.text))
        for i, s in enumerate(segments)
        if s.segment_type == SegmentType.ASSISTANT_TURN
    ]

    result: list[Segment] = []
    touched: list[str] = []

    for i, seg in enumerate(segments):
        if seg.segment_type != SegmentType.TOOL_RESULT or i == most_recent:
            result.append(seg)
            continue

        asst_after = sum(1 for ai, _ in asst_sigs if ai > i)
        if asst_after <= _K_TOOL:
            result.append(seg)
            continue

        seg_sig = minhash_signature(seg.text)
        superseded = any(
            ai > i and jaccard_estimate(seg_sig, asig) >= _SUPERSEDE_JACCARD
            for ai, asig in asst_sigs
        )
        if not superseded:
            result.append(seg)
            continue

        tool_name = (seg.metadata or {}).get("tool_name", "tool")
        stub = f"[tool_result:{tool_name}:expired]"
        new_seg = update_segment(seg, stub)
        touched.append(new_seg.segment_hash)
        result.append(new_seg)

    return result, touched


# ---------------------------------------------------------------------------
# (e) Tool-schema pruning
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _prune_tool_schemas(
    segments: list[Segment],
) -> tuple[list[Segment], list[str]]:
    """Trim TOOL_SCHEMA description text to the first sentence."""
    result: list[Segment] = []
    touched: list[str] = []

    for seg in segments:
        if seg.segment_type != SegmentType.TOOL_SCHEMA:
            result.append(seg)
            continue
        new_text = _first_sentence(seg.text)
        if new_text != seg.text:
            new_seg = update_segment(seg, new_text)
            touched.append(new_seg.segment_hash)
            result.append(new_seg)
        else:
            result.append(seg)

    return result, touched


def _first_sentence(text: str) -> str:
    m = _SENTENCE_END.search(text)
    return text[: m.start() + 1].strip() if m else text
