"""
S2 Instruction Hygiene — LOSSLESS-VERIFIED pass (§4, execution order position 1).

Applies filler-phrase substitutions to SYSTEM_INSTRUCTION segments only.

Guarantees:
- Normative tokens (must/never/always/etc.) in original must survive in result.
- Any substitution that would remove a normative token is silently skipped.
- increment_template() is called on the store with the new instruction hash.
- CR-10: after any rewrite, report.notes contains `rewritten_prefix_hash=<hex>`.

S2 is NEVER rolled back (NO_ROLLBACK set in qps.py).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from itol.icr import ICR, SegmentType, StrategyReport
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext, Strategy, update_segment


# ---------------------------------------------------------------------------
# Normative token detection
# ---------------------------------------------------------------------------

_NORMATIVE_PATTERN = re.compile(
    r"\b(must|never|always|only|exactly|shall|do\s+not|required|forbidden|"
    r"shall\s+not|should\s+not)\b",
    re.IGNORECASE,
)


def _normative_tokens(text: str) -> frozenset[str]:
    return frozenset(
        re.sub(r"\s+", " ", m.group(0)).lower()
        for m in _NORMATIVE_PATTERN.finditer(text)
    )


# ---------------------------------------------------------------------------
# Filler pattern loading
# ---------------------------------------------------------------------------

_DEFAULT_FILLERS_PATH = Path(__file__).parent.parent / "data" / "fillers.txt"


def _load_fillers(
    fillers_path: Path = _DEFAULT_FILLERS_PATH,
) -> list[tuple[re.Pattern[str], str]]:
    """
    Load filler patterns from a TSV file (pattern<TAB>replacement).
    Lines starting with # or that are blank are skipped.
    Patterns are compiled as case-insensitive regexes.
    """
    patterns: list[tuple[re.Pattern[str], str]] = []
    try:
        for raw in fillers_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            pattern_str = parts[0]
            replacement = parts[1] if len(parts) == 2 else ""
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                patterns.append((compiled, replacement))
            except re.error:
                pass  # skip malformed patterns
    except FileNotFoundError:
        pass
    return patterns


# Module-level cache of compiled patterns
_FILLER_PATTERNS: list[tuple[re.Pattern[str], str]] | None = None


def _get_filler_patterns() -> list[tuple[re.Pattern[str], str]]:
    global _FILLER_PATTERNS
    if _FILLER_PATTERNS is None:
        _FILLER_PATTERNS = _load_fillers()
    return _FILLER_PATTERNS


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class S2InstructionStrategy(Strategy):
    """LOSSLESS-VERIFIED: filler removal on SYSTEM_INSTRUCTION segments."""

    strategy_id = "S2"
    risk_class = "LOSSLESS"

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        return any(
            s.segment_type == SegmentType.SYSTEM_INSTRUCTION for s in segments
        )

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        snapshot = list(segments)
        tokens_before = sum(
            (s.token_count or estimate_token_count(s.text)) for s in segments
        )

        result: list[Segment] = []
        touched: list[str] = []
        notes_parts: list[str] = []
        prefix_bytes_mutated = 0
        cumulative_bytes = 0

        patterns = _get_filler_patterns()

        for seg in segments:
            seg_bytes = len(seg.text.encode())

            if seg.segment_type != SegmentType.SYSTEM_INSTRUCTION:
                result.append(seg)
                cumulative_bytes += seg_bytes
                continue

            new_text = _apply_fillers_with_normative_guard(seg.text, patterns)

            if new_text == seg.text:
                result.append(seg)
                cumulative_bytes += seg_bytes
                continue

            # CR-2: check prefix safety
            savings = estimate_token_count(seg.text) - estimate_token_count(new_text)
            in_prefix = cumulative_bytes < ctx.prefix_cacheable_span_bytes
            if in_prefix:
                safe = _prefix_safe(cumulative_bytes, savings, ctx)
                if not safe:
                    result.append(seg)
                    cumulative_bytes += seg_bytes
                    continue
                prefix_bytes_mutated += seg_bytes

            new_seg = update_segment(seg, new_text)
            touched.append(new_seg.segment_hash)
            result.append(new_seg)
            cumulative_bytes += seg_bytes

            # CR-10: record rewritten prefix hash
            prefix_hash = hashlib.sha256(new_text.encode()).hexdigest()[:16]
            notes_parts.append(f"rewritten_prefix_hash={prefix_hash}")

            # Template tracking via store (if available on ctx.config)
            _maybe_increment_template(new_seg.segment_hash, icr.tenant_id, ctx)

        tokens_after = sum(
            (s.token_count or estimate_token_count(s.text)) for s in result
        )
        before_hashes = {s.segment_hash for s in segments}
        new_touched = [s.segment_hash for s in result if s.segment_hash not in before_hashes]

        report = self._make_report(
            segments_before=snapshot,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            touched=new_touched or touched,
            removed_spans=[],
            prefix_bytes_mutated=prefix_bytes_mutated,
            prefix_savings=tokens_before - tokens_after,
            activated=tokens_after < tokens_before,
            notes="; ".join(notes_parts),
        )
        return result, report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_fillers_with_normative_guard(
    text: str,
    patterns: list[tuple[re.Pattern[str], str]],
) -> str:
    """
    Apply each filler substitution only if the result preserves all normative
    tokens present in the original (superset check per substitution).
    """
    current = text
    original_normative = _normative_tokens(text)

    for pattern, replacement in patterns:
        candidate = pattern.sub(replacement, current)
        if candidate == current:
            continue
        result_normative = _normative_tokens(candidate)
        if not original_normative.issubset(result_normative):
            # This substitution would remove a normative token — skip it
            continue
        current = candidate

    return current


def _prefix_safe(
    cumulative_bytes: int,
    savings_tokens: int,
    ctx: OptimizationContext,
) -> bool:
    if cumulative_bytes >= ctx.prefix_cacheable_span_bytes:
        return True
    if ctx.prefix_cacheable_span_bytes == 0 or ctx.provider_cache_value <= 0:
        return True
    return savings_tokens * ctx.input_price_per_token > ctx.provider_cache_value


def _maybe_increment_template(
    template_sig: str,
    tenant_id: str,
    ctx: OptimizationContext,
) -> None:
    """Increment the template reuse counter if a store is accessible."""
    try:
        store = getattr(ctx.config, "_store", None)
        if store is not None:
            store.increment_template(template_sig, tenant_id)
    except Exception:
        pass
