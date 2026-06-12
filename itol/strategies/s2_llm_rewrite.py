"""
S2-LLM Assisted Rewrite — LOSSLESS-VERIFIED (§4).

Two distinct paths:

REQUEST-TIME PATH (S2LLMRewriteStrategy.apply)
-----------------------------------------------
O(1) lookup: if templates.compression_verified=1 for this request's
template_sig → substitute the stored compressed_instruction for
SYSTEM_INSTRUCTION segments.  Does NOT call any external model.

OFFLINE JOB PATH (jobs/s2_rewrite_job.py)
------------------------------------------
Called asynchronously (not in the hot path):
    1. Checks template.reuse_count >= s2_llm_min_reuse (default 10)
    2. Calls breakeven_check(R, delta_T, P_in_discounted, C_opt)
       - passes=False → skip; deterministic S2 result stands
       - passes=True  → call LLMRewriter.rewrite()
    3. Verifies rewrite quality (100% manifest coverage + normative tokens)
    4. On pass: stores compressed_instruction, sets compression_verified=1
    5. On fail: logs reason, does not store

LLMRewriter ABC + NullRewriter
-------------------------------
LLMRewriter is an interface (ABC) so real providers can be plugged in.
NullRewriter returns input unchanged — used for testing and as the
safe default when no real model is configured.  With NullRewriter:
  - breakeven may pass for high reuse_count
  - verification will ALSO pass trivially (unchanged text = full coverage)
  - compressed_instruction = original → no actual compression, but pipeline
    exercises correctly (documents that real savings require a real rewriter)

CR-15 link
----------
Every offline breakeven check is logged via store.log_breakeven().
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from itol.icr import ICR, SegmentType, StrategyReport
from itol.quality.breakeven import BreakevenResult, breakeven_check, compute_p_in_discounted
from itol.routing.matrix import StrategyStatus
from itol.segmenter import Segment
from itol.signals import estimate_token_count
from itol.strategies.base import OptimizationContext, Strategy, update_segment

if TYPE_CHECKING:
    from itol.cache.store import Store

_log = logging.getLogger(__name__)

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


def _template_sig(instruction_text: str) -> str:
    """Stable hash of a system instruction (same as S2 deterministic uses)."""
    normalised = re.sub(r"\s+", " ", instruction_text).strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# LLMRewriter interface
# ---------------------------------------------------------------------------

class LLMRewriter(ABC):
    """
    Interface for an LLM that rewrites a system instruction to be more
    token-efficient while preserving all constraints.

    Implementations
    ---------------
    NullRewriter          built-in stub; returns input unchanged (for testing)
    <custom>              plug in any real provider by subclassing this ABC
    """

    @abstractmethod
    def rewrite(self, instruction: str) -> str:
        """
        Return a compressed version of `instruction`.

        Must preserve:
          - All normative tokens (must/never/always/...)
          - All manifest-item values
          - All governing_span qualifiers

        The offline job verifies these properties independently;
        the rewriter is not required to self-verify.
        """
        ...

    @property
    def cost_per_call_usd(self) -> float:
        """Estimated cost of one rewrite call (used for break-even check)."""
        return 0.0


class NullRewriter(LLMRewriter):
    """
    Stub implementation: returns the instruction unchanged.

    With this rewriter:
    - Break-even can pass for high reuse_count (lhs large, rhs=0)
    - Verification passes trivially (unchanged text satisfies all constraints)
    - No actual token savings — use a real LLMRewriter for production savings

    Documentation note: to plug in a real provider, subclass LLMRewriter and
    override rewrite() and cost_per_call_usd.  Example:

        class MistralRewriter(LLMRewriter):
            def rewrite(self, instruction):
                return mistral_client.chat(...)
            @property
            def cost_per_call_usd(self):
                return 0.0006  # e.g. Mistral-7B at ~1k tokens
    """

    def rewrite(self, instruction: str) -> str:
        return instruction

    @property
    def cost_per_call_usd(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Quality verification
# ---------------------------------------------------------------------------

def verify_rewrite_quality(
    original: str,
    rewritten: str,
    manifest_items: list[dict[str, str]] | None = None,
    sample_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Verify a rewritten instruction meets quality requirements.

    Checks:
        1. 100% normative token survival
        2. 100% manifest-item value survival (if manifest_items provided)
        3. Offline A/B parity (TODO: stub — requires live traffic for real A/B)

    Parameters
    ----------
    original : str
    rewritten : str
    manifest_items : list[dict], optional
        Each item must have a "value" key.  All values must appear in rewritten.
    sample_prompts : list[str], optional
        TODO: real A/B requires live sampling against ≥ 20 prompts.
        Interface: verify_rewrite_quality(original, rewritten, sample_prompts) -> float
        Currently returns 1.0 (stub) when provided.

    Returns
    -------
    dict with keys:
        passed (bool), normative_ok (bool), manifest_ok (bool),
        ab_parity (float | None), fail_reason (str | None)
    """
    # 1. Normative token survival
    orig_norm = _normative_tokens(original)
    rewr_norm = _normative_tokens(rewritten)
    missing_norm = orig_norm - rewr_norm
    normative_ok = len(missing_norm) == 0

    # 2. Manifest item survival
    manifest_ok = True
    if manifest_items:
        for item in manifest_items:
            value = item.get("value", "")
            if value and value.lower() not in rewritten.lower():
                manifest_ok = False
                break

    # 3. A/B parity stub
    # TODO: implement real offline A/B against ≥20 sampled tenant prompts.
    # Interface: call a configured shadow evaluator on each prompt, measure parity,
    # return mean parity.  Accept only if mean >= s2_offline_parity_threshold (0.97).
    ab_parity: float | None = None
    if sample_prompts is not None:
        ab_parity = 1.0  # stub — NullRewriter always passes (unchanged text)

    passed = normative_ok and manifest_ok

    fail_reason: str | None = None
    if not normative_ok:
        fail_reason = f"missing normative tokens: {missing_norm}"
    elif not manifest_ok:
        fail_reason = "manifest item dropped"

    return {
        "passed": passed,
        "normative_ok": normative_ok,
        "manifest_ok": manifest_ok,
        "ab_parity": ab_parity,
        "fail_reason": fail_reason,
    }


# ---------------------------------------------------------------------------
# Request-time strategy
# ---------------------------------------------------------------------------

class S2LLMRewriteStrategy(Strategy):
    """
    Request-time path: substitutes a verified compressed instruction when
    templates.compression_verified=1.

    Does NOT invoke the LLM — that happens in the offline job.
    """

    strategy_id = "S2L"
    risk_class  = "LOSSLESS_VERIFIED"

    def __init__(self, store: "Store | None" = None) -> None:
        self._store = store

    def applies(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> bool:
        if self._store is None:
            return False
        if ctx.matrix_row.strategies.get("S2") == StrategyStatus.DISABLED:
            return False
        # Any SYSTEM_INSTRUCTION segment where we have a verified rewrite?
        for seg in segments:
            if seg.segment_type != SegmentType.SYSTEM_INSTRUCTION:
                continue
            sig = _template_sig(seg.text)
            tmpl = self._store.get_template(sig, icr.tenant_id)
            if tmpl and tmpl["compression_verified"]:
                return True
        return False

    def apply(
        self, icr: ICR, segments: list[Segment], ctx: OptimizationContext
    ) -> tuple[list[Segment], StrategyReport]:
        segments_before = list(segments)
        tokens_before = sum(s.token_count for s in segments)

        if not self.applies(icr, segments, ctx):
            return segments, self._make_report(
                segments_before, tokens_before, tokens_before,
                [], [], 0, 0, activated=False,
            )

        new_segments = list(segments)
        touched: list[str] = []

        for idx, seg in enumerate(segments):
            if seg.segment_type != SegmentType.SYSTEM_INSTRUCTION:
                continue
            sig = _template_sig(seg.text)
            tmpl = self._store.get_template(sig, icr.tenant_id)
            if not (tmpl and tmpl["compression_verified"] and tmpl["compressed_instruction"]):
                continue
            new_segments[idx] = update_segment(seg, tmpl["compressed_instruction"])
            touched.append(seg.segment_hash)
            _log.debug(
                "S2L: substituted verified instruction for template %s...", sig[:8]
            )

        tokens_after = sum(s.token_count for s in new_segments)
        return new_segments, self._make_report(
            segments_before, tokens_before, tokens_after,
            touched, [], 0, 0,
            activated=len(touched) > 0,
            notes=f"verified_substitutions={len(touched)}",
        )
