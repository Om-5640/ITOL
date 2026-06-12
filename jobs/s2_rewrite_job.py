"""
jobs/s2_rewrite_job.py — offline S2-LLM rewrite job.

Called asynchronously (not in the request hot path).  For each candidate
template:
    1. Load from store — verify reuse_count >= N_min
    2. Compute break-even: if fails → log and exit
    3. Call LLMRewriter.rewrite()
    4. Verify quality (normative tokens + manifest coverage)
    5. On pass → store.set_template_compressed(verified=True)
    6. On fail → log reason, do NOT store

CR-15: every breakeven_check call is logged to breakeven_log.

The job is designed to be called from:
    - A cron/scheduler (e.g. daily batch over all templates)
    - A queue worker
    - Directly from tests (with NullRewriter)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from itol.quality.breakeven import BreakevenResult, breakeven_check, compute_p_in_discounted
from itol.strategies.s2_llm_rewrite import LLMRewriter, NullRewriter, verify_rewrite_quality

if TYPE_CHECKING:
    from itol.cache.store import Store

_log = logging.getLogger(__name__)

# Default parameters used when not provided by caller
_DEFAULT_MIN_REUSE = 10
_DEFAULT_PRICE_PER_TOKEN = 3e-6   # $3/MTok
_DEFAULT_CACHE_DISCOUNT = 0.0


def run_for_template(
    template_sig: str,
    tenant_id: str,
    store: "Store",
    rewriter: LLMRewriter | None = None,
    tokens_before: int | None = None,
    tokens_after_deterministic: int | None = None,
    base_price_per_token: float = _DEFAULT_PRICE_PER_TOKEN,
    cache_read_discount: float = _DEFAULT_CACHE_DISCOUNT,
    min_reuse: int | None = None,
    manifest_items: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Run the offline S2-LLM rewrite job for a single template.

    Parameters
    ----------
    template_sig : str
        Template identifier (sha256 hash of the normalised instruction).
    tenant_id : str
    store : Store
    rewriter : LLMRewriter, optional
        Model to use for rewriting.  Defaults to NullRewriter if None.
    tokens_before : int, optional
        Token count of the original instruction.  If None, estimated from
        the stored text.
    tokens_after_deterministic : int, optional
        Token count after the deterministic S2 pass.  delta_T =
        tokens_before - tokens_after_deterministic.  If None, delta_T = 0
        (conservative — break-even will be harder to pass).
    base_price_per_token : float
        List input price for the target model (USD per token).
    cache_read_discount : float
        Provider cache-read discount [0, 1).
    min_reuse : int, optional
        Minimum reuse_count to proceed.  Defaults to _DEFAULT_MIN_REUSE.
    manifest_items : list[dict], optional
        Manifest items to verify against the rewritten instruction.

    Returns
    -------
    dict with keys:
        status   — "skipped_low_reuse" | "breakeven_failed" | "verify_failed" | "ok"
        breakeven (BreakevenResult | None)
        verify   (dict | None)
        reason   (str | None)
    """
    if rewriter is None:
        rewriter = NullRewriter()

    n_min = min_reuse if min_reuse is not None else _DEFAULT_MIN_REUSE

    # 1. Load template
    tmpl = store.get_template(template_sig, tenant_id)
    if tmpl is None:
        return {"status": "not_found", "breakeven": None, "verify": None, "reason": "template not found"}

    reuse_count = tmpl["reuse_count"]
    if reuse_count < n_min:
        _log.debug(
            "S2L job: template %s... reuse_count=%d < %d — skipping",
            template_sig[:8], reuse_count, n_min,
        )
        return {
            "status": "skipped_low_reuse",
            "breakeven": None,
            "verify": None,
            "reason": f"reuse_count={reuse_count} < min_reuse={n_min}",
        }

    # 2. Break-even check
    original_text = tmpl.get("compressed_instruction") or ""
    # Use reuse_count directly as proxy for expected reuses (documented simplification:
    # a proper estimate would model future traffic decay, but reuse_count is a
    # conservative lower bound on total future uses).
    R = reuse_count
    T_before = tokens_before or _estimate_tokens(original_text)
    T_after  = tokens_after_deterministic or T_before
    delta_T  = max(0, T_before - T_after)

    P_in_disc = compute_p_in_discounted(base_price_per_token, cache_read_discount)
    C_opt = rewriter.cost_per_call_usd

    breakeven_result = breakeven_check(
        R=R,
        delta_T=delta_T,
        P_in_discounted=P_in_disc,
        C_opt=C_opt,
        store=store,
        context={
            "strategy_id": "S2L",
            "tenant_id": tenant_id,
            "template_sig": template_sig,
        },
    )

    if not breakeven_result.passed:
        _log.info(
            "S2L job: template %s... breakeven failed (lhs=%.4f rhs=%.4f)",
            template_sig[:8], breakeven_result.lhs, breakeven_result.rhs,
        )
        return {
            "status": "breakeven_failed",
            "breakeven": breakeven_result,
            "verify": None,
            "reason": f"lhs={breakeven_result.lhs:.4f} <= rhs={breakeven_result.rhs:.4f}",
        }

    # 3. Rewrite
    try:
        rewritten = rewriter.rewrite(original_text)
    except Exception as exc:
        _log.error("S2L job: rewriter raised: %s", exc)
        return {
            "status": "rewriter_error",
            "breakeven": breakeven_result,
            "verify": None,
            "reason": str(exc),
        }

    # 4. Verify quality
    verify_result = verify_rewrite_quality(
        original=original_text,
        rewritten=rewritten,
        manifest_items=manifest_items,
    )

    if not verify_result["passed"]:
        _log.warning(
            "S2L job: template %s... verification failed: %s",
            template_sig[:8], verify_result["fail_reason"],
        )
        return {
            "status": "verify_failed",
            "breakeven": breakeven_result,
            "verify": verify_result,
            "reason": verify_result["fail_reason"],
        }

    # 5. Store verified rewrite
    store.set_template_compressed(
        template_sig=template_sig,
        tenant_id=tenant_id,
        compressed_instruction=rewritten,
        verified=True,
    )
    _log.info(
        "S2L job: template %s... verified and stored (reuse=%d, delta_T=%d)",
        template_sig[:8], R, delta_T,
    )
    return {
        "status": "ok",
        "breakeven": breakeven_result,
        "verify": verify_result,
        "reason": None,
    }


def _estimate_tokens(text: str) -> int:
    """Fast token estimate (4 chars per token heuristic)."""
    return max(1, len(text) // 4)


def run_batch(
    tenant_id: str,
    store: "Store",
    rewriter: LLMRewriter | None = None,
    min_reuse: int = _DEFAULT_MIN_REUSE,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Run the offline job for ALL templates belonging to tenant_id that have
    reuse_count >= min_reuse and compression_verified=0.

    Returns a list of per-template result dicts.
    """
    rewriter = rewriter or NullRewriter()
    conn = store._conn()
    rows = conn.execute(
        "SELECT template_sig FROM templates "
        "WHERE tenant_id=? AND reuse_count>=? AND (compression_verified=0 OR compression_verified IS NULL)",
        (tenant_id, min_reuse),
    ).fetchall()

    results = []
    for (sig,) in rows:
        r = run_for_template(
            template_sig=sig,
            tenant_id=tenant_id,
            store=store,
            rewriter=rewriter,
            min_reuse=min_reuse,
            **kwargs,
        )
        results.append({"template_sig": sig, **r})
    return results
