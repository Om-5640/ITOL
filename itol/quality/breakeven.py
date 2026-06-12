"""
itol/quality/breakeven.py — §4.S5 break-even theorem (CR-15).

The break-even gate prevents LLM-assisted optimisation from costing MORE than
it saves.  It is called by:
  - S2-LLM offline rewrite job (before invoking the rewriter)
  - S5 LLM-assisted distillation path (Step 13, once wired)

Formula (CR-15):
    lhs = R × ΔT × P_in_discounted
    rhs = 5 × C_opt        # 5× safety multiplier
    passed = lhs > rhs

Where:
    R                  number of expected reuses
    ΔT                 tokens saved per reuse (delta_tokens)
    P_in_discounted    target model input price AFTER provider cache discount
    C_opt              cost of the optimisation call itself

CR-15 invariant: every call is logged to the breakeven_log table.

Helper:
    compute_p_in_discounted(base_price, cache_read_discount)
        → base_price × (1 − cache_read_discount)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from itol.cache.store import Store

_log = logging.getLogger(__name__)

# §4.S5 safety multiplier — locked as an invariant, never changed to make tests pass
_SAFETY_MULTIPLIER: float = 5.0


@dataclass
class BreakevenResult:
    """Outcome of a single break-even check."""
    passed: bool
    lhs: float      # R × ΔT × P_in_discounted
    rhs: float      # 5 × C_opt
    ratio: float    # lhs / rhs  (inf when rhs == 0)


def compute_p_in_discounted(
    base_price_per_token: float,
    cache_read_discount: float,
) -> float:
    """
    Apply the provider's cache-read discount to the list input price.

    Parameters
    ----------
    base_price_per_token : float
        List price (USD per input token) for the target model.
    cache_read_discount : float
        Fraction of the price saved on a cache-read hit [0, 1).
        e.g. Anthropic = 0.90; OpenAI = 0.50; no caching = 0.0.

    Returns
    -------
    float
        Discounted price per input token.
    """
    return base_price_per_token * (1.0 - max(0.0, min(1.0, cache_read_discount)))


def breakeven_check(
    R: int,
    delta_T: int,
    P_in_discounted: float,
    C_opt: float,
    store: "Store | None" = None,
    context: dict[str, Any] | None = None,
) -> BreakevenResult:
    """
    §4.S5 CR-15 break-even gate.

    Parameters
    ----------
    R : int
        Expected number of reuses (proxy: template reuse_count).
    delta_T : int
        Tokens saved per reuse (e.g. tokens_before - tokens_after_deterministic).
    P_in_discounted : float
        Input price per token AFTER cache-read discount (use
        compute_p_in_discounted() to obtain this).
    C_opt : float
        Cost (USD) of the optimisation call itself.
    store : Store, optional
        If provided, the result is logged to breakeven_log (CR-15 requirement).
    context : dict, optional
        Arbitrary context info included in the log row (strategy_id, tenant_id,
        template_sig, etc.).

    Returns
    -------
    BreakevenResult
    """
    lhs = R * delta_T * P_in_discounted
    rhs = _SAFETY_MULTIPLIER * C_opt
    ratio = lhs / rhs if rhs > 0 else float("inf")
    passed = lhs > rhs
    result = BreakevenResult(passed=passed, lhs=lhs, rhs=rhs, ratio=ratio)

    _log.debug(
        "breakeven_check R=%d delta_T=%d P=%.2e C_opt=%.2e "
        "lhs=%.4f rhs=%.4f passed=%s",
        R, delta_T, P_in_discounted, C_opt, lhs, rhs, passed,
    )

    if store is not None:
        try:
            store.log_breakeven(
                r_reuses=R,
                delta_t=delta_T,
                p_in_disc=P_in_discounted,
                c_opt=C_opt,
                lhs=lhs,
                rhs=rhs,
                passed=passed,
                context=context,
            )
        except Exception as exc:
            _log.warning("breakeven_log write failed: %s", exc)

    return result
