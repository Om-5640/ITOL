"""
§7.2 Compatibility matrix — strategy + cache permissions per request class.

This module is a pure data constant.  The classifier produces a class;
routing reads this matrix to decide which strategies and cache tiers are
permitted.  No logic lives here — only the encoded spec table.

Strategy status values
----------------------
ALLOWED    — strategy may run unrestricted
DISABLED   — strategy must not run for this class
RESTRICTED — strategy may run but must use the given QPS mass threshold

L1 status
---------
ALLOWED    — L1 semantic cache enabled; τ (similarity floor) applies
DISABLED   — L1 cache must NOT be used for this class

Usage
-----
    from itol.routing.matrix import MATRIX, StrategyStatus, L1Status
    entry = MATRIX["EXTRACTION"]
    if entry.strategies["S7"] == StrategyStatus.DISABLED:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StrategyStatus(str, Enum):
    ALLOWED     = "ALLOWED"
    DISABLED    = "DISABLED"
    RESTRICTED  = "RESTRICTED"   # mass threshold applies (see `restrictions` map)


class L1Status(str, Enum):
    ALLOWED  = "ALLOWED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class ClassMatrix:
    """Per-class row of the compatibility matrix."""
    strategies: dict[str, StrategyStatus]           # S1–S7 status
    l1: L1Status
    l1_tau: float | None = None                     # similarity floor when l1=ALLOWED
    restrictions: dict[str, float] = field(default_factory=dict)  # strategy → mass floor


# ---------------------------------------------------------------------------
# §7.2 Matrix — encoded once, never duplicated
# ---------------------------------------------------------------------------
#
# Legend:
#   ✓  = ALLOWED
#   ✗  = DISABLED
#   ⚠  = RESTRICTED (threshold in parentheses)
#   S6✓✓ = ALLOWED + prioritised (treated as ALLOWED here; priority is an
#           execution hint, not a gating decision)
#   S7opt = optional/opportunistic (ALLOWED in matrix; routing opts in if
#           token budget justifies — treated as ALLOWED here)
#
#                  S1  S2  S3      S4   S5  S6  S7      L1          τ
# EXTRACTION:       ✓   ✓   ⚠0.97  ✗    ✓   ✓   ✗   ALLOWED      0.97
# REASONING:        ✓   ✓   ⚠0.97  ✗    ✓   ✓   ✗   ALLOWED      0.97
# SUMMARIZATION:    ✓   ✓   ✓      ✓    ✓   ✓   opt ALLOWED      0.95
# GENERATION_FACTUAL:✓  ✓   ✓      ✓    ✓   ✓   ✗   ALLOWED      0.96
# GENERATION_CREATIVE:✓ ✓   ⚠      ✗    ✓   ✓   ✗   DISABLED     —
# CLASSIFICATION_SHORT:✓✓   ✓      ✗    ✗   ✓   ✗   ALLOWED      0.93
# AGENT_TOOL_LOOP:  ✓   ✓   ✓      ✓    ✓   ✓✓  ✗   DISABLED     —
# CHAT_OPEN:        ✓   ✓   ⚠cons  ✗    ✓   ✓   opt DISABLED     —
# ---------------------------------------------------------------------------

MATRIX: dict[str, ClassMatrix] = {

    "EXTRACTION": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.RESTRICTED,
            "S4": StrategyStatus.DISABLED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.97,
        restrictions={"S3": 0.97},
    ),

    "REASONING": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.RESTRICTED,
            "S4": StrategyStatus.DISABLED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.97,
        restrictions={"S3": 0.97},
    ),

    "SUMMARIZATION": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.ALLOWED,
            "S4": StrategyStatus.ALLOWED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.ALLOWED,    # opt → ALLOWED
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.95,
    ),

    "GENERATION_FACTUAL": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.ALLOWED,
            "S4": StrategyStatus.ALLOWED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.96,
    ),

    "GENERATION_CREATIVE": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.RESTRICTED,
            "S4": StrategyStatus.DISABLED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.DISABLED,
        l1_tau=None,
        restrictions={"S3": 0.97},   # conservative threshold (no explicit value in spec)
    ),

    "CLASSIFICATION_SHORT": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.ALLOWED,
            "S4": StrategyStatus.DISABLED,
            "S5": StrategyStatus.DISABLED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.ALLOWED,
        l1_tau=0.93,
    ),

    "AGENT_TOOL_LOOP": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.ALLOWED,
            "S4": StrategyStatus.ALLOWED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,   # S6✓✓ — prioritised, still ALLOWED
            "S7": StrategyStatus.DISABLED,
        },
        l1=L1Status.DISABLED,
        l1_tau=None,
    ),

    "CHAT_OPEN": ClassMatrix(
        strategies={
            "S1": StrategyStatus.ALLOWED,
            "S2": StrategyStatus.ALLOWED,
            "S3": StrategyStatus.RESTRICTED,
            "S4": StrategyStatus.DISABLED,
            "S5": StrategyStatus.ALLOWED,
            "S6": StrategyStatus.ALLOWED,
            "S7": StrategyStatus.ALLOWED,   # opt → ALLOWED
        },
        l1=L1Status.DISABLED,
        l1_tau=None,
        restrictions={"S3": 0.97},   # conservative ("cons") threshold
    ),
}


def ambiguous_matrix(class_a: str, class_b: str) -> ClassMatrix:
    """
    §3.4 AMBIGUOUS: return the intersection of two classes' permitted strategy sets.

    A strategy is ALLOWED only if it is ALLOWED in BOTH classes.
    RESTRICTED in either → RESTRICTED in intersection (stricter threshold wins).
    DISABLED in either   → DISABLED in intersection.
    L1 is DISABLED if either class disables it; τ takes the higher (stricter) value.
    """
    a = MATRIX[class_a]
    b = MATRIX[class_b]

    merged: dict[str, StrategyStatus] = {}
    merged_restrictions: dict[str, float] = {}

    for s in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        sa, sb = a.strategies[s], b.strategies[s]
        if sa == StrategyStatus.DISABLED or sb == StrategyStatus.DISABLED:
            merged[s] = StrategyStatus.DISABLED
        elif sa == StrategyStatus.RESTRICTED or sb == StrategyStatus.RESTRICTED:
            merged[s] = StrategyStatus.RESTRICTED
            # pick the stricter (higher) threshold
            ta = a.restrictions.get(s, 0.97)
            tb = b.restrictions.get(s, 0.97)
            merged_restrictions[s] = max(ta, tb)
        else:
            merged[s] = StrategyStatus.ALLOWED

    l1_status = (
        L1Status.DISABLED
        if (a.l1 == L1Status.DISABLED or b.l1 == L1Status.DISABLED)
        else L1Status.ALLOWED
    )
    taus = [t for t in (a.l1_tau, b.l1_tau) if t is not None]
    l1_tau = max(taus) if taus and l1_status == L1Status.ALLOWED else None

    return ClassMatrix(
        strategies=merged,
        l1=l1_status,
        l1_tau=l1_tau,
        restrictions=merged_restrictions,
    )
