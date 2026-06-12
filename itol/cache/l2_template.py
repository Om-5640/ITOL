"""
L2 template-delta cache — §6.1.

Keyed by template_signature (from segmenter.py signals). Stores the
S2-compressed instruction, per-segment strategy plans, token counts,
cache breakpoint positions, and doc-version dependencies (CR-9).

On a template hit the pipeline skips re-segmentation / re-classification
of static segments and only recomputes the dynamic parts.

CR-9: depends_on_doc_versions tracks which external doc versions this plan
assumed; when any of those docs changes, the plan is tombstoned by
invalidation.py / store.invalidate_l2_by_doc().
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# TemplatePlan
# ---------------------------------------------------------------------------

@dataclass
class TemplatePlan:
    """
    A reusable optimisation plan keyed by template_signature.

    strategies_config maps strategy_id → kwargs dict (what params each strategy
    used on the static instruction segments — dynamic query segments are always
    re-processed).

    segment_token_counts maps segment_hash → token_count so the pipeline can
    quickly recompute the total without re-segmenting unchanged segments.

    breakpoint_positions is a list of byte offsets where S2 rewrote instruction
    boundaries; the adapter uses these to inject provider cache-control markers.

    depends_on_doc_versions is empty until S4 (RACR) exists; CR-9 guards it.
    """
    template_sig: str
    tenant_id: str
    strategies_config: dict = field(default_factory=dict)
    segment_token_counts: dict = field(default_factory=dict)
    breakpoint_positions: list[int] = field(default_factory=list)
    depends_on_doc_versions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# L2PlanCache
# ---------------------------------------------------------------------------

class L2PlanCache:
    """
    Template-delta plan cache backed by the persistent Store.

    Plans are invalidated by:
      - Explicit doc-version change (CR-9 — invalidate_by_doc_version).
      - Future: TTL-based eviction in invalidation.py.
    """

    def __init__(self, store: object) -> None:
        self._store = store

    def get_plan(self, template_sig: str, tenant_id: str) -> TemplatePlan | None:
        """Return the stored plan or None if not found."""
        row = self._store.get_l2_plan(template_sig, tenant_id)  # type: ignore[attr-defined]
        if row is None:
            return None
        try:
            d = json.loads(row["plan_json"])
            return TemplatePlan(
                template_sig=d.get("template_sig", template_sig),
                tenant_id=d.get("tenant_id", tenant_id),
                strategies_config=d.get("strategies_config", {}),
                segment_token_counts=d.get("segment_token_counts", {}),
                breakpoint_positions=d.get("breakpoint_positions", []),
                depends_on_doc_versions=d.get("depends_on_doc_versions", []),
                created_at=d.get("created_at", 0.0),
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def store_plan(
        self, template_sig: str, tenant_id: str, plan: TemplatePlan
    ) -> None:
        """Persist a plan (insert or replace)."""
        plan_dict = {
            "template_sig": plan.template_sig,
            "tenant_id": plan.tenant_id,
            "strategies_config": plan.strategies_config,
            "segment_token_counts": plan.segment_token_counts,
            "breakpoint_positions": plan.breakpoint_positions,
            "depends_on_doc_versions": plan.depends_on_doc_versions,
            "created_at": plan.created_at,
        }
        self._store.set_l2_plan(  # type: ignore[attr-defined]
            template_sig=template_sig,
            tenant_id=tenant_id,
            plan_json=json.dumps(plan_dict),
            depends_on_doc_versions=json.dumps(plan.depends_on_doc_versions),
        )

    def invalidate_by_doc_version(self, doc_hash: str, new_version_hash: str) -> int:
        """
        CR-9: tombstone all plans that depend on doc_hash.
        new_version_hash is logged but not stored — the tombstone is the deletion.
        Returns the count of plans deleted.
        """
        return self._store.invalidate_l2_by_doc(doc_hash)  # type: ignore[attr-defined]
