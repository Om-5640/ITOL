"""
Request recorder — §9.1.

Writes every request outcome to:
  1. SQLite via Store.record_request()
  2. JSON-lines file at <data_dir>/telemetry.jsonl

CR-13
-----
est_cost_saved_usd = gross_saved_usd - shadow_cost_usd
shadow_cost_usd is stored in the requests table separately.
The net value is what appears in est_cost_saved_usd — never the gross alone.

CR-17
-----
tokens_out is always derived from provider_usage (the raw API usage dict),
never from local estimates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from itol.cache.store import Store


class Recorder:
    """
    Writes request telemetry to SQLite and to a rolling JSON-lines file.

    Parameters
    ----------
    store    : Store instance (SQLite backend)
    data_dir : directory for the telemetry.jsonl file
    """

    def __init__(self, store: Store, data_dir: str | Path) -> None:
        self._store = store
        self._jsonl_path = Path(data_dir) / "telemetry.jsonl"

    def record(
        self,
        *,
        request_id: str,
        tenant_id: str,
        provider: str | None = None,
        model: str | None = None,
        request_class: str | None = None,
        classifier_conf: float | None = None,
        template_sig: str | None = None,
        tokens_in_original: int = 0,
        tokens_in_optimized: int = 0,
        tokens_saved: int = 0,
        gross_cost_saved_usd: float = 0.0,
        shadow_cost_usd: float = 0.0,        # CR-13: stored separately; subtracted below
        provider_usage: dict[str, Any] | None = None,  # CR-17: authoritative token source
        qps: float | None = None,
        rollback_stage: str | None = None,
        cache_result: dict[str, Any] | None = None,
        latency_ms: dict[str, float] | None = None,
        strategies_applied: list[str] | None = None,
        strategy_savings: dict[str, Any] | None = None,
        shadow_sampled: bool = False,
        shadow_parity: float | None = None,
        error: str | None = None,
    ) -> None:
        # CR-13: net savings = gross − shadow eval cost
        est_cost_saved_usd = gross_cost_saved_usd - shadow_cost_usd

        # Write to SQLite
        self._store.record_request(
            request_id=request_id,
            tenant_id=tenant_id,
            provider=provider,
            model=model,
            request_class=request_class,
            classifier_conf=classifier_conf,
            template_sig=template_sig,
            tokens_in_original=tokens_in_original,
            tokens_in_optimized=tokens_in_optimized,
            tokens_saved=tokens_saved,
            est_cost_saved_usd=est_cost_saved_usd,
            shadow_cost_usd=shadow_cost_usd,
            provider_usage=provider_usage,
            qps=qps,
            rollback_stage=rollback_stage,
            cache_result=cache_result,
            latency_ms=latency_ms,
            strategies_applied=strategies_applied,
            strategy_savings=strategy_savings,
            shadow_sampled=shadow_sampled,
            shadow_parity=shadow_parity,
            error=error,
        )

        # Write to JSON-lines file
        record_dict: dict[str, Any] = {
            "request_id":          request_id,
            "tenant_id":           tenant_id,
            "ts":                  time.time(),
            "provider":            provider,
            "model":               model,
            "request_class":       request_class,
            "classifier_conf":     classifier_conf,
            "tokens_in_original":  tokens_in_original,
            "tokens_in_optimized": tokens_in_optimized,
            "tokens_saved":        tokens_saved,
            "est_cost_saved_usd":  est_cost_saved_usd,
            "shadow_cost_usd":     shadow_cost_usd,
            "qps":                 qps,
            "rollback_stage":      rollback_stage,
            "cache_result":        cache_result,
            "latency_ms":          latency_ms,
            "strategies_applied":  strategies_applied,
            "strategy_savings":    strategy_savings,
            "shadow_sampled":      shadow_sampled,
            "shadow_parity":       shadow_parity,
            "error":               error,
        }
        try:
            with self._jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record_dict, default=str) + "\n")
        except OSError:
            pass    # best-effort; never raise from telemetry path
