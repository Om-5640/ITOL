"""
Prometheus metrics registry — §9.2.

Registers all ITOL counters and histograms.  If prometheus_client is not
installed, all calls are no-ops and get_metrics_text() returns an empty string
(so the /metrics endpoint degrades gracefully in environments without the dep).

Metrics
-------
itol_requests_total          counter  labels: tenant, provider, model, class, cache_result
itol_tokens_saved_total      counter  labels: tenant, strategy
itol_cost_saved_usd_total    counter  labels: tenant
itol_qps_histogram           histogram buckets: 0.80 0.85 0.90 0.95 0.98 0.99 1.0
itol_latency_ms_histogram    histogram buckets: 10 25 50 100 200 500
itol_rollback_total          counter  labels: tenant, strategy_disabled
itol_cache_hits_total        counter  labels: tier (l0/l1/l2)
"""

from __future__ import annotations

try:
    import prometheus_client as _prom
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False


# ---------------------------------------------------------------------------
# Histogram bucket specs (§9.2)
# ---------------------------------------------------------------------------

_QPS_BUCKETS     = (0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 1.0)
_LATENCY_BUCKETS = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """
    Thin wrapper around a Prometheus CollectorRegistry.

    Use `get_metrics_text()` to produce the /metrics payload.
    All observation methods are safe no-ops when prometheus_client is absent.
    """

    def __init__(self) -> None:
        if not _HAS_PROM:
            self._registry = None
            return

        self._registry = CollectorRegistry()

        self.requests_total = Counter(
            "itol_requests_total",
            "Total requests processed by ITOL",
            ["tenant", "provider", "model", "class", "cache_result"],
            registry=self._registry,
        )
        self.tokens_saved_total = Counter(
            "itol_tokens_saved_total",
            "Total tokens saved by optimisation strategies",
            ["tenant", "strategy"],
            registry=self._registry,
        )
        self.cost_saved_usd_total = Counter(
            "itol_cost_saved_usd_total",
            "Total estimated cost saved (USD, net of shadow eval cost — CR-13)",
            ["tenant"],
            registry=self._registry,
        )
        self.qps_histogram = Histogram(
            "itol_qps_histogram",
            "Distribution of Quality Preservation Scores",
            buckets=_QPS_BUCKETS,
            registry=self._registry,
        )
        self.latency_ms_histogram = Histogram(
            "itol_latency_ms_histogram",
            "ITOL processing latency (ms)",
            buckets=_LATENCY_BUCKETS,
            registry=self._registry,
        )
        self.rollback_total = Counter(
            "itol_rollback_total",
            "Total strategy rollbacks triggered by QPS gate",
            ["tenant", "strategy_disabled"],
            registry=self._registry,
        )
        self.cache_hits_total = Counter(
            "itol_cache_hits_total",
            "Total cache hits by tier",
            ["tier"],
            registry=self._registry,
        )

    # ------------------------------------------------------------------
    # Observation helpers  (all no-op if prometheus_client missing)
    # ------------------------------------------------------------------

    def observe_request(
        self,
        tenant: str,
        provider: str,
        model: str,
        request_class: str,
        cache_result: str,
    ) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.requests_total.labels(
            tenant=tenant, provider=provider, model=model,
            **{"class": request_class}, cache_result=cache_result,
        ).inc()

    def observe_tokens_saved(self, tenant: str, strategy: str, count: int) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.tokens_saved_total.labels(tenant=tenant, strategy=strategy).inc(count)

    def observe_cost_saved(self, tenant: str, usd: float) -> None:
        if not _HAS_PROM or self._registry is None or usd <= 0:
            return
        self.cost_saved_usd_total.labels(tenant=tenant).inc(usd)

    def observe_qps(self, qps: float) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.qps_histogram.observe(qps)

    def observe_latency(self, latency_ms: float) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.latency_ms_histogram.observe(latency_ms)

    def observe_rollback(self, tenant: str, strategy_disabled: str) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.rollback_total.labels(tenant=tenant, strategy_disabled=strategy_disabled).inc()

    def observe_cache_hit(self, tier: str) -> None:
        if not _HAS_PROM or self._registry is None:
            return
        self.cache_hits_total.labels(tier=tier).inc()

    def get_metrics_text(self) -> str:
        """Return Prometheus text-format metrics for the /metrics endpoint."""
        if not _HAS_PROM or self._registry is None:
            return ""
        return generate_latest(self._registry).decode("utf-8")

    @property
    def content_type(self) -> str:
        if not _HAS_PROM:
            return "text/plain"
        return CONTENT_TYPE_LATEST


# Module-level singleton
_default_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """Return (or create) the module-level default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetricsRegistry()
    return _default_registry
