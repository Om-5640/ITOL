"""
OpenTelemetry distributed tracing — §8.3.

No-ops cleanly when `opentelemetry` is not installed (same three-tier-optional
pattern as the embedder/reranker).  Zero overhead beyond a function call when
OTel is absent.

When OTel is present:
  - Console exporter by default (no external collector — hard-constraint-1)
  - OTLP exporter if OTEL_EXPORTER_OTLP_ENDPOINT env var is set

Usage
-----
    from itol.telemetry.tracing import setup_tracing, span

    setup_tracing("itol")            # once at startup

    with span("itol.request", tenant_id="t1") as s:
        s.set_attribute("tokens_saved", 42)
        with span("analyze"):
            ...
        with span("optimize.S1"):
            ...
        with span("quality_gate"):
            ...
        with span("dispatch"):
            ...
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter as _ConsoleExporter,
        SimpleSpanProcessor as _SimpleProcessor,
    )
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


# ---------------------------------------------------------------------------
# No-op span  (zero overhead when OTel absent or not set up)
# ---------------------------------------------------------------------------

class _NoopSpan:
    """Returned by span() when OpenTelemetry is absent or not initialised."""
    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def add_event(self, name: str) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Global tracer  (None until setup_tracing() is called)
# ---------------------------------------------------------------------------

_tracer: Any = None   # opentelemetry.Tracer | None


def setup_tracing(
    service_name: str = "itol",
    exporter: str = "console",
) -> None:
    """
    Initialise the global ITOL tracer.

    Parameters
    ----------
    service_name : str   default "itol"
    exporter     : str   "console" (default) or "otlp"
                         OTLP is also activated automatically when
                         OTEL_EXPORTER_OTLP_ENDPOINT env var is set.

    Safe to call when opentelemetry is not installed — becomes a no-op.
    """
    global _tracer
    if not _HAS_OTEL:
        return

    provider = _TracerProvider()

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    use_otlp = exporter == "otlp" or bool(otlp_endpoint)

    if use_otlp:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            ep = otlp_endpoint or "http://localhost:4317"
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=ep)))
        except ImportError:
            # OTLP package absent → fall back to console
            provider.add_span_processor(_SimpleProcessor(_ConsoleExporter()))
    else:
        provider.add_span_processor(_SimpleProcessor(_ConsoleExporter()))

    _otel_trace.set_tracer_provider(provider)
    _tracer = _otel_trace.get_tracer(service_name)


@contextmanager
def span(name: str, **attributes: Any) -> Generator[Any, None, None]:
    """
    Context manager that creates a named OTel trace span.

    Yields a _NoopSpan (zero overhead) when:
    - opentelemetry is not installed
    - setup_tracing() was never called

    Usage::

        with span("itol.request", tenant_id="default") as s:
            s.set_attribute("tokens_saved", 42)
            with span("analyze"):
                ...
    """
    if not _HAS_OTEL or _tracer is None:
        s = _NoopSpan()
        yield s
        return

    with _tracer.start_as_current_span(name) as otel_span:
        for k, v in attributes.items():
            # OTel only accepts bool, int, float, str — coerce everything else
            if not isinstance(v, (bool, int, float, str)):
                v = str(v)
            otel_span.set_attribute(k, v)
        yield otel_span
