"""
§8.3 OTel tracing invariants.

test_tracing_noop_without_otel  — pipeline + span() operate identically
                                   when OTel is absent (mocked out).
test_tracing_spans_created       — with OTel installed + in-memory exporter,
                                   assert correct span names and attributes.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Noop test — simulate OTel not installed
# ---------------------------------------------------------------------------

def test_tracing_noop_without_otel() -> None:
    """span() must return a _NoopSpan and never raise when OTel is absent."""
    import itol.telemetry.tracing as tr

    # Force noop mode regardless of what's installed
    original_has_otel = tr._HAS_OTEL
    original_tracer   = tr._tracer
    try:
        tr._HAS_OTEL = False
        tr._tracer   = None

        results = []
        with tr.span("itol.request", tenant_id="t1") as root:
            root.set_attribute("tokens_saved", 99)
            root.add_event("pipeline_start")
            with tr.span("analyze") as child:
                child.set_attribute("request_class", "REASONING")
                results.append("analyze_ran")
            with tr.span("optimize.S1") as s:
                s.set_attribute("tokens_removed", 10)
                results.append("s1_ran")
            with tr.span("quality_gate") as qg:
                qg.set_attribute("qps", 0.99)
                results.append("qgate_ran")
            with tr.span("dispatch") as d:
                d.set_attribute("upstream", "openai")
                results.append("dispatch_ran")

        # All stages ran — span() must not swallow or skip user code
        assert results == ["analyze_ran", "s1_ran", "qgate_ran", "dispatch_ran"]

    finally:
        tr._HAS_OTEL = original_has_otel
        tr._tracer   = original_tracer


def test_tracing_noop_returns_noopspan() -> None:
    """span() yields _NoopSpan instance when OTel is off."""
    import itol.telemetry.tracing as tr
    from itol.telemetry.tracing import _NoopSpan

    original_has_otel = tr._HAS_OTEL
    original_tracer   = tr._tracer
    try:
        tr._HAS_OTEL = False
        tr._tracer   = None

        with tr.span("test.span") as s:
            assert isinstance(s, _NoopSpan)
    finally:
        tr._HAS_OTEL = original_has_otel
        tr._tracer   = original_tracer


def test_setup_tracing_noop_when_otel_absent() -> None:
    """setup_tracing() must not raise when OTel is absent."""
    import itol.telemetry.tracing as tr

    original_has_otel = tr._HAS_OTEL
    original_tracer   = tr._tracer
    try:
        tr._HAS_OTEL = False
        tr._tracer   = None
        tr.setup_tracing("itol", exporter="console")   # must not raise
        assert tr._tracer is None
    finally:
        tr._HAS_OTEL = original_has_otel
        tr._tracer   = original_tracer


# ---------------------------------------------------------------------------
# OTel spans test — requires opentelemetry-sdk
# ---------------------------------------------------------------------------

otel_sdk = pytest.importorskip(
    "opentelemetry.sdk.trace",
    reason="opentelemetry-sdk not installed — skipping OTel span tests",
)


def _make_in_memory_tracer():
    """Return (tracer, exporter) backed by InMemorySpanExporter."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def test_tracing_spans_created() -> None:
    """
    With OTel + in-memory exporter, assert spans are created with
    correct names and attributes.
    """
    import itol.telemetry.tracing as tr
    from opentelemetry import trace as otel_trace

    tracer, exporter = _make_in_memory_tracer()

    original_tracer   = tr._tracer
    original_has_otel = tr._HAS_OTEL
    try:
        tr._HAS_OTEL = True
        tr._tracer   = tracer

        with tr.span("itol.request", tenant_id="tenant1", request_class="REASONING") as root:
            root.set_attribute("tokens_saved", 150)
            root.set_attribute("qps", 0.987)
            root.set_attribute("cache_result", "l1")
            with tr.span("analyze"):
                pass
            with tr.span("cache_lookup"):
                pass
            with tr.span("optimize.S1"):
                pass
            with tr.span("optimize.S6"):
                pass
            with tr.span("quality_gate"):
                pass
            with tr.span("dispatch"):
                pass

    finally:
        tr._tracer   = original_tracer
        tr._HAS_OTEL = original_has_otel

    finished = exporter.get_finished_spans()
    names = {s.name for s in finished}

    # Required span names
    assert "itol.request" in names, f"Missing itol.request in {names}"
    assert "analyze"       in names, f"Missing analyze in {names}"
    assert "cache_lookup"  in names, f"Missing cache_lookup in {names}"
    assert "quality_gate"  in names, f"Missing quality_gate in {names}"
    assert "dispatch"      in names, f"Missing dispatch in {names}"

    # At least one optimize.* span
    opt_spans = [n for n in names if n.startswith("optimize.")]
    assert len(opt_spans) >= 1, f"No optimize.* spans found in {names}"

    # Root span attributes
    req_span = next(s for s in finished if s.name == "itol.request")
    attrs = req_span.attributes
    assert attrs.get("tenant_id")     == "tenant1"
    assert attrs.get("request_class") == "REASONING"
    assert attrs.get("tokens_saved")  == 150
    assert attrs.get("cache_result")  == "l1"


def test_tracing_child_spans_are_nested() -> None:
    """Child spans must have the root span as parent."""
    import itol.telemetry.tracing as tr

    tracer, exporter = _make_in_memory_tracer()
    original_tracer   = tr._tracer
    original_has_otel = tr._HAS_OTEL
    try:
        tr._HAS_OTEL = True
        tr._tracer   = tracer
        with tr.span("itol.request") as _root:
            with tr.span("analyze"):
                pass
    finally:
        tr._tracer   = original_tracer
        tr._HAS_OTEL = original_has_otel

    finished = exporter.get_finished_spans()
    root  = next(s for s in finished if s.name == "itol.request")
    child = next(s for s in finished if s.name == "analyze")
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id


def test_tracing_qps_attribute_as_string() -> None:
    """Non-str/int/float attributes must be coerced to str without raising."""
    import itol.telemetry.tracing as tr

    tracer, exporter = _make_in_memory_tracer()
    original_tracer   = tr._tracer
    original_has_otel = tr._HAS_OTEL
    try:
        tr._HAS_OTEL = True
        tr._tracer   = tracer
        with tr.span("test", some_list=[1, 2, 3]) as s:
            pass
    finally:
        tr._tracer   = original_tracer
        tr._HAS_OTEL = original_has_otel

    finished = exporter.get_finished_spans()
    sp = next(s for s in finished if s.name == "test")
    assert sp.attributes["some_list"] == "[1, 2, 3]"
