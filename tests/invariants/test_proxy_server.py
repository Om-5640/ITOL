"""
§14.3 Invariant tests for itol/proxy/server.py.

Invariants:
  1. /healthz always returns 200.
  2. GET /dashboard serves the HTML without errors.
  3. CR-16: exception in pipeline → raw upstream dispatch, NOT 500.
  4. SSE /api/dashboard/stream emits at least one event per connection.
  5. X-ITOL-Upstream header routes the request to the correct upstream URL.
  6. Custom response headers present on every optimized response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MOCK_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello from mock upstream!"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_MOCK_ANTHROPIC_RESPONSE = {
    "id": "msg-test",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from mock Anthropic!"}],
    "model": "claude-opus-4-8-20251101",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


async def _mock_dispatch(url: str, headers: dict, body: dict) -> tuple[dict, int]:
    """Mock upstream that records calls and returns a fixed response."""
    if "anthropic" in url:
        return _MOCK_ANTHROPIC_RESPONSE, 200
    return _MOCK_RESPONSE, 200


async def _mock_dispatch_with_recorder(
    calls: list,
    url: str,
    headers: dict,
    body: dict,
) -> tuple[dict, int]:
    calls.append({"url": url, "body": body})
    return _MOCK_RESPONSE, 200


async def _passthrough_pipeline(icr: Any, state: Any) -> dict:
    """Minimal no-op pipeline that returns the original body unchanged."""
    return {
        "optimized_body": icr.raw,
        "tokens_saved": 0,
        "qps": 1.0,
        "cache_result": "miss",
        "strategies_applied": [],
        "rollback_stage": None,
    }


async def _exploding_pipeline(icr: Any, state: Any) -> dict:
    """Pipeline that always raises — used for CR-16 testing."""
    raise RuntimeError("Simulated pipeline failure for CR-16 test")


def _make_app(tmp_path, dispatch_fn=None, pipeline_fn=None):
    from itol.proxy.server import create_app
    return create_app(
        data_dir=str(tmp_path),
        dispatch_fn=dispatch_fn or _mock_dispatch,
        pipeline_fn=pipeline_fn or _passthrough_pipeline,
    )


_OPENAI_BODY = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ],
}

_ANTHROPIC_BODY = {
    "model": "claude-opus-4-8-20251101",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
}


# ===========================================================================
# Invariant 1: /healthz always 200
# ===========================================================================

class TestHealthz:

    @pytest.mark.asyncio
    async def test_healthz_returns_200(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/healthz")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_healthz_body_has_status_ok(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/healthz")
        data = resp.json()
        assert data.get("status") == "ok"

    @pytest.mark.asyncio
    async def test_healthz_body_has_version(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/healthz")
        data = resp.json()
        assert "version" in data


# ===========================================================================
# Invariant 2: /dashboard serves HTML
# ===========================================================================

class TestDashboard:

    @pytest.mark.asyncio
    async def test_dashboard_returns_200(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/dashboard")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_dashboard_content_type_html(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/dashboard")
        assert "text/html" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_dashboard_contains_chart_js(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/dashboard")
        assert "Chart.js" in resp.text or "chart.umd" in resp.text or "chartjs" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_dashboard_contains_sse_eventsource(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/dashboard")
        assert "EventSource" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_contains_animate_value(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/dashboard")
        assert "animateValue" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_html_file_exists(self):
        html_path = Path(__file__).resolve().parent.parent.parent / "itol" / "proxy" / "static" / "dashboard.html"
        assert html_path.exists(), f"dashboard.html not found at {html_path}"
        assert html_path.stat().st_size > 1000, "dashboard.html appears too small"


# ===========================================================================
# Invariant 3: CR-16 — pipeline exception → bypass, not 500
# ===========================================================================

class TestCR16PipelineBypass:

    @pytest.mark.asyncio
    async def test_pipeline_exception_does_not_return_500(self, tmp_path):
        """
        CR-16: an exception in the optimization pipeline must never return 500.
        The request must be forwarded to upstream unchanged (bypass mode).
        """
        app = _make_app(tmp_path, pipeline_fn=_exploding_pipeline)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock-upstream/v1/chat/completions"},
            )
        assert resp.status_code != 500, (
            "CR-16: pipeline exception must not result in 500 — got "
            f"{resp.status_code}: {resp.text[:200]}"
        )

    @pytest.mark.asyncio
    async def test_pipeline_exception_forwards_to_upstream(self, tmp_path):
        """
        CR-16: when pipeline raises, the original body must be forwarded
        to upstream (bypass), and the upstream response returned to the client.
        """
        calls: list = []

        async def recording_dispatch(url, headers, body):
            await _mock_dispatch_with_recorder(calls, url, headers, body)
            return _MOCK_RESPONSE, 200

        app = _make_app(tmp_path, dispatch_fn=recording_dispatch, pipeline_fn=_exploding_pipeline)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock-upstream/v1/chat/completions"},
            )

        assert resp.status_code == 200, f"Expected upstream response 200, got {resp.status_code}"
        assert len(calls) == 1, "Upstream must be called exactly once in bypass mode"
        # Bypass mode: forwarded body must be the original, unoptimized body
        forwarded = calls[0]["body"]
        assert forwarded.get("model") == _OPENAI_BODY["model"], (
            "Bypass must forward the original request body"
        )

    @pytest.mark.asyncio
    async def test_pipeline_exception_response_has_itol_headers(self, tmp_path):
        """Headers must be present even in bypass mode (tokens_saved=0)."""
        app = _make_app(tmp_path, pipeline_fn=_exploding_pipeline)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock-upstream/v1/chat/completions"},
            )
        assert "x-itol-saved-tokens" in resp.headers or "X-ITOL-Saved-Tokens" in resp.headers


# ===========================================================================
# Invariant 4: SSE streams at least one event
# ===========================================================================

class TestSSEStream:

    @pytest.mark.asyncio
    async def test_sse_broker_publishes_to_subscriber(self, tmp_path):
        """
        The in-process SSEBroker must deliver published events to all
        subscribed queues — the foundation of the SSE fan-out.
        """
        from itol.proxy.server import SSEBroker
        broker = SSEBroker()
        q = broker._subscribe()

        await broker.publish({"type": "test", "tokens_saved": 42})

        payload = q.get_nowait()
        data = json.loads(payload)
        assert data["type"] == "test"
        assert data["tokens_saved"] == 42
        broker._unsubscribe(q)

    @pytest.mark.asyncio
    async def test_sse_broker_fan_out_multiple_subscribers(self, tmp_path):
        """SSEBroker must fan out to ALL connected subscribers."""
        from itol.proxy.server import SSEBroker
        broker = SSEBroker()
        queues = [broker._subscribe() for _ in range(3)]

        await broker.publish({"event": "broadcast", "n": 7})

        for q in queues:
            payload = q.get_nowait()
            data = json.loads(payload)
            assert data["event"] == "broadcast"
            broker._unsubscribe(q)

    @pytest.mark.asyncio
    async def test_sse_initial_snapshot_is_serializable(self, tmp_path):
        """
        The data the SSE generator sends on connect (the stats snapshot) must
        be JSON-serializable — i.e. get_stats() must return a dict that can be
        encoded to JSON.  This is the direct source of the SSE initial event.

        We test the serialization directly rather than via HTTP streaming because
        httpx.AsyncClient+ASGITransport buffers the full response body and cannot
        read from an infinite SSE generator mid-stream.
        """
        from itol.proxy.dashboard import get_stats
        from itol.cache.store import Store
        store = Store(str(tmp_path))
        stats = get_stats(store)
        store.close()

        assert isinstance(stats, dict), "get_stats() must return a dict"
        try:
            encoded = json.dumps(stats)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"get_stats() result is not JSON-serializable: {exc}")

        # The encoded payload is what gets sent as "data: <payload>\n\n"
        assert len(encoded) > 2, "Stats payload must be non-empty"

    @pytest.mark.asyncio
    async def test_sse_endpoint_registered_with_event_stream_mediatype(self, tmp_path):
        """
        The SSE endpoint must be registered on the FastAPI app and return
        text/event-stream content-type.  Verified by inspecting the app's routes
        rather than consuming the infinite stream.
        """
        app = _make_app(tmp_path)

        # Verify the route is registered
        routes = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/api/dashboard/stream" in routes, (
            "/api/dashboard/stream route must be registered on the app"
        )

        # Verify the SSE broker is wired into app state
        assert hasattr(app.state.itol, "sse_broker"), (
            "app.state.itol must have an sse_broker attribute"
        )
        from itol.proxy.server import SSEBroker
        assert isinstance(app.state.itol.sse_broker, SSEBroker)

    @pytest.mark.asyncio
    async def test_post_request_publishes_sse_event(self, tmp_path):
        """
        After a /v1/chat/completions call, the SSE broker must have published
        at least one event (verified by subscribing directly to the broker).
        """
        from itol.proxy.server import create_app

        app = create_app(
            data_dir=str(tmp_path),
            dispatch_fn=_mock_dispatch,
            pipeline_fn=_passthrough_pipeline,
        )

        # Subscribe to the broker directly (bypasses HTTP streaming)
        broker = app.state.itol.sse_broker
        q = broker._subscribe()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock/v1/chat/completions"},
            )

        broker._unsubscribe(q)

        assert not q.empty(), (
            "SSE broker must publish at least one event after a request is processed"
        )
        payload = q.get_nowait()
        data = json.loads(payload)
        assert "request_id" in data, "SSE event must include request_id"


# ===========================================================================
# Invariant 5: X-ITOL-Upstream routes correctly
# ===========================================================================

class TestUpstreamRouting:

    @pytest.mark.asyncio
    async def test_x_itol_upstream_header_used_as_url(self, tmp_path):
        """
        X-ITOL-Upstream header must be used as the upstream URL instead of
        the default OpenAI endpoint.
        """
        calls: list = []

        async def routing_dispatch(url, headers, body):
            calls.append(url)
            return _MOCK_RESPONSE, 200

        app = _make_app(tmp_path, dispatch_fn=routing_dispatch)
        custom_url = "http://my-custom-llm.example.com/v1/chat/completions"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": custom_url},
            )

        assert len(calls) == 1, "Dispatch must be called exactly once"
        assert calls[0] == custom_url, (
            f"Expected upstream URL {custom_url!r}, got {calls[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_without_header_defaults_to_openai(self, tmp_path):
        """Without X-ITOL-Upstream, default must be the OpenAI endpoint."""
        calls: list = []

        async def routing_dispatch(url, headers, body):
            calls.append(url)
            return _MOCK_RESPONSE, 200

        app = _make_app(tmp_path, dispatch_fn=routing_dispatch)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/v1/chat/completions", json=_OPENAI_BODY)

        assert calls, "Dispatch must be called"
        assert "openai.com" in calls[0], (
            f"Default upstream must be OpenAI, got {calls[0]!r}"
        )

    @pytest.mark.asyncio
    async def test_anthropic_endpoint_uses_anthropic_default(self, tmp_path):
        """POST /v1/messages without X-ITOL-Upstream defaults to Anthropic."""
        calls: list = []

        async def routing_dispatch(url, headers, body):
            calls.append(url)
            return _MOCK_ANTHROPIC_RESPONSE, 200

        app = _make_app(tmp_path, dispatch_fn=routing_dispatch)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/v1/messages", json=_ANTHROPIC_BODY)

        assert calls, "Dispatch must be called"
        assert "anthropic.com" in calls[0], (
            f"Default Anthropic upstream expected, got {calls[0]!r}"
        )


# ===========================================================================
# Invariant 6: custom response headers on every response
# ===========================================================================

class TestResponseHeaders:

    @pytest.mark.asyncio
    async def test_x_itol_saved_tokens_header_present(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock/v1/chat/completions"},
            )
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-itol-saved-tokens" in headers_lower, (
            "X-ITOL-Saved-Tokens header must be present on every response"
        )

    @pytest.mark.asyncio
    async def test_x_itol_cache_header_present(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock/v1/chat/completions"},
            )
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-itol-cache" in headers_lower, (
            "X-ITOL-Cache header must be present on every response"
        )

    @pytest.mark.asyncio
    async def test_x_itol_qps_header_present(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock/v1/chat/completions"},
            )
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-itol-qps" in headers_lower, (
            "X-ITOL-QPS header must be present on every response"
        )

    @pytest.mark.asyncio
    async def test_x_itol_cache_value_is_valid_level(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json=_OPENAI_BODY,
                headers={"X-ITOL-Upstream": "http://mock/v1/chat/completions"},
            )
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        cache_val = headers_lower.get("x-itol-cache", "")
        assert cache_val in ("l0", "l1", "l2", "miss"), (
            f"X-ITOL-Cache must be l0/l1/l2/miss, got {cache_val!r}"
        )


# ===========================================================================
# Metrics endpoint
# ===========================================================================

class TestMetrics:

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_prometheus_format(self, tmp_path):
        app = _make_app(tmp_path)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/metrics")
        assert "itol_requests_total" in resp.text
        assert "# HELP" in resp.text
        assert "# TYPE" in resp.text
