"""
ITOL proxy server — §14 (Step 14).

Endpoints
---------
POST /v1/chat/completions   OpenAI-compatible; routes through pipeline
POST /v1/messages           Anthropic dialect
GET  /healthz               Liveness
GET  /metrics               Prometheus text
GET  /dashboard             Dashboard HTML
GET  /api/dashboard/stats   JSON stats snapshot
GET  /api/dashboard/stream  Server-Sent Events live feed

Response headers on every optimized request
-------------------------------------------
X-ITOL-Saved-Tokens   integer
X-ITOL-Cache          l0 | l1 | l2 | miss
X-ITOL-QPS            float

CR-16
-----
Any unhandled exception in the pipeline → bypass mode: forward original body
to upstream unchanged.  Never returns 500 for pipeline errors.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# SSE broker (in-process pub/sub — no Redis in default mode)
# ---------------------------------------------------------------------------

class SSEBroker:
    """Fan-out SSE events to all connected clients via asyncio.Queue."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []

    def _subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return q

    def _unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    async def publish(self, event: dict) -> None:
        payload = json.dumps(event)
        for q in list(self._queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def stream(self, q: asyncio.Queue):
        """Async generator that yields SSE-formatted strings."""
        try:
            while True:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {payload}\n\n"
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

@dataclass
class _AppState:
    config: Any
    store: Any
    recorder: Any | None
    sse_broker: SSEBroker
    data_dir: Path
    dispatch_fn: Callable | None = None
    pipeline_fn: Callable | None = None


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _icr_from_openai(body: dict) -> Any:
    from itol.icr import ICR, Message, ContentBlock, ContentType

    messages = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            blocks = [ContentBlock(type=ContentType.TEXT, text=content)]
        elif isinstance(content, list):
            blocks = [
                ContentBlock(type=ContentType.TEXT, text=blk.get("text", ""))
                for blk in content
                if blk.get("type") == "text"
            ]
        else:
            blocks = []
        messages.append(Message(role=role, content=blocks))

    return ICR.create(
        provider="openai",
        model=body.get("model", "gpt-4o"),
        messages=messages,
        raw=body,
    )


def _icr_from_anthropic(body: dict) -> Any:
    from itol.icr import ICR, Message, ContentBlock, ContentType

    messages = []
    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            blocks = [ContentBlock(type=ContentType.TEXT, text=content)]
        elif isinstance(content, list):
            blocks = [
                ContentBlock(type=ContentType.TEXT, text=blk.get("text", ""))
                for blk in content
                if blk.get("type") == "text"
            ]
        else:
            blocks = []
        messages.append(Message(role=role, content=blocks))

    system_txt = body.get("system", "")
    if system_txt:
        from itol.icr import ContentType as CT
        messages.insert(0, Message(
            role="system",
            content=[ContentBlock(type=CT.TEXT, text=system_txt)],
        ))

    return ICR.create(
        provider="anthropic",
        model=body.get("model", "claude-opus-4-8"),
        messages=messages,
        raw=body,
    )


async def _run_real_pipeline(icr: Any, state: _AppState) -> dict:
    """
    Lightweight pipeline: classify → manifest → strategies → QPS gate.
    Returns dict with keys: optimized_body, tokens_saved, qps, cache_result,
    strategies_applied, rollback_stage.
    """
    from itol.segmenter import segment
    from itol.signals import extract_signals, estimate_token_count
    from itol.analysis.classifier import classify
    from itol.analysis.manifest import extract_manifest
    from itol.strategies.base import OptimizationContext
    from itol.routing.matrix import MATRIX
    from itol.quality.qps import score_and_rollback
    from itol.icr import ConstraintManifest, SegmentSignals
    from itol.telemetry.tracing import span as _span

    config = state.config
    store = state.store

    # Segment
    segments = segment(icr)

    with _span("analyze",
               tenant_id=icr.tenant_id,
               provider=icr.provider,
               model=icr.model) as analyze_span:
        # Signals
        signals = extract_signals(segments)

        # Classify
        cls_result = classify(icr, config)
        request_class = cls_result.request_class
        analyze_span.set_attribute("request_class", request_class)

        # Manifest
        manifest = extract_manifest(icr, config)

    # Build optimization context
    cls_cfg = config.class_configs.get(request_class)
    matrix_row = MATRIX.get(request_class)

    ctx = OptimizationContext(
        request_class=request_class,
        matrix_row=matrix_row,
        manifest=manifest,
        signals=signals,
        config=config,
    )

    tokens_before = sum(s.token_count for s in segments if s.token_count)

    # Run strategies in pipeline order
    from itol.strategies.s1_dedupe import S1DedupeStrategy
    from itol.strategies.s6_minify import S6MinifyStrategy

    strategies = [S1DedupeStrategy(), S6MinifyStrategy()]
    if cls_cfg and cls_cfg.s3_enabled:
        from itol.strategies.s3_window import S3WindowStrategy
        strategies.append(S3WindowStrategy())
    if cls_cfg and cls_cfg.s4_enabled:
        from itol.strategies.s4_racr import S4RACRStrategy
        strategies.append(S4RACRStrategy())
    if cls_cfg and cls_cfg.s7_enabled:
        from itol.strategies.s7_lossy import S7LossyStrategy
        strategies.append(S7LossyStrategy())

    current_segments = list(segments)
    all_reports = []
    for strat in strategies:
        try:
            if strat.applies(icr, current_segments, ctx):
                with _span(f"optimize.{strat.strategy_id}"):
                    new_segs, report = strat.apply(icr, current_segments, ctx)
                current_segments = new_segs
                all_reports.append(report)
        except Exception:
            pass

    # QPS gate
    quality_cfg = config.quality
    with _span("quality_gate") as qg_span:
        score_result = score_and_rollback(
            icr=icr,
            strategy_reports=all_reports,
            manifest=manifest,
            quality_cfg=quality_cfg,
            optimised_segments=current_segments,
        )
        qg_span.set_attribute("qps", score_result.qps_result.qps or 0.0)
        qg_span.set_attribute("use_raw", score_result.use_raw)

    tokens_after = sum(s.token_count for s in (score_result.segments or current_segments) if s.token_count)
    tokens_saved = max(0, tokens_before - tokens_after)

    if score_result.use_raw:
        body_to_send = icr.raw
        strategies_applied = []
        rollback_stage = score_result.qps_result.rollback_stage_passed
    else:
        # Reconstruct body from optimized segments
        body_to_send = _reconstruct_openai_body(icr.raw, score_result.segments or current_segments)
        strategies_applied = [r.strategy_id for r in all_reports if r.activated]
        rollback_stage = score_result.qps_result.rollback_stage_passed

    return {
        "optimized_body": body_to_send,
        "tokens_saved": tokens_saved,
        "qps": score_result.qps_result.qps,
        "cache_result": "miss",
        "strategies_applied": strategies_applied,
        "rollback_stage": rollback_stage,
    }


def _reconstruct_openai_body(original: dict, segments: list) -> dict:
    """Rebuild the OpenAI messages array from optimized segments."""
    from itol.segmenter import segments_full_text
    full_text = segments_full_text(segments)

    body = dict(original)
    messages = list(original.get("messages", []))
    if messages:
        # Replace the last user message content with optimized text
        last = dict(messages[-1])
        last["content"] = full_text
        messages[-1] = last
        body["messages"] = messages
    return body


async def _dispatch_upstream(
    url: str,
    headers: dict,
    body: dict,
    dispatch_fn: Callable | None,
) -> tuple[dict, int]:
    """Forward the request to the upstream LLM API."""
    if dispatch_fn is not None:
        return await dispatch_fn(url=url, headers=headers, body=body)

    import httpx
    clean_headers = {
        k: v for k, v in headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
        and not k.lower().startswith("x-itol-")
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=clean_headers, json=body)
    return resp.json(), resp.status_code


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    data_dir: str | Path | None = None,
    config: Any | None = None,
    store: Any | None = None,
    recorder: Any | None = None,
    dispatch_fn: Callable | None = None,
    pipeline_fn: Callable | None = None,
) -> FastAPI:
    """
    Build the ITOL FastAPI application.

    Parameters
    ----------
    data_dir    : directory for SQLite + telemetry files
    config      : ITOLConfig (default: ITOLConfig())
    store       : Store instance (default: new Store(data_dir))
    recorder    : Recorder instance (default: new Recorder(store, data_dir))
    dispatch_fn : async (url, headers, body) -> (dict, int) override for upstream
    pipeline_fn : async (icr, state) -> pipeline_result override for testing
    """
    from itol.config import ITOLConfig
    from itol.cache.store import Store
    from itol.telemetry.recorder import Recorder

    if data_dir is None:
        data_dir = Path("data")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = ITOLConfig()
    if store is None:
        store = Store(str(data_dir))
    if recorder is None:
        recorder = Recorder(store, data_dir)

    broker = SSEBroker()

    state = _AppState(
        config=config,
        store=store,
        recorder=recorder,
        sse_broker=broker,
        data_dir=data_dir,
        dispatch_fn=dispatch_fn,
        pipeline_fn=pipeline_fn,
    )

    app = FastAPI(title="ITOL Proxy", version="0.1.0")
    app.state.itol = state

    # -----------------------------------------------------------------------
    # /healthz
    # -----------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "version": "0.1.0"}

    # -----------------------------------------------------------------------
    # /metrics  (Prometheus text format)
    # -----------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics(request: Request) -> Response:
        st: _AppState = request.app.state.itol
        try:
            conn = st.store._conn()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), COALESCE(SUM(tokens_saved),0) FROM requests")
            total_req, total_tokens = cur.fetchone()
        except Exception:
            total_req, total_tokens = 0, 0

        body = (
            "# HELP itol_requests_total Total requests processed\n"
            "# TYPE itol_requests_total counter\n"
            f"itol_requests_total {total_req}\n"
            "# HELP itol_tokens_saved_total Total tokens saved\n"
            "# TYPE itol_tokens_saved_total counter\n"
            f"itol_tokens_saved_total {total_tokens}\n"
            "# HELP itol_sse_clients Current SSE clients connected\n"
            "# TYPE itol_sse_clients gauge\n"
            f"itol_sse_clients {len(st.sse_broker._queues)}\n"
        )
        return Response(content=body, media_type="text/plain; version=0.0.4")

    # -----------------------------------------------------------------------
    # /dashboard  — serves the HTML file
    # -----------------------------------------------------------------------
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "dashboard.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)

    # -----------------------------------------------------------------------
    # /api/dashboard/stats
    # -----------------------------------------------------------------------
    @app.get("/api/dashboard/stats")
    async def dashboard_stats(
        request: Request,
        window: str = "24h",
        tenant_id: str = "default",
    ) -> dict:
        from itol.proxy.dashboard import get_stats
        st: _AppState = request.app.state.itol
        try:
            return get_stats(st.store, tenant_id=tenant_id, window=window)
        except Exception as exc:
            return {"error": str(exc), "total_requests": 0}

    # -----------------------------------------------------------------------
    # /api/dashboard/stream  — SSE
    # -----------------------------------------------------------------------
    @app.get("/api/dashboard/stream")
    async def dashboard_stream(request: Request) -> StreamingResponse:
        st: _AppState = request.app.state.itol
        q = st.sse_broker._subscribe()

        async def event_generator():
            # Race between queue events and disconnect check using two tasks.
            # This ensures the generator exits within ~50 ms of client disconnect.
            q_task: asyncio.Task | None = None
            try:
                from itol.proxy.dashboard import get_stats
                try:
                    snap = get_stats(st.store)
                    yield f"data: {json.dumps(snap)}\n\n"
                except Exception:
                    yield "data: {}\n\n"

                while True:
                    if q_task is None or q_task.done():
                        q_task = asyncio.ensure_future(q.get())
                    disc_task = asyncio.ensure_future(request.is_disconnected())
                    done, _ = await asyncio.wait(
                        {q_task, disc_task},
                        timeout=25.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Cancel the loser
                    if disc_task not in done:
                        disc_task.cancel()
                    if q_task not in done:
                        q_task.cancel()
                        q_task = None

                    if not done:
                        yield ": keepalive\n\n"
                        continue

                    if disc_task in done and disc_task.result():
                        return

                    if q_task in done:
                        try:
                            payload = q_task.result()
                            yield f"data: {payload}\n\n"
                        except Exception:
                            pass
                        q_task = None

            except (asyncio.CancelledError, GeneratorExit):
                pass
            finally:
                if q_task is not None and not q_task.done():
                    q_task.cancel()
                st.sse_broker._unsubscribe(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # -----------------------------------------------------------------------
    # POST /v1/chat/completions  (OpenAI)
    # -----------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _handle_request(request, dialect="openai")

    # -----------------------------------------------------------------------
    # POST /v1/messages  (Anthropic)
    # -----------------------------------------------------------------------
    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Response:
        return await _handle_request(request, dialect="anthropic")

    # -----------------------------------------------------------------------
    # Shared request handler
    # -----------------------------------------------------------------------
    async def _handle_request(request: Request, dialect: str) -> Response:
        from itol.telemetry.tracing import span as _span

        st: _AppState = request.app.state.itol
        request_id = str(uuid.uuid4())
        t_start = time.time()

        # Resolve upstream URL from header or default
        upstream_url = (
            request.headers.get("x-itol-upstream")
            or request.headers.get("X-ITOL-Upstream")
        )
        if not upstream_url:
            upstream_url = (
                "https://api.anthropic.com/v1/messages"
                if dialect == "anthropic"
                else "https://api.openai.com/v1/chat/completions"
            )

        try:
            body = await request.json()
        except Exception:
            body = {}

        # Forward headers (strip hop-by-hop + ITOL headers)
        forward_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in (
                "host", "content-length", "transfer-encoding", "connection"
            ) and not k.lower().startswith("x-itol-")
        }

        tokens_saved = 0
        qps_val = 1.0
        cache_level = "miss"
        strategies_applied: list[str] = []
        rollback_stage: str | None = None

        with _span("itol.request",
                   request_id=request_id,
                   dialect=dialect,
                   model=body.get("model", "")) as req_span:

            try:
                # Build ICR
                if dialect == "openai":
                    icr = _icr_from_openai(body)
                else:
                    icr = _icr_from_anthropic(body)

                req_span.set_attribute("tenant_id", "default")

                # Run pipeline (or injected override)
                _pipeline = st.pipeline_fn or _run_real_pipeline
                result = await _pipeline(icr, st)

                body_to_send = result.get("optimized_body", body)
                tokens_saved = result.get("tokens_saved", 0)
                qps_val = result.get("qps", 1.0)
                cache_level = result.get("cache_result", "miss")
                strategies_applied = result.get("strategies_applied", [])
                rollback_stage = result.get("rollback_stage")

                req_span.set_attribute("tokens_saved", tokens_saved)
                req_span.set_attribute("qps", qps_val)
                req_span.set_attribute("cache_result", cache_level)

            except Exception:
                # CR-16: bypass — forward original body unchanged
                body_to_send = body
                tokens_saved = 0
                qps_val = 1.0
                cache_level = "miss"

            # Dispatch to upstream
            try:
                with _span("dispatch", upstream_url=upstream_url):
                    upstream_body, status_code = await _dispatch_upstream(
                        url=upstream_url,
                        headers=forward_headers,
                        body=body_to_send,
                        dispatch_fn=st.dispatch_fn,
                    )
            except Exception as exc:
                return JSONResponse(
                    {"error": f"upstream dispatch failed: {exc}"},
                    status_code=502,
                )

        latency_ms = (time.time() - t_start) * 1000

        # Record telemetry
        if st.recorder is not None:
            try:
                st.recorder.record(
                    request_id=request_id,
                    tenant_id="default",
                    provider=dialect,
                    model=body.get("model"),
                    tokens_saved=tokens_saved,
                    qps=qps_val,
                    cache_result={"level": cache_level},
                    strategies_applied=strategies_applied,
                    rollback_stage=rollback_stage,
                    latency_ms={"total": latency_ms},
                )
            except Exception:
                pass

        # Publish SSE event
        try:
            await st.sse_broker.publish({
                "request_id": request_id,
                "tokens_saved": tokens_saved,
                "qps": qps_val,
                "cache": cache_level,
                "ts": time.time(),
            })
        except Exception:
            pass

        response = JSONResponse(content=upstream_body, status_code=status_code)
        response.headers["X-ITOL-Saved-Tokens"] = str(tokens_saved)
        response.headers["X-ITOL-Cache"] = cache_level
        response.headers["X-ITOL-QPS"] = f"{qps_val:.4f}"
        return response

    return app
