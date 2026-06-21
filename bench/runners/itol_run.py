"""
ITOL runner — sends prompts through the full ITOL optimization pipeline
before dispatching to the provider.

Pipeline per request:
  ICR.create → segment_icr → classify → extract_manifest → extract_signals
  → build OptimizationContext → run strategies (S1,S2,S3,S4,S5,S6)
  → QPS gate → dispatch optimized body → record savings

L0 cache is checked before running the pipeline (FAQ workload benefit).
Results are also written to the L0 cache for subsequent paraphrase hits.

All JSONL output uses the same schema as baseline.py, plus ITOL-specific fields:
tokens_saved, strategies_fired, cache_tier, qps, rollback, pipeline_ms.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench.config import ProviderConfig, BenchConfig
from bench.judge import judge
from bench.metrics import BenchResult, append_result, completed_ids, result_path
from bench.rate_limit import call_with_retry
from bench.runners.baseline import (
    _build_openai_body, _build_cohere_body,
    _dispatch_openai, _dispatch_cohere,
    _parse_openai_response, _parse_cohere_response,
    _mock_dispatch,
)
from bench.workloads import WorkloadSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ITOL pipeline (self-contained; uses correct APIs)
# ---------------------------------------------------------------------------

def _build_icr(sample: WorkloadSample, provider_name: str, model: str):
    """Build an ICR from a WorkloadSample."""
    from itol.icr import ICR, Message, ContentBlock, ContentType, ToolDef

    messages = []
    for m in sample.messages:
        role = m.get("role", "user")
        content = m.get("content")

        # Skip system messages — they go into ICR.system separately
        if role == "system":
            continue

        if isinstance(content, str):
            blocks = [ContentBlock(type=ContentType.TEXT, text=content)]
        elif isinstance(content, list):
            blocks = [
                ContentBlock(type=ContentType.TEXT, text=blk.get("text", ""))
                for blk in content if blk.get("type") == "text"
            ]
        else:
            blocks = []

        # Tool messages: handle as assistant with tool results
        if role == "tool":
            tool_content = m.get("content", "")
            blocks = [ContentBlock(type=ContentType.TOOL_RESULT,
                                   text=tool_content if isinstance(tool_content, str)
                                   else json.dumps(tool_content))]
            role = "user"

        messages.append(Message(role=role, content=blocks))

    # System prompt
    system_blocks = []
    for m in sample.messages:
        if m.get("role") == "system":
            system_blocks.append(ContentBlock(type=ContentType.TEXT, text=m.get("content", "")))
            break

    # Tool defs from agent workload
    tools = []
    if sample.workload == "agent":
        for m in sample.messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    tools.append(ToolDef(
                        name=fn.get("name", "unknown"),
                        description=f"Tool: {fn.get('name', '')}",
                        input_schema={},
                    ))
                break

    raw_body = _build_openai_body(sample, model, 0.0, 42)

    return ICR.create(
        provider=provider_name,
        model=model,
        messages=messages,
        system=system_blocks if system_blocks else None,
        tools=tools if tools else None,
        raw=raw_body,
        tenant_id="bench",
        conversation_id=sample.sample_id,
    )


import json


def _run_pipeline(icr, store, config) -> dict:
    """
    Run the ITOL optimization pipeline synchronously.
    Returns dict with: optimized_body, tokens_saved, qps, strategies_applied, rollback.
    """
    from itol.segmenter import segment_icr
    from itol.signals import extract_signals
    from itol.analysis.classifier import classify
    from itol.analysis.manifest import extract_manifest
    from itol.strategies.base import OptimizationContext
    from itol.routing.matrix import MATRIX
    from itol.quality.qps import score_and_rollback
    from itol.strategies.s1_dedupe import S1DedupeStrategy
    from itol.strategies.s6_hygiene import S6HygieneStrategy
    from itol.icr import ConstraintManifest, SegmentSignals

    segments = segment_icr(icr)
    signals  = extract_signals(icr, segments)
    cls_result = classify(icr)
    request_class = cls_result.primary
    manifest = extract_manifest(icr)

    matrix_row = MATRIX.get(request_class) or MATRIX.get("CHAT_OPEN")
    cls_cfg = config.class_configs.get(request_class)

    ctx = OptimizationContext(
        request_class=request_class,
        matrix_row=matrix_row,
        manifest=manifest or ConstraintManifest(),
        signals=signals,
        config=config,
    )

    def _est_tokens(seg) -> int:
        if seg.token_count:
            return seg.token_count
        return max(1, len((seg.text or "").split()) * 4 // 3)

    tokens_before = sum(_est_tokens(s) for s in segments)

    # Build strategy list (same order as proxy pipeline)
    strategies = [S1DedupeStrategy(), S6HygieneStrategy()]
    if cls_cfg and getattr(cls_cfg, "s3_enabled", True):
        from itol.strategies.s3_window import S3WindowStrategy
        strategies.append(S3WindowStrategy())
    if cls_cfg and getattr(cls_cfg, "s4_enabled", False):
        from itol.strategies.s4_racr import S4RACRStrategy
        strategies.append(S4RACRStrategy())
    if cls_cfg and getattr(cls_cfg, "s5_enabled", True):
        from itol.strategies.s5_distill import S5DistillStrategy
        strategies.append(S5DistillStrategy(store=store))

    current_segments = list(segments)
    all_reports = []
    for strat in strategies:
        try:
            if strat.applies(icr, current_segments, ctx):
                new_segs, report = strat.apply(icr, current_segments, ctx)
                current_segments = new_segs
                all_reports.append(report)
        except Exception as exc:
            logger.debug("Strategy %s failed: %s", type(strat).__name__, exc)

    score_result = score_and_rollback(
        icr=icr,
        strategy_reports=all_reports,
        manifest=manifest or ConstraintManifest(),
        quality_cfg=config.quality,
        optimised_segments=current_segments,
    )

    tokens_after = sum(
        _est_tokens(s)
        for s in (score_result.segments or current_segments)
    )
    tokens_saved = max(0, tokens_before - tokens_after)

    if score_result.use_raw:
        body_to_send = icr.raw
        strategies_applied = []
    else:
        # Rebuild the request body from optimized segments
        from itol.segmenter import segments_full_text
        opt_segs = score_result.segments or current_segments
        opt_text = segments_full_text(opt_segs)

        body_to_send = dict(icr.raw)
        messages = list(icr.raw.get("messages", []))
        if messages:
            last = dict(messages[-1])
            last["content"] = opt_text
            messages[-1] = last
            body_to_send["messages"] = messages

        strategies_applied = [r.strategy_id for r in all_reports if r.activated]

    return {
        "optimized_body": body_to_send,
        "tokens_saved": tokens_saved,
        "qps": score_result.qps_result.qps,
        "strategies_applied": strategies_applied,
        "rollback": score_result.use_raw,
    }


# ---------------------------------------------------------------------------
# L0 cache helpers for benchmark
# ---------------------------------------------------------------------------

def _l0_get(icr, store) -> Optional[Any]:
    try:
        from itol.cache.l0_exact import L0Cache
        cache = L0Cache(store)
        key = cache.make_key(icr)
        if key is None:
            return None
        return cache.get(key, "bench"), key
    except Exception:
        return None, None


def _l0_set(icr, store, response_text: str, tokens_in: int, tokens_out: int) -> None:
    try:
        from itol.cache.l0_exact import L0Cache
        from itol.icr import ICRResponse, ContentBlock, UsageStats
        cache = L0Cache(store)
        key = cache.make_key(icr)
        if key is None:
            return
        fake_resp = ICRResponse(
            request_id=str(uuid.uuid4()),
            provider=icr.provider,
            model=icr.model,
            content=[ContentBlock.text(response_text[:2000])],
            usage=UsageStats(input_tokens=tokens_in, output_tokens=tokens_out),
            finish_reason="stop",
        )
        cache.set(key, "bench", fake_resp)
    except Exception as exc:
        logger.debug("L0 cache set failed: %s", exc)


# ---------------------------------------------------------------------------
# Single-sample ITOL run
# ---------------------------------------------------------------------------

async def _run_one_itol(
    sample: WorkloadSample,
    provider: ProviderConfig,
    config: BenchConfig,
    store,
    itol_config,
    baseline_result: Optional[BenchResult] = None,
) -> BenchResult:
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    error = None
    tokens_in = tokens_out = tokens_saved_actual = 0
    response_text = ""
    strategies_fired: list[str] = []
    cache_tier = "miss"
    qps_val: Optional[float] = None
    rollback = False
    pipeline_ms = 0.0

    try:
        if provider.name == "mock":
            # Mock: simulate ITOL savings
            import random
            rng = random.Random(hash(sample.sample_id + "itol"))
            raw, status, latency_ms = _mock_dispatch(sample, provider.model)
            response_text, tokens_in, tokens_out = _parse_openai_response(
                raw, provider.name, provider.model, request_id
            )
            # Simulate ITOL savings
            tokens_saved_actual = int(tokens_in * rng.uniform(0.15, 0.45))
            tokens_in = max(10, tokens_in - tokens_saved_actual)
            strategies_fired = rng.sample(["S1", "S3", "S6"], k=rng.randint(1, 3))
            qps_val = rng.uniform(0.97, 1.0)
            # FAQ: simulate cache hits for paraphrases
            if sample.workload == "faq" and sample.paraphrase_of:
                cache_tier = rng.choice(["l0", "l1", "miss"])
                if cache_tier != "miss":
                    tokens_saved_actual = tokens_in
            pipeline_ms = rng.uniform(5, 25)

        else:
            # Real ITOL pipeline
            t_pipe_start = time.perf_counter()
            icr = _build_icr(sample, provider.name, provider.model)

            # L0 cache check (FAQ workload: paraphrases may hit)
            if sample.workload == "faq":
                cached, cache_key = _l0_get(icr, store)
                if cached is not None:
                    cache_tier = "l0"
                    tokens_in_est = max(10, len(sample.prompt_text.split()) * 4 // 3)
                    tokens_saved_actual = tokens_in_est
                    response_text = cached.content[0].text if cached.content else ""
                    tokens_in = 0  # no dispatch needed
                    tokens_out = 0
                    pipeline_ms = (time.perf_counter() - t_pipe_start) * 1000
                    qps_val = 1.0
                    # Write result immediately for cache hit
                    cost = provider.cost_usd(tokens_in, tokens_out)
                    return BenchResult(
                        request_id=request_id,
                        sample_id=sample.sample_id,
                        workload=sample.workload,
                        provider=provider.name,
                        model=provider.model,
                        run_type="itol",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        tokens_saved=tokens_saved_actual,
                        latency_ms=pipeline_ms,
                        pipeline_ms=pipeline_ms,
                        strategies_fired=[],
                        cache_tier=cache_tier,
                        qps=1.0,
                        rollback=False,
                        equivalent_paid_cost_usd=cost,
                        prompt_text=sample.prompt_text[:600],
                        response_text=response_text[:600],
                        gold_answer=sample.gold_answer,
                    )

            # Run pipeline
            pipe_result = _run_pipeline(icr, store, itol_config)
            pipeline_ms = (time.perf_counter() - t_pipe_start) * 1000

            optimized_body = pipe_result["optimized_body"]
            tokens_saved_actual = pipe_result["tokens_saved"]
            strategies_fired = pipe_result["strategies_applied"]
            qps_val = pipe_result["qps"]
            rollback = pipe_result["rollback"]

            # Dispatch optimized body
            if provider.name == "cohere":
                # Convert optimized OpenAI body back to Cohere format for dispatch
                temp_sample = WorkloadSample(
                    sample_id=sample.sample_id, workload=sample.workload,
                    messages=optimized_body.get("messages", sample.messages),
                )
                dispatch_body = _build_cohere_body(temp_sample, provider.model, 0.0)
                url = f"{provider.base_url}{provider.chat_path}"

                async def do_cohere():
                    return await _dispatch_cohere(url, provider.api_key, dispatch_body)

                raw, status = await call_with_retry(do_cohere, provider_name=provider.name, tokens_estimate=300)
                if status != 200:
                    error = f"HTTP {status}"
                else:
                    response_text, tokens_in, tokens_out = _parse_cohere_response(raw)
            else:
                optimized_body["temperature"] = 0.0
                optimized_body["seed"] = 42
                url = f"{provider.base_url}{provider.chat_path}"

                async def do_openai():
                    return await _dispatch_openai(url, provider.api_key, optimized_body)

                msgs = optimized_body.get("messages")
                if isinstance(msgs, list):
                    tok_est = max(50, sum(len(str(m.get("content", ""))) for m in msgs) // 4)
                else:
                    tok_est = 200
                raw, status = await call_with_retry(do_openai, provider_name=provider.name, tokens_estimate=tok_est)
                if status != 200:
                    error = f"HTTP {status}: {str(raw.get('error',''))[:200]}"
                else:
                    response_text, tokens_in, tokens_out = _parse_openai_response(
                        raw, provider.name, provider.model, request_id
                    )

            latency_ms = (time.perf_counter() - t_start) * 1000

            # Store successful responses in L0 for FAQ
            if sample.workload == "faq" and not error and response_text:
                _l0_set(icr, store, response_text, tokens_in, tokens_out)

    except Exception as exc:
        error = str(exc)[:300]
        latency_ms = (time.perf_counter() - t_start) * 1000

    # Quality scoring (if we have both baseline and ITOL responses)
    quality_score: Optional[float] = None
    quality_method = "deterministic"
    if baseline_result and baseline_result.response_text and response_text and not error:
        q, qm = judge(
            workload=sample.workload,
            response_itol=response_text,
            response_baseline=baseline_result.response_text,
            gold_answer=sample.gold_answer,
            gold_entities=sample.gold_entities,
            cache_hit=(cache_tier != "miss"),
        )
        quality_score = q
        quality_method = qm

    cost = provider.cost_usd(tokens_in, tokens_out)

    return BenchResult(
        request_id=request_id,
        sample_id=sample.sample_id,
        workload=sample.workload,
        provider=provider.name,
        model=provider.model,
        run_type="itol",
        timestamp=datetime.now(timezone.utc).isoformat(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_saved=tokens_saved_actual,
        latency_ms=latency_ms,
        pipeline_ms=pipeline_ms,
        strategies_fired=strategies_fired,
        cache_tier=cache_tier,
        qps=qps_val,
        rollback=rollback,
        quality_score=quality_score,
        quality_method=quality_method,
        equivalent_paid_cost_usd=cost,
        prompt_text=sample.prompt_text[:600],
        response_text=response_text[:600],
        gold_answer=sample.gold_answer,
        error=error,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_itol(
    workload: str,
    provider: ProviderConfig,
    config: BenchConfig,
    samples: list[WorkloadSample],
    baseline_results: Optional[list[BenchResult]] = None,
    progress_cb=None,
) -> list[BenchResult]:
    """
    Run ITOL-optimized pipeline against provider for all samples.
    Resumable: skips sample_ids already in the JSONL file.
    """
    from itol.cache.store import Store
    from itol.config import ITOLConfig

    # Set up ITOL data dir + store
    data_dir = config.itol_data_dir
    store = Store(str(data_dir))

    # Calibrate if needed (enables optimize mode)
    itol_config = ITOLConfig()
    calib_dir = data_dir / "calibration"
    required = ["qps.json", "tau.json", "bandit_priors.json", "manifest_recall.json"]
    if not all((calib_dir / f).exists() for f in required):
        logger.info("Calibration data absent — running offline calibration...")
        from calibration.bootstrap import run_calibration
        run_calibration(offline=True, calib_dir=calib_dir, verbose=False)
        logger.info("Calibration complete.")

    # Build baseline lookup by sample_id for quality scoring
    baseline_map = {r.sample_id: r for r in (baseline_results or [])}

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = result_path(workload, f"{provider.name}_itol", date_str, config.output_dir)
    done_ids = completed_ids(out_path) if config.resume else set()

    results = []
    for i, sample in enumerate(samples):
        if sample.sample_id in done_ids:
            logger.debug("Skipping %s (already done)", sample.sample_id)
            continue

        logger.info("[itol/%s/%s] %d/%d sample=%s",
                    workload, provider.name, i+1, len(samples), sample.sample_id)

        baseline = baseline_map.get(sample.sample_id)
        result = await _run_one_itol(sample, provider, config, store, itol_config, baseline)
        append_result(result, out_path)
        results.append(result)

        if progress_cb:
            progress_cb(workload, provider.name, "itol", i + 1, len(samples))

    store.close()
    return results
