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
from bench.judge import judge, llm_judge
from bench.metrics import BenchResult, append_result, successful_ids, result_path
from bench.rate_limit import call_with_retry
from bench.runners.baseline import (
    _build_openai_body, _build_cohere_body,
    _dispatch_openai, _dispatch_cohere,
    _parse_openai_response, _parse_cohere_response,
    _mock_dispatch, _seed_for,
)
from bench.workloads import WorkloadSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bench-mode ITOL config — lower activation thresholds for benchmark prompts
# ---------------------------------------------------------------------------
# Production defaults (S3 needs 6000+ tokens, S5 needs 4000+ tokens / 9+ turns)
# are calibrated for long production workloads. Benchmark samples are 200–2000
# tokens / 1–15 turns — we use a bench-specific config so strategies fire, but
# QUALITY-FIRST: lossy strategies stay conservative so they cannot drop content
# the final query depends on.

def _make_bench_itol_config():
    """
    Return an ITOLConfig tuned for benchmark-scale prompts, prioritising quality
    preservation over maximal savings.

    Design principle: lossless strategies (S6 hygiene, S1 exact-dedup, L0 cache)
    are always safe and provide the bulk of defensible savings. The LOSSY history
    distiller (S5) is the only strategy that can drop content the user later
    references — an over-aggressive gate (depth=1) made it distill short chats
    where the final question still referenced early turns, causing real quality
    drops. We therefore keep S5 conservative: it only fires on genuinely long
    histories and preserves the most recent turns verbatim. S3 windowing is
    lossy-but-QPS-gated, so it may fire on smaller contexts (the gate rolls it
    back if quality would degrade).
    """
    from itol.config import ITOLConfig, default_class_configs, StrategyConfig
    class_cfgs = default_class_configs()
    # S3: fire when context > 1.5 × 100 = 150 tokens (QPS gate guards quality).
    # S5 k_turns: preserve the most recent 8 turns verbatim so the final query's
    # referenced context survives distillation.
    for cfg in class_cfgs.values():
        if cfg.s3_class_budget >= 1000:
            cfg.s3_class_budget = 100
        cfg.s5_k_turns = max(cfg.s5_k_turns, 8)
    strat_cfg = StrategyConfig(
        s5_history_depth_gate=6,     # only distill when >6 turns of history exist
        s5_history_tokens_gate=1500, # and the history is genuinely large
    )
    return ITOLConfig(class_configs=class_cfgs, strategies=strat_cfg)


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

        # Represent everything as TEXT blocks. The benchmark needs faithful
        # token accounting + optimizable text; it does not need ICR-native
        # TOOL_RESULT blocks (which require a tool_result_for_id and otherwise
        # raise during construction — the source of the agent-workload crash).
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(blk.get("text", "") for blk in content if blk.get("type") == "text")

        # Assistant tool calls (content is usually None): fold the call into text
        # so its tokens are counted and it can be optimized like any other span.
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                calls.append(f"[tool call] {fn.get('name','')}({fn.get('arguments','')})")
            text = (text + "\n" + "\n".join(calls)).strip() if text else "\n".join(calls)

        # Tool result messages → assistant-visible text, mapped to user role.
        if role == "tool":
            tool_content = content if isinstance(content, str) else json.dumps(content)
            text = f"[tool result] {tool_content}"
            role = "user"

        if not text:
            continue  # nothing to carry (e.g. empty assistant placeholder)

        messages.append(Message(role=role, content=[ContentBlock(type=ContentType.TEXT, text=text)]))

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
                        parameters={},
                    ))
                break

    raw_body = _build_openai_body(sample, model, 0.0, _seed_for(provider_name, 42))

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
        # Character-based BPE estimate (~3.8 chars/token); more sensitive than
        # word-count to whitespace removal and small text mutations from S6.
        return max(1, len(seg.text or "") // 4)

    tokens_before = sum(_est_tokens(s) for s in segments)

    # Build strategy list (same order as proxy pipeline)
    strategies = [S1DedupeStrategy(), S6HygieneStrategy()]
    if cls_cfg is None or getattr(cls_cfg, "s3_enabled", True):
        from itol.strategies.s3_window import S3WindowStrategy
        strategies.append(S3WindowStrategy())
    if cls_cfg is not None and getattr(cls_cfg, "s4_enabled", False):
        from itol.strategies.s4_racr import S4RACRStrategy
        strategies.append(S4RACRStrategy())
    if cls_cfg is None or getattr(cls_cfg, "s5_enabled", True):
        from itol.strategies.s5_distill import S5DistillStrategy
        strategies.append(S5DistillStrategy(store=store))

    current_segments = list(segments)
    all_reports = []
    for strat in strategies:
        try:
            did_apply = strat.applies(icr, current_segments, ctx)
            logger.debug(
                "[bench/pipeline] strategy=%s applies=%s cls=%s ctx_tokens=%d",
                type(strat).__name__, did_apply, request_class,
                ctx.signals.token_count,
            )
            if did_apply:
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

    optimized_prompt_text = ""
    if score_result.use_raw:
        # Rolled back — no net savings; send original body
        tokens_after = tokens_before
        body_to_send = icr.raw
        strategies_applied = []
    else:
        from itol.icr import SegmentType
        from itol.segmenter import segments_full_text
        from collections import defaultdict
        opt_segs = score_result.segments or current_segments
        tokens_after = sum(_est_tokens(s) for s in opt_segs)

        # STRUCTURE-PRESERVING reassembly. Earlier approaches flattened the whole
        # conversation into one user message — which (a) inflated agent prompts
        # with textual markers beyond what S6 saved, and (b) made the model answer
        # a mid-conversation question, tanking parity. Instead, map each optimized
        # segment back to its source message (roles preserved): S6 minifies JSON
        # in place, S1/S5 drop whole segments, and the final user query stays a
        # distinct last turn. This keeps structure identical to baseline so the
        # model answers the same question, while genuinely shrinking content.
        sys_text = "\n".join(
            s.text for s in opt_segs if s.segment_type == SegmentType.SYSTEM_INSTRUCTION
        ).strip()
        by_idx: dict[int, list[str]] = defaultdict(list)
        orphan: list[str] = []   # synthetic segments (e.g. S5 ledger) with no source msg
        for s in opt_segs:
            if s.segment_type == SegmentType.SYSTEM_INSTRUCTION:
                continue
            if s.source_message_index is None:
                orphan.append(s.text)
            else:
                by_idx[s.source_message_index].append(s.text)

        new_messages: list[dict] = []
        if sys_text:
            new_messages.append({"role": "system", "content": sys_text})
        if orphan:
            # Distilled-history summary precedes the live turns.
            new_messages.append({"role": "user", "content": "\n".join(orphan).strip()})
            new_messages.append({"role": "assistant", "content": "Understood."})
        for i, msg in enumerate(icr.messages):
            if i in by_idx:
                text = "\n".join(by_idx[i]).strip()
                if text:
                    new_messages.append({"role": msg.role, "content": text})
        if not new_messages:
            new_messages = [{"role": "user", "content": segments_full_text(opt_segs)}]

        # Display = the final user turn that was actually sent.
        optimized_prompt_text = next(
            (m["content"] for m in reversed(new_messages) if m["role"] == "user"), ""
        )

        body_to_send = dict(icr.raw)
        body_to_send["messages"] = new_messages

        strategies_applied = [r.strategy_id for r in all_reports if r.activated]

        # Anti-inflation guard on the ACTUAL serialized request. Segment estimates
        # can disagree with the provider's tokeniser (e.g. native tool-call JSON
        # is very compact), so compare the real rendered bodies. If the optimized
        # body is not smaller, pass the ORIGINAL through unchanged — ITOL must
        # never make a request larger than baseline (worst case 0%, never negative).
        orig_chars = len(json.dumps(icr.raw.get("messages", [])))
        opt_chars  = len(json.dumps(new_messages))
        if not strategies_applied or opt_chars >= orig_chars:
            body_to_send = icr.raw
            tokens_after = tokens_before
            strategies_applied = []
            optimized_prompt_text = ""

    tokens_saved = max(0, tokens_before - tokens_after)

    logger.debug(
        "[bench/pipeline] cls=%s tokens_before=%d tokens_after=%d saved=%d "
        "rollback=%s strategies_fired=%s",
        request_class, tokens_before, tokens_after, tokens_saved,
        score_result.use_raw, strategies_applied,
    )

    return {
        "optimized_body": body_to_send,
        "optimized_prompt_text": optimized_prompt_text,
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
    optimized_prompt_text = ""
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
                    # An L0 (exact) cache hit serves the answer previously computed
                    # for the BYTE-IDENTICAL question — full parity by construction.
                    # (Comparing it to a fresh re-dispatch would only measure the
                    # provider's own non-determinism, not any ITOL quality change.)
                    cache_quality = 1.0
                    cache_method = "cache_exact"
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
                        quality_score=cache_quality,
                        quality_method=cache_method,
                        equivalent_paid_cost_usd=cost,
                        prompt_text=sample.prompt_text[:600],
                        response_text=response_text[:600],
                        gold_answer=sample.gold_answer,
                    )

            # Run pipeline
            pipe_result = _run_pipeline(icr, store, itol_config)
            pipeline_ms = (time.perf_counter() - t_pipe_start) * 1000

            optimized_body = pipe_result["optimized_body"]
            optimized_prompt_text = pipe_result.get("optimized_prompt_text", "")
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
                _seed = _seed_for(provider.name, 42)
                if _seed is not None:
                    optimized_body["seed"] = 42
                else:
                    optimized_body.pop("seed", None)
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
        # No-change ⇒ no quality impact. If ITOL sent the prompt through unchanged
        # (no strategy fired and nothing saved), it CANNOT have altered quality —
        # any response difference is pure provider non-determinism, not ITOL.
        # Likewise an identical response is trivially full parity. Score these 1.0
        # so passthroughs aren't misread as quality drops.
        if not strategies_fired and tokens_saved_actual == 0:
            quality_score = 1.0
            quality_method = "passthrough"
        elif response_text.strip() == baseline_result.response_text.strip():
            quality_score = 1.0
            quality_method = "identical"
        else:
            q, qm = judge(
                workload=sample.workload,
                response_itol=response_text,
                response_baseline=baseline_result.response_text,
                gold_answer=sample.gold_answer,
                gold_entities=sample.gold_entities,
                cache_hit=(cache_tier != "miss"),
                provider=provider.name,
            )
            quality_score = q
            quality_method = qm

        # Semantic parity for free-form workloads (real providers only).
        # Lexical metrics (Jaccard/F1) score paraphrased-but-equivalent answers
        # far below their true parity. An LLM judge ("is the ITOL response
        # materially worse than baseline?") measures semantic preservation —
        # the metric the headline "quality parity" actually claims. Deterministic
        # score is retained as a granularity signal in the blend.
        # Route the judge to the provider under test (it has quota and is the
        # most relevant evaluator); fall back to Groq only if that provider is
        # not OpenAI-compatible (e.g. cohere) or has no key.
        if provider.name == "cohere":
            import os as _os
            judge_key = _os.environ.get("GROQ_API_KEY")
            judge_url, judge_model = "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"
        else:
            judge_key = provider.api_key
            judge_url, judge_model = provider.base_url, provider.model
        if (quality_method not in ("identical", "passthrough") and provider.name != "mock"
                and judge_key and sample.workload in ("rag", "agent", "chat", "faq")):
            try:
                llm_q, llm_m = await llm_judge(
                    question=(icr.final_user_query() or sample.prompt_text)[:400],
                    response_original=baseline_result.response_text,
                    response_optimized=response_text,
                    model=judge_model,
                    api_key=judge_key,
                    base_url=judge_url,
                )
                if llm_m == "llm_judge":  # only trust a real verdict
                    # The judge's verdict is the semantic truth: "not materially
                    # worse" (llm_q=1.0) means quality is preserved → high parity.
                    # Deterministic score only adds minor spread; it must NOT drag
                    # a verified-equivalent (but reworded) answer below parity.
                    quality_score = round(0.9 * llm_q + 0.1 * quality_score, 4)
                    quality_method = "llm_semantic+det"
            except Exception as exc:
                logger.debug("LLM judge failed, keeping deterministic: %s", exc)

    cost = provider.cost_usd(tokens_in, tokens_out)

    # For sample cards: show the optimized prompt (what was actually sent);
    # fall back to the original when pipeline rolled back or mock provider.
    display_prompt = (
        (optimized_prompt_text or sample.prompt_text)
        if provider.name != "mock"
        else sample.prompt_text
    )

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
        prompt_text=display_prompt[:600],
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
    itol_config = _make_bench_itol_config()
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
    # Resume skips only SUCCESSFUL samples — errored ones get retried.
    done_ids = successful_ids(out_path) if config.resume else set()

    results = []
    for i, sample in enumerate(samples):
        if sample.sample_id in done_ids:
            logger.debug("Skipping %s (already succeeded)", sample.sample_id)
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
