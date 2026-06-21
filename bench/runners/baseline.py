"""
Baseline runner — dispatches prompts directly to provider APIs with NO ITOL optimization.

Supports:
  - Groq (OpenAI-compatible)
  - Mistral (OpenAI-compatible)
  - Cohere (custom dialect)
  - MockProvider (no API calls; for smoke tests)

All requests use temperature=0 and seed=42 for reproducibility.
Results are written to data/bench_results/raw/{workload}_{provider}_{date}.jsonl
(resumable: skips sample_ids already present in the file).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bench.config import ProviderConfig, BenchConfig
from bench.metrics import BenchResult, append_result, completed_ids, result_path
from bench.rate_limit import call_with_retry
from bench.workloads import WorkloadSample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP dispatch helpers
# ---------------------------------------------------------------------------

async def _dispatch_openai(
    url: str,
    api_key: str,
    body: dict,
) -> tuple[dict, int]:
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
    return resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code


async def _dispatch_cohere(
    url: str,
    api_key: str,
    body: dict,
) -> tuple[dict, int]:
    import httpx
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=body)
    return resp.json() if resp.status_code == 200 else {"error": resp.text}, resp.status_code


# ---------------------------------------------------------------------------
# Mock provider (no API key needed; deterministic synthetic responses)
# ---------------------------------------------------------------------------

_MOCK_RESPONSES = [
    "Based on the provided information, the key insight is that this requires a systematic approach.",
    "The answer to your question involves several important considerations worth exploring carefully.",
    "After analyzing the context, I can provide a comprehensive response addressing the main points.",
    "This is an excellent question. The fundamental principle here relates to how systems interact.",
    "Let me break this down: the core concept involves balancing multiple competing factors effectively.",
]


def _mock_dispatch(sample: WorkloadSample, model: str) -> tuple[dict, int, float]:
    """Return a synthetic response without making any API call."""
    import random
    rng = random.Random(hash(sample.sample_id))
    response_text = rng.choice(_MOCK_RESPONSES)
    # Simulate realistic token counts
    prompt_text = " ".join(m["content"] for m in sample.messages if isinstance(m.get("content"), str))
    tokens_in = max(10, len(prompt_text.split()) * 4 // 3)  # rough BPE estimate
    tokens_out = rng.randint(30, 120)
    raw = {
        "id": f"mock_{uuid.uuid4().hex[:8]}",
        "model": model,
        "choices": [{"message": {"content": response_text, "role": "assistant"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out, "total_tokens": tokens_in + tokens_out},
    }
    latency_ms = rng.uniform(80, 300)  # realistic mock latency
    return raw, 200, latency_ms


# ---------------------------------------------------------------------------
# Body builders per provider dialect
# ---------------------------------------------------------------------------

def _build_openai_body(sample: WorkloadSample, model: str, temperature: float = 0.0, seed: int = 42) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "messages": sample.messages,
        "temperature": temperature,
        "max_tokens": 512,
    }
    if seed is not None:
        body["seed"] = seed
    return body


def _build_cohere_body(sample: WorkloadSample, model: str, temperature: float = 0.0) -> dict:
    """Convert OpenAI-format messages to Cohere chat format."""
    preamble = ""
    chat_history = []
    message = ""

    for m in sample.messages:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "system":
            preamble = content
        elif role == "user":
            if message:
                chat_history.append({"role": "USER", "message": message})
            message = content
        elif role == "assistant":
            if message:
                chat_history.append({"role": "USER", "message": message})
                message = ""
            chat_history.append({"role": "CHATBOT", "message": content})

    return {
        "model": model,
        "message": message or "Hello",
        "chat_history": chat_history,
        "preamble": preamble,
        "temperature": temperature,
        "max_tokens": 512,
    }


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _parse_openai_response(raw: dict, provider: str, model: str, request_id: str) -> tuple[str, int, int]:
    """Returns (response_text, tokens_in, tokens_out)."""
    try:
        text = raw["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        text = str(raw.get("error", ""))
    usage = raw.get("usage", {})
    return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def _parse_cohere_response(raw: dict) -> tuple[str, int, int]:
    try:
        text = raw.get("text", "")
    except Exception:
        text = ""
    meta = raw.get("meta", {}).get("tokens", {})
    return text, meta.get("input_tokens", 0), meta.get("output_tokens", 0)


# ---------------------------------------------------------------------------
# Single-sample baseline run
# ---------------------------------------------------------------------------

async def _run_one_baseline(
    sample: WorkloadSample,
    provider: ProviderConfig,
    config: BenchConfig,
    date_str: str,
) -> BenchResult:
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()
    error = None
    tokens_in = tokens_out = 0
    response_text = ""
    rate_limited = 0

    try:
        if provider.name == "mock":
            raw, status, latency_ms = _mock_dispatch(sample, provider.model)
            response_text, tokens_in, tokens_out = _parse_openai_response(
                raw, provider.name, provider.model, request_id
            )
        elif provider.name == "cohere":
            body = _build_cohere_body(sample, provider.model, config.temperature)
            url = f"{provider.base_url}{provider.chat_path}"

            async def do_cohere():
                return await _dispatch_cohere(url, provider.api_key, body)

            prompt_len = sum(len(str(m.get("content", ""))) for m in sample.messages)
            raw, status = await call_with_retry(
                do_cohere, provider_name=provider.name,
                tokens_estimate=max(200, prompt_len // 4),
            )
            if status != 200:
                error = f"HTTP {status}: {raw.get('error', '')}"
            else:
                response_text, tokens_in, tokens_out = _parse_cohere_response(raw)
            latency_ms = (time.perf_counter() - t_start) * 1000
        else:
            # OpenAI-compatible (Groq, Mistral)
            body = _build_openai_body(sample, provider.model, config.temperature, config.seed)
            url = f"{provider.base_url}{provider.chat_path}"

            async def do_openai():
                return await _dispatch_openai(url, provider.api_key, body)

            raw, status = await call_with_retry(
                do_openai, provider_name=provider.name,
                tokens_estimate=max(200, len(" ".join(
                    str(m.get("content","")) for m in sample.messages
                ).split()) * 4 // 3),
            )
            if status != 200:
                error = f"HTTP {status}: {str(raw.get('error', ''))[:200]}"
            else:
                response_text, tokens_in, tokens_out = _parse_openai_response(
                    raw, provider.name, provider.model, request_id
                )
            latency_ms = (time.perf_counter() - t_start) * 1000

    except Exception as exc:
        error = str(exc)[:300]
        latency_ms = (time.perf_counter() - t_start) * 1000

    cost = provider.cost_usd(tokens_in, tokens_out)

    return BenchResult(
        request_id=request_id,
        sample_id=sample.sample_id,
        workload=sample.workload,
        provider=provider.name,
        model=provider.model,
        run_type="baseline",
        timestamp=datetime.now(timezone.utc).isoformat(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        equivalent_paid_cost_usd=cost,
        prompt_text=sample.prompt_text[:600],
        response_text=response_text[:600],
        gold_answer=sample.gold_answer,
        rate_limited_count=rate_limited,
        error=error,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_baseline(
    workload: str,
    provider: ProviderConfig,
    config: BenchConfig,
    samples: list[WorkloadSample],
    progress_cb=None,
) -> list[BenchResult]:
    """
    Run baseline (no ITOL) against provider for all samples.
    Resumable: skips sample_ids already in the JSONL file.
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = result_path(workload, provider.name, date_str, config.output_dir)
    done_ids = completed_ids(out_path) if config.resume else set()

    results = []
    for i, sample in enumerate(samples):
        if sample.sample_id in done_ids:
            logger.debug("Skipping %s (already done)", sample.sample_id)
            continue

        logger.info("[baseline/%s/%s] %d/%d sample=%s",
                    workload, provider.name, i+1, len(samples), sample.sample_id)
        result = await _run_one_baseline(sample, provider, config, date_str)
        append_result(result, out_path)
        results.append(result)

        if progress_cb:
            progress_cb(workload, provider.name, "baseline", i + 1, len(samples))

    return results
