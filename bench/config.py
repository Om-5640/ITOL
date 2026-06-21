"""
Benchmark configuration — providers, models, workloads, pricing, rate limits.

All prices are for paid-tier equivalent cost calculation.
Free-tier actual cost = $0; we report what the SAME requests cost on paid tier.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Repository root + default data paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
DATA_DIR = _REPO / "data"
BENCH_RESULTS_DIR = DATA_DIR / "bench_results"
BENCH_CORPORA_DIR = DATA_DIR / "bench_corpora"
REPORT_DIR = BENCH_RESULTS_DIR / "report"


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str                        # slug: "groq" | "mistral" | "cohere" | "mock"
    model: str                       # model ID to call
    base_url: str                    # API base URL (no trailing slash)
    api_key_env: str                 # env var that holds the API key
    chat_path: str                   # path appended to base_url for chat
    rps: float                       # max requests per second (conservative)
    tpm: int                         # max tokens per minute
    input_price_mtoken: float        # paid-tier $/million input tokens
    output_price_mtoken: float       # paid-tier $/million output tokens
    judge_model: Optional[str] = None  # override model for LLM-judge role
    retry_max: int = 5

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)

    @property
    def available(self) -> bool:
        return self.name == "mock" or bool(self.api_key)

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in  * self.input_price_mtoken  / 1_000_000
            + tokens_out * self.output_price_mtoken / 1_000_000
        )

    def scale_monthly_cost(self, tokens_per_day: int, savings_pct: float) -> dict:
        """Cost without/with ITOL for a given daily token volume (monthly = ×30)."""
        monthly = tokens_per_day * 30
        # Assume 75% input, 25% output split (typical chat ratio)
        tin = int(monthly * 0.75)
        tout = int(monthly * 0.25)
        without = self.cost_usd(tin, tout)
        with_itol = self.cost_usd(int(tin * (1 - savings_pct)), tout)
        return {
            "without_usd": round(without, 2),
            "with_usd":    round(with_itol, 2),
            "saved_usd":   round(without - with_itol, 2),
            "savings_pct": round(savings_pct * 100, 1),
        }


# Paid-tier pricing (June 2026 — update if stale via `python -m bench check --prices`)
PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        model="llama-3.1-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        chat_path="/chat/completions",
        rps=0.5,           # very conservative; free tier ~30 req/min
        tpm=6_000,
        input_price_mtoken=0.59,
        output_price_mtoken=0.79,
        judge_model="llama-3.1-8b-instant",
    ),
    "mistral": ProviderConfig(
        name="mistral",
        model="mistral-small-latest",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY",
        chat_path="/chat/completions",
        rps=1.0,
        tpm=500_000,
        input_price_mtoken=0.20,
        output_price_mtoken=0.60,
    ),
    "cohere": ProviderConfig(
        name="cohere",
        model="command-r",
        base_url="https://api.cohere.ai/v1",
        api_key_env="COHERE_API_KEY",
        chat_path="/chat",
        rps=0.3,
        tpm=10_000,
        input_price_mtoken=0.15,
        output_price_mtoken=0.60,
    ),
    "mock": ProviderConfig(
        name="mock",
        model="mock-model",
        base_url="http://localhost:0",
        api_key_env="",
        chat_path="/chat/completions",
        rps=1000.0,
        tpm=10_000_000,
        input_price_mtoken=0.59,   # use groq pricing for cost illustration
        output_price_mtoken=0.79,
    ),
}


# ---------------------------------------------------------------------------
# Workload configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    name: str
    label: str
    description: str
    n_samples: int = 150
    turns_per_sample: int = 1
    request_class_hint: str = "CHAT_OPEN"


WORKLOADS: dict[str, WorkloadConfig] = {
    "rag": WorkloadConfig(
        name="rag",
        label="RAG / Doc-Chat",
        description="HotpotQA multi-hop questions with retrieved docs (3 turns/example)",
        n_samples=150,
        turns_per_sample=3,
        request_class_hint="SUMMARIZATION",
    ),
    "agent": WorkloadConfig(
        name="agent",
        label="Agent Tool Loop",
        description="Synthetic agent trajectories (4–8 turns, tool calls + results)",
        n_samples=150,
        turns_per_sample=6,
        request_class_hint="AGENT_TOOL_LOOP",
    ),
    "chat": WorkloadConfig(
        name="chat",
        label="Multi-Turn Chat",
        description="10–15-turn dialogues; ITOL optimizes history on each turn",
        n_samples=150,
        turns_per_sample=10,
        request_class_hint="CHAT_OPEN",
    ),
    "faq": WorkloadConfig(
        name="faq",
        label="FAQ Cache",
        description="50 base queries + 100 paraphrases — headline metric is cache hit rate",
        n_samples=150,
        turns_per_sample=1,
        request_class_hint="GENERATION_FACTUAL",
    ),
}


# ---------------------------------------------------------------------------
# Benchmark run configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    providers: list[str] = field(default_factory=lambda: ["groq", "mistral", "cohere"])
    workloads: list[str] = field(default_factory=lambda: list(WORKLOADS))
    n_samples: int = 150
    smoke: bool = False              # 5 samples, mock provider only
    resume: bool = True              # skip already-completed request_ids
    temperature: float = 0.0         # deterministic outputs
    seed: int = 42
    output_dir: Path = BENCH_RESULTS_DIR
    itol_data_dir: Path = DATA_DIR

    def active_providers(self) -> list[ProviderConfig]:
        names = ["mock"] if self.smoke else self.providers
        return [PROVIDERS[n] for n in names if n in PROVIDERS and PROVIDERS[n].available]

    def active_workloads(self) -> list[WorkloadConfig]:
        return [WORKLOADS[n] for n in self.workloads if n in WORKLOADS]

    def n_for_workload(self) -> int:
        return 5 if self.smoke else self.n_samples
