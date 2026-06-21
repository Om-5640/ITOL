"""
Benchmark workloads — each produces a list of WorkloadSample objects.

WorkloadSample is the canonical input unit for both runners.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkloadSample:
    sample_id: str                          # stable ID for resumability
    workload: str                           # "rag" | "agent" | "chat" | "faq"
    messages: list[dict]                    # OpenAI-format message list
    gold_answer: Optional[str] = None       # expected short answer (EM/F1 scoring)
    gold_entities: Optional[list[str]] = None  # for agent entity coverage
    paraphrase_of: Optional[str] = None     # for FAQ: base query ID
    metadata: dict = field(default_factory=dict)

    @property
    def prompt_text(self) -> str:
        """Concatenation of all user turns (for display in report)."""
        parts = [m["content"] for m in self.messages if m.get("role") == "user"]
        return "\n".join(parts)[:800]

    @property
    def system_text(self) -> str:
        for m in self.messages:
            if m.get("role") == "system":
                return m["content"]
        return ""


def load_workload(name: str, n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """Load n samples for the named workload. Auto-selects data source."""
    if name == "rag":
        from bench.workloads.rag_doc_chat import load_rag_samples
        return load_rag_samples(n=n, seed=seed)
    elif name == "agent":
        from bench.workloads.agent_loop import load_agent_samples
        return load_agent_samples(n=n, seed=seed)
    elif name == "chat":
        from bench.workloads.multi_turn_chat import load_chat_samples
        return load_chat_samples(n=n, seed=seed)
    elif name == "faq":
        from bench.workloads.faq_cache import load_faq_samples
        return load_faq_samples(n=n, seed=seed)
    else:
        raise ValueError(f"Unknown workload: {name!r}")
