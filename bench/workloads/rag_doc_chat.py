"""
RAG / doc-chat workload — real HotpotQA (distractor split, validation set).

Data source priority:
  1. Bundled slice:  data/bench_corpora/hotpotqa_150_real.jsonl  (primary — no
     network required; commit this file so the benchmark is fully offline).
  2. HuggingFace Hub streaming: hotpot_qa / distractor / validation
     (fallback for anyone regenerating the slice with prepare_hotpotqa_slice.py).

Prompt construction (3-turn multi-hop RAG):
  Each user message includes the FULL retrieved context (all 10 paragraphs).
  Repeating the documents across turns gives ITOL's deduplication and context-
  windowing strategies real work to do:

    System: "You are a helpful assistant. Answer using only the provided
             documents. Cite the source document number."
    Turn 1: [10 docs] + warmup question  →  placeholder assistant answer
    Turn 2: [10 docs] + follow-up about supporting entity  →  placeholder
    Turn 3: [10 docs] + ORIGINAL HOTPOTQA QUESTION  ← judged against gold

  Expected strategy activation per request:
    S3 (dynamic windowing): fires on Turn 1's large retrieval context; keeps
       the 2 gold docs, drops 8 distractors → ~80 % context reduction for that
       segment.
    S1 (exact dedup):        fires on Turns 2 & 3; the document block is byte-
       identical to Turn 1's — S1 removes the repeated copies.
    S4 (RACR):               may fire for EXTRACTION/GENERATION_FACTUAL classes;
       replaces repeated full-text docs with compact references.
    S6 (hygiene):            always fires for minor whitespace savings.

Gold answer: HotpotQA's short factual answer for the final question.
The judge scores the ITOL vs baseline responses via EM + token-F1 against the
gold answer (standard HotpotQA evaluation protocol).

License: HotpotQA is CC BY-SA 4.0.
  Source: https://hotpotqa.github.io / https://huggingface.co/datasets/hotpot_qa
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from bench.workloads import WorkloadSample

_CORPORA_DIR = Path(__file__).parent.parent.parent / "data" / "bench_corpora"
_BUNDLED = _CORPORA_DIR / "hotpotqa_150_real.jsonl"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided documents. Cite the source document number in your answer."
)

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_bundled(n: int) -> Optional[list[dict]]:
    """Load from the pre-generated bundled JSONL slice (primary path)."""
    if not _BUNDLED.exists():
        return None
    examples = []
    with open(_BUNDLED, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue  # skip header
            examples.append(obj)
            if len(examples) >= n:
                break
    return examples if examples else None


def _load_from_hf(n: int) -> Optional[list[dict]]:
    """Stream from HuggingFace Hub (fallback; requires `datasets` + network)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
        examples = []
        for item in ds:
            if len(examples) >= n:
                break
            docs = []
            for title, sents in zip(item["context"]["title"],
                                    item["context"]["sentences"]):
                docs.append({"title": title, "text": " ".join(sents)})
            supporting = item["supporting_facts"]["title"]
            examples.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item.get("type", ""),
                "supporting_titles": list(dict.fromkeys(supporting)),
                "documents": docs,
            })
        return examples if examples else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _docs_block(documents: list[dict]) -> str:
    """Format 10 retrieved paragraphs as a numbered document block."""
    parts = []
    for i, doc in enumerate(documents, 1):
        parts.append(f"Document {i} [{doc['title']}]:\n{doc['text']}")
    return "\n\n".join(parts)


def _warmup_question(ex: dict, rng: random.Random) -> str:
    """A general overview question for Turn 1 (no gold answer needed)."""
    titles = [d["title"] for d in ex["documents"][:2]]
    templates = [
        f"What are the main topics covered in these documents?",
        f"What information do these documents provide about {titles[0]}?",
        f"Which documents are most relevant to the topic of {titles[0]}?",
    ]
    return rng.choice(templates)


def _warmup_answer(ex: dict) -> str:
    """Plausible placeholder assistant answer for the warmup question."""
    supporting = ex.get("supporting_titles", [])
    if supporting:
        return (
            f"The documents cover several topics. "
            f"Documents mentioning {supporting[0]} are most directly relevant."
        )
    return "The documents cover several related topics and entities."


def _followup_question(ex: dict) -> str:
    """Turn 2: ask about the first supporting entity (never 'yes'/'no')."""
    supporting = ex.get("supporting_titles", [])
    if supporting:
        entity = supporting[0]
    else:
        # fallback: use the answer if it's not a yes/no
        ans = ex.get("answer", "")
        entity = ans if len(ans.split()) > 1 else ex["question"].split()[-1]
    return f"Tell me more about {entity} based on the provided documents."


def _followup_answer(ex: dict) -> str:
    """Plausible placeholder assistant answer for the follow-up."""
    supporting = ex.get("supporting_titles", [])
    entity = supporting[0] if supporting else "the topic"
    return f"Based on the documents, {entity} is discussed in the context of the main question."


# ---------------------------------------------------------------------------
# Sample builder
# ---------------------------------------------------------------------------

def _example_to_sample(ex: dict, idx: int, seed: int) -> WorkloadSample:
    rng = random.Random(seed + idx)

    docs_block = _docs_block(ex["documents"])

    # All three user turns include the FULL document block.
    # This repetition is intentional: S1 (exact dedup) and S4 (RACR) fire on
    # Turns 2 and 3 to remove or compress the repeated document copies.
    turn1_user = f"{docs_block}\n\nQuestion: {_warmup_question(ex, rng)}"
    turn2_user = f"{docs_block}\n\nFollow-up: {_followup_question(ex)}"
    turn3_user = f"{docs_block}\n\nFinal question: {ex['question']}"

    messages = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": turn1_user},
        {"role": "assistant", "content": _warmup_answer(ex)},
        {"role": "user",      "content": turn2_user},
        {"role": "assistant", "content": _followup_answer(ex)},
        {"role": "user",      "content": turn3_user},
    ]

    sid = hashlib.sha256(f"rag_real_{ex['id']}".encode()).hexdigest()[:16]
    return WorkloadSample(
        sample_id=f"rag_{sid}",
        workload="rag",
        messages=messages,
        gold_answer=ex.get("answer"),
        gold_entities=ex.get("supporting_titles", []),
        metadata={
            "original_id": ex["id"],
            "type": ex.get("type", ""),
            "supporting_titles": ex.get("supporting_titles", []),
            "n_docs": len(ex.get("documents", [])),
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rag_samples(n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """
    Load n RAG samples from real HotpotQA data.

    Primary path: data/bench_corpora/hotpotqa_150_real.jsonl (bundled, offline).
    Run  scripts/prepare_hotpotqa_slice.py  to regenerate the bundled file.
    """
    raw = _load_bundled(n) or _load_from_hf(n)
    if raw is None:
        raise RuntimeError(
            "No HotpotQA data available. "
            "Run: python scripts/prepare_hotpotqa_slice.py"
        )
    raw = raw[:n]
    return [_example_to_sample(ex, i, seed) for i, ex in enumerate(raw)]
