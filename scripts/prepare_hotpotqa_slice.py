"""
Download 150 examples from HotpotQA (distractor, validation) and write them to
data/bench_corpora/hotpotqa_150_real.jsonl.

Usage:
    python scripts/prepare_hotpotqa_slice.py [--n 150] [--seed 42]

Requires: pip install datasets pyarrow

License of the output file: CC BY-SA 4.0 (inherited from HotpotQA).
Dataset URL: https://huggingface.co/datasets/hotpot_qa
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "data" / "bench_corpora" / "hotpotqa_150_real.jsonl"

_HEADER = {
    "_meta": True,
    "source": "HotpotQA",
    "config": "distractor",
    "split": "validation",
    "license": "CC BY-SA 4.0",
    "url": "https://huggingface.co/datasets/hotpot_qa",
    "generator": "scripts/prepare_hotpotqa_slice.py",
    "description": (
        "150 examples from HotpotQA distractor/validation. "
        "Each example has ~10 paragraphs (2 gold + 8 distractors), a multi-hop "
        "question, and a short answer. Used by bench/workloads/rag_doc_chat.py."
    ),
}


def _load_from_hf(n: int) -> list[dict]:
    from datasets import load_dataset
    print("Loading HotpotQA distractor/validation from HuggingFace Hub...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    examples = []
    for item in ds:
        if len(examples) >= n:
            break
        docs = []
        for title, sents in zip(item["context"]["title"], item["context"]["sentences"]):
            docs.append({"title": title, "text": " ".join(sents)})
        supporting = item["supporting_facts"]["title"]
        examples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item.get("type", ""),
            "supporting_titles": list(dict.fromkeys(supporting)),  # dedup, preserve order
            "documents": docs,
        })
    print(f"  Loaded {len(examples)} examples.")
    return examples


def _enrich_turns(examples: list[dict], seed: int) -> list[dict]:
    """
    Add turn_questions: 3 questions per example for the multi-turn RAG prompt.

    Turn 1: original multi-hop question (asks about something in gold docs)
    Turn 2: "Tell me more about <entity from gold answer / supporting title>"
    Turn 3: "How does this relate to <entity from a distractor paragraph>?"
    """
    rng = random.Random(seed)
    enriched = []
    for ex in examples:
        gold_titles = ex.get("supporting_titles", [])
        distractor_docs = [d for d in ex["documents"] if d["title"] not in gold_titles]

        # Turn 2 entity: prefer gold answer if short, else first supporting title
        answer = ex["answer"].strip()
        t2_entity = answer if len(answer.split()) <= 4 else (gold_titles[0] if gold_titles else answer)
        turn2 = f"Tell me more about {t2_entity}."

        # Turn 3 entity: pick a distractor title (or second gold title)
        if distractor_docs:
            dist_title = rng.choice(distractor_docs)["title"]
        elif len(gold_titles) > 1:
            dist_title = gold_titles[1]
        else:
            dist_title = "the other topics mentioned"
        turn3 = f"How does this relate to {dist_title}?"

        enriched.append({
            **ex,
            "turn_questions": [ex["question"], turn2, turn3],
        })
    return enriched


def main():
    ap = argparse.ArgumentParser(description="Prepare HotpotQA 150-example slice.")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        examples = _load_from_hf(args.n)
    except Exception as e:
        print(f"ERROR: Could not load from HuggingFace: {e}", file=sys.stderr)
        sys.exit(1)

    examples = _enrich_turns(examples, args.seed)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_HEADER) + "\n")
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"Written {len(examples)} examples to {OUT_PATH} ({size_kb} KB)")
    print("Done. Commit this file so the benchmark runs offline without HF.")


if __name__ == "__main__":
    main()
