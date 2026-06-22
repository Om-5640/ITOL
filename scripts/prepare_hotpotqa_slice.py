"""
Download 150 examples from HotpotQA (distractor, validation) and write them to
data/bench_corpora/hotpotqa_150_real.jsonl.

Usage:
    python scripts/prepare_hotpotqa_slice.py [--n 150] [--seed 42]
    python scripts/prepare_hotpotqa_slice.py --from-bundled   # re-enrich existing file

Requires (HF path only): pip install datasets pyarrow

License of the output file: CC BY-SA 4.0 (inherited from HotpotQA).
Dataset URL: https://huggingface.co/datasets/hotpot_qa
"""
from __future__ import annotations

import argparse
import json
import random
import re
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

# Tokens that indicate a pure number answer (stripped before digit check)
_NUMBER_UNITS = ("seated", "people", "members", "years", "%", "$", "km", "mi",
                 "million", "billion", "thousand")


def _generate_turn2(answer: str, supporting_titles: list[str], qtype: str) -> str:
    """
    Generate a natural-sounding turn-2 follow-up question.

    For yes/no comparison questions and number-only answers the literal gold
    answer doesn't form a meaningful follow-up ("Tell me more about yes.").
    In those cases fall back to asking about one of the supporting Wikipedia
    entities — which always produces a coherent follow-up and keeps all three
    turns grounded in the actual documents.
    """
    ans = answer.strip()
    ans_lower = ans.lower()

    # Yes/no comparison answers
    if ans_lower in ("yes", "no"):
        entity = supporting_titles[0] if supporting_titles else "the topic"
        return f"Tell me more about {entity}."

    # Pure numbers or number+unit answers (e.g. "3,677 seated", "42")
    stripped = ans.replace(",", "").replace(" ", "")
    for unit in _NUMBER_UNITS:
        stripped = stripped.replace(unit, "")
    if stripped.isdigit():
        entity = supporting_titles[0] if supporting_titles else "the topic"
        return f"Tell me more about {entity}."

    # Empty or single-character answers
    if len(ans) < 2:
        entity = supporting_titles[0] if supporting_titles else "the topic"
        return f"Tell me more about {entity}."

    # Normal entity/phrase answer
    return f"Tell me more about {ans}."


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
            "supporting_titles": list(dict.fromkeys(supporting)),
            "documents": docs,
        })
    print(f"  Loaded {len(examples)} examples.")
    return examples


def _load_from_bundled() -> list[dict]:
    """
    Load raw examples from the existing bundled JSONL (strips turn_questions so
    _enrich_turns re-generates them cleanly with the fixed logic).
    """
    if not OUT_PATH.exists():
        return []
    examples = []
    with open(OUT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_meta"):
                continue
            # Drop stale turn_questions so _enrich_turns regenerates them
            obj.pop("turn_questions", None)
            examples.append(obj)
    print(f"  Re-loaded {len(examples)} examples from bundled file.")
    return examples


def _enrich_turns(examples: list[dict], seed: int) -> list[dict]:
    """
    Add turn_questions: 3 questions per example for the multi-turn RAG prompt.

    Turn 1: the original multi-hop question (the gold question, used in Turn 3
            of the workload prompt — stored here for documentation/reference).
    Turn 2: a sensible follow-up about a named entity in the documents (never
            "Tell me more about yes." or "Tell me more about 42.").
    Turn 3: "How does this relate to {distractor_title}?" — cross-doc link.
    """
    rng = random.Random(seed)
    enriched = []
    for ex in examples:
        gold_titles = ex.get("supporting_titles", [])
        distractor_docs = [d for d in ex["documents"] if d["title"] not in gold_titles]

        turn2 = _generate_turn2(ex["answer"], gold_titles, ex.get("type", ""))

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
    ap.add_argument(
        "--from-bundled",
        action="store_true",
        help=(
            "Re-enrich from the existing bundled JSONL instead of fetching "
            "from HuggingFace Hub. Useful for fixing turn_questions without "
            "requiring network access or the datasets library."
        ),
    )
    args = ap.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.from_bundled:
        examples = _load_from_bundled()
        if not examples:
            print("ERROR: No bundled file found. Run without --from-bundled first.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        try:
            examples = _load_from_hf(args.n)
        except Exception as e:
            print(f"ERROR: Could not load from HuggingFace: {e}", file=sys.stderr)
            sys.exit(1)

    examples = examples[: args.n]
    examples = _enrich_turns(examples, args.seed)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(_HEADER) + "\n")
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"Written {len(examples)} examples to {OUT_PATH} ({size_kb} KB)")

    # Quick sanity check
    bad = [ex for ex in examples
           if ex["turn_questions"][1].lower() in ("tell me more about yes.",
                                                   "tell me more about no.")]
    if bad:
        print(f"WARNING: {len(bad)} examples still have yes/no turn-2 questions!",
              file=sys.stderr)
    else:
        print("Sanity check: no yes/no turn-2 questions found.")
    print("Done. Commit this file so the benchmark runs offline without HF.")


if __name__ == "__main__":
    main()
