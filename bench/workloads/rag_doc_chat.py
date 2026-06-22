"""
RAG / doc-chat workload.

Data source priority:
  1. datasets library → HotpotQA (auto-download)
  2. Bundled synthetic file: data/bench_corpora/hotpotqa_150.jsonl
  3. Pure in-memory synthetic generator (always works)

Each HotpotQA example becomes a 3-turn conversation:
  Turn 1: original question + retrieved context paragraphs
  Turn 2: follow-up "Tell me more about X" (forces S5/S3 to fire)
  Turn 3: "What is the connection between A and B?" (history distillation)

~20% deliberate doc redundancy injected across turns so S1/S4 fire.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from bench.workloads import WorkloadSample

_CORPORA_DIR = Path(__file__).parent.parent.parent / "data" / "bench_corpora"
_BUNDLED = _CORPORA_DIR / "hotpotqa_150.jsonl"

# ---------------------------------------------------------------------------
# Synthetic HotpotQA generator (always available, no internet required)
# ---------------------------------------------------------------------------

_TOPICS = [
    ("Albert Einstein", "theory of relativity", "physics", "Germany", "E=mc²"),
    ("Marie Curie", "radioactivity research", "chemistry", "Poland", "polonium"),
    ("Isaac Newton", "law of gravitation", "physics", "England", "calculus"),
    ("Charles Darwin", "theory of evolution", "biology", "England", "natural selection"),
    ("Nikola Tesla", "alternating current", "electrical engineering", "Serbia", "Tesla coil"),
    ("Galileo Galilei", "heliocentric model", "astronomy", "Italy", "telescope improvements"),
    ("Alan Turing", "theoretical computing", "mathematics", "England", "Turing machine"),
    ("Ada Lovelace", "first algorithm", "mathematics", "England", "Babbage engine"),
    ("Richard Feynman", "quantum electrodynamics", "physics", "USA", "Feynman diagrams"),
    ("James Watson", "DNA double helix", "biology", "USA", "nucleotide base pairs"),
    ("Rosalind Franklin", "X-ray crystallography", "chemistry", "England", "Photo 51"),
    ("Carl Sagan", "planetary science", "astronomy", "USA", "Cosmos series"),
    ("Linus Torvalds", "Linux kernel", "computer science", "Finland", "open source"),
    ("Tim Berners-Lee", "World Wide Web", "computer science", "England", "HTTP protocol"),
    ("Grace Hopper", "COBOL language", "computer science", "USA", "compiler"),
    ("Stephen Hawking", "black hole radiation", "physics", "England", "A Brief History of Time"),
    ("Noam Chomsky", "generative grammar", "linguistics", "USA", "universal grammar"),
    ("Claude Shannon", "information theory", "mathematics", "USA", "entropy in communication"),
    ("John von Neumann", "computer architecture", "mathematics", "Hungary", "stored-program concept"),
    ("Leonhard Euler", "graph theory", "mathematics", "Switzerland", "Euler's identity"),
]

_COMPARISON_TEMPLATES = [
    ("Who was born earlier, {a} or {b}?", "{a}"),
    ("Which field was {a} known for — {field_a} or {field_b}?", "{field_a}"),
    ("Did {a} and {b} both work in {shared_field}?", "No" if True else "Yes"),
    ("Which scientist contributed to {contribution}?", "{a}"),
]

_DOC_TEMPLATE = """\
{title} ({dates}) was a pioneering figure in {field}. \
{pronoun} is best known for {contribution}, which fundamentally changed \
our understanding of {field}. Born in {country}, {name_short} pursued \
research that led to breakthroughs including {discovery}. \
{name_short}'s work continues to influence modern {field} today."""

_FOLLOW_UP_TEMPLATES = [
    "Tell me more about {name}'s contribution to {field}.",
    "Can you elaborate on how {name}'s discovery of {discovery} changed {field}?",
    "What was the historical significance of {name}'s work in {country}?",
]

_BRIDGE_TEMPLATES = [
    "What is the connection between {a}'s work in {field_a} and {b}'s work in {field_b}?",
    "How did {a}'s discovery of {disc_a} influence later work like {b}'s {disc_b}?",
    "Compare the contributions of {a} and {b} to their respective fields.",
]


def _make_doc(topic: tuple, pronoun: str = "They") -> str:
    name, contribution, field, country, discovery = topic
    name_short = name.split()[0]
    dates = f"1{random.randint(6, 9)}{random.randint(0, 9)}{random.randint(0, 9)}-"
    dates += f"1{random.randint(9, 9)}{random.randint(0, 9)}{random.randint(0, 9)}"
    return _DOC_TEMPLATE.format(
        title=name, dates=dates, field=field, pronoun=pronoun,
        contribution=contribution, country=country,
        name_short=name_short, discovery=discovery,
    )


def _generate_synthetic(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    topics_pool = _TOPICS * (n // len(_TOPICS) + 2)
    rng.shuffle(topics_pool)

    for i in range(n):
        t_a = topics_pool[i * 2 % len(_TOPICS)]
        t_b = topics_pool[(i * 2 + 1) % len(_TOPICS)]
        name_a, contrib_a, field_a, country_a, disc_a = t_a
        name_b, contrib_b, field_b, country_b, disc_b = t_b

        doc_a = _make_doc(t_a)
        doc_b = _make_doc(t_b)
        # Inject ~20% redundancy: occasionally repeat doc_a
        redundant = rng.random() < 0.20

        question = rng.choice([
            f"What was {name_a}'s most famous contribution to {field_a}?",
            f"Which country was {name_a} born in?",
            f"What is {name_a} best known for?",
            f"In what field did {name_a} make their key discovery?",
        ])
        answer = rng.choice([disc_a, country_a, contrib_a, field_a])

        follow_up_tmpl = rng.choice(_FOLLOW_UP_TEMPLATES)
        follow_up = follow_up_tmpl.format(
            name=name_a, field=field_a, discovery=disc_a, country=country_a
        )

        bridge = rng.choice(_BRIDGE_TEMPLATES).format(
            a=name_a, b=name_b, field_a=field_a, field_b=field_b,
            disc_a=disc_a, disc_b=disc_b,
        )

        context_docs = [doc_a, doc_b]
        if redundant:
            context_docs.append(doc_a)  # deliberate redundancy for S1/S4

        examples.append({
            "id": f"synth_{i:04d}",
            "question": question,
            "answer": answer,
            "follow_up": follow_up,
            "bridge_question": bridge,
            "context_docs": context_docs,
            "entities": [name_a, name_b, disc_a, field_a],
        })
    return examples


# ---------------------------------------------------------------------------
# HotpotQA loader (tries datasets library first, then bundled, then synthetic)
# ---------------------------------------------------------------------------

def _try_hf_hotpotqa(n: int) -> Optional[list[dict]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation", streaming=True)
        examples = []
        for item in ds:
            if len(examples) >= n:
                break
            # Flatten context paragraphs into doc strings
            docs = []
            for title, sentences in zip(item["context"]["title"],
                                        item["context"]["sentences"]):
                docs.append(f"[{title}] " + " ".join(sentences))
            examples.append({
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "context_docs": docs,
                "entities": [],
            })
        if examples:
            return examples
    except Exception:
        pass
    return None


def _load_bundled(n: int) -> Optional[list[dict]]:
    if not _BUNDLED.exists():
        return None
    examples = []
    with open(_BUNDLED, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                if len(examples) >= n:
                    break
    return examples if examples else None


# ---------------------------------------------------------------------------
# Convert raw example → 3-turn WorkloadSample
# ---------------------------------------------------------------------------

def _example_to_sample(ex: dict, idx: int, seed: int) -> WorkloadSample:
    rng = random.Random(seed + idx)
    docs_text = "\n\n".join(f"Document {i+1}:\n{d}" for i, d in enumerate(ex["context_docs"]))

    turn1_content = f"{docs_text}\n\nQuestion: {ex['question']}"
    turn2_content = ex.get("follow_up") or rng.choice([
        f"Tell me more about the key entities mentioned in the previous answer.",
        f"Can you expand on the most important point from your response?",
        f"What additional context is relevant here?",
    ])
    turn3_content = ex.get("bridge_question") or (
        "Based on everything we've discussed, what is the most significant takeaway?"
    )

    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Answer questions based on the provided documents."},
        {"role": "user",      "content": turn1_content},
        {"role": "assistant", "content": f"Based on the documents, {ex.get('answer', 'the answer is in the context')}."},
        {"role": "user",      "content": turn2_content},
        {"role": "assistant", "content": "Let me provide more detail on that point."},
        {"role": "user",      "content": turn3_content},
    ]

    # Inject the redundant doc again in turn 3 message (triggers S1/S4)
    if rng.random() < 0.20 and ex["context_docs"]:
        messages[-1]["content"] += f"\n\nContext reminder:\n{ex['context_docs'][0][:300]}"

    sid = hashlib.sha256(f"rag_{ex['id']}".encode()).hexdigest()[:16]
    return WorkloadSample(
        sample_id=f"rag_{sid}",
        workload="rag",
        messages=messages,
        gold_answer=ex.get("answer"),
        gold_entities=ex.get("entities", []),
        metadata={"original_id": ex["id"]},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_rag_samples(n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """Load n RAG samples (HotpotQA or synthetic fallback)."""
    raw = _try_hf_hotpotqa(n) or _load_bundled(n) or _generate_synthetic(n, seed)
    raw = raw[:n]
    return [_example_to_sample(ex, i, seed) for i, ex in enumerate(raw)]
