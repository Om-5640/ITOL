"""
calibration/synth_agent.py — synthetic data generation for calibration.

Generates deterministic (no-LLM) synthetic corpus pairs covering all 8
request classes.  Each pair has: {request_class, input, reference, compressed}.

The "compressed" field is a 50%-token truncation of the input (deterministic
stand-in until real model compression is available).
"""

from __future__ import annotations

import random
from typing import Any

_SEED = 42

# ---------------------------------------------------------------------------
# Templates per class
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[dict]] = {
    "EXTRACTION": [
        {
            "input": "Invoice #{inv} dated {date} for ${amount}. Contact {email} for queries.",
            "reference": "{inv}, {date}, ${amount}, {email}",
        },
        {
            "input": "Meeting on {date} at {time}. Attendees: {name1}, {name2}. Location: {place}.",
            "reference": "{date}, {time}, {name1}, {name2}, {place}",
        },
    ],
    "REASONING": [
        {
            "input": "Alice is {rel1} than Bob. Bob is {rel1} than Carol. Who is {rel2}?",
            "reference": "Alice",
        },
        {
            "input": "All {category} are {property}. {subject} is a {category}. Is {subject} {property}?",
            "reference": "Yes",
        },
    ],
    "SUMMARIZATION": [
        {
            "input": (
                "The quarterly report for Q{q} showed revenue of ${rev}M, up {pct}% YoY. "
                "Operating expenses were ${opex}M. Net income reached ${ni}M. "
                "The company attributed growth to its {segment} division."
            ),
            "reference": "Revenue ${rev}M (+{pct}% YoY), net income ${ni}M, growth from {segment}.",
        },
        {
            "input": (
                "A study of {n} patients found {drug} reduced symptoms by {effect}% vs placebo "
                "(p<0.05). Side effects were mild. Treatment is now recommended first-line."
            ),
            "reference": "{drug} reduced symptoms {effect}%; recommended first-line.",
        },
    ],
    "GENERATION_FACTUAL": [
        {
            "input": "Explain {concept} in one sentence for a general audience.",
            "reference": "{concept} is a fundamental principle in {field}.",
        },
        {
            "input": "What is the capital of {country}?",
            "reference": "The capital of {country} is {capital}.",
        },
    ],
    "GENERATION_CREATIVE": [
        {
            "input": "Write a two-line poem about {subject}.",
            "reference": "The {subject} shines bright, filling the world with {adjective} light.",
        },
        {
            "input": "Give a creative name for a {product} that evokes {mood}.",
            "reference": "{mood_cap}spark {product_cap}",
        },
    ],
    "CLASSIFICATION_SHORT": [
        {
            "input": "Classify the sentiment of: '{sentence}'",
            "reference": "{sentiment}",
        },
        {
            "input": "Is this spam: '{text}'?",
            "reference": "{label}",
        },
    ],
    "AGENT_TOOL_LOOP": [
        {
            "input": "Fetch the current price of {ticker} and compare to its {period} average.",
            "reference": "{ticker} current: ${price}, {period} avg: ${avg}.",
        },
        {
            "input": "Run this command: {cmd}. Report the output.",
            "reference": "Command output: {output}",
        },
    ],
    "CHAT_OPEN": [
        {
            "input": "What's a fun fact about {topic}?",
            "reference": "{topic} has a fascinating property: {fact}.",
        },
        {
            "input": "Can you recommend a {genre} book?",
            "reference": "For {genre}, I'd suggest \"{title}\" by {author}.",
        },
    ],
}

# ---------------------------------------------------------------------------
# Fill data
# ---------------------------------------------------------------------------

_FILL: dict[str, list[Any]] = {
    "inv": ["INV-001", "INV-002", "INV-003", "INV-004", "INV-005"],
    "date": ["2024-01-15", "2024-03-22", "2024-07-04", "2024-09-11", "2024-12-01"],
    "amount": ["1200", "4500", "800", "3200", "9999"],
    "email": ["alice@corp.com", "bob@example.io", "carol@firm.net"],
    "time": ["09:00", "14:30", "11:00"],
    "name1": ["Alice", "Bob", "Carol", "Dave"],
    "name2": ["Eve", "Frank", "Grace"],
    "place": ["Room 4B", "HQ Lobby", "Zoom"],
    "rel1": ["taller", "faster", "smarter"],
    "rel2": ["tallest", "fastest", "smartest"],
    "category": ["mammals", "prime numbers", "metals"],
    "property": ["warm-blooded", "odd above 2", "conductors"],
    "subject": ["whales", "7", "copper"],
    "q": ["1", "2", "3", "4"],
    "rev": ["120", "340", "80", "500"],
    "pct": ["12", "22", "5", "18"],
    "opex": ["40", "90", "30", "150"],
    "ni": ["30", "70", "20", "120"],
    "segment": ["cloud", "enterprise", "mobile", "AI"],
    "n": ["400", "2400", "850", "150"],
    "drug": ["DrugA", "Compound-X", "MedZ"],
    "effect": ["43", "28", "55"],
    "concept": ["entropy", "machine learning", "recursion"],
    "field": ["thermodynamics", "computer science", "mathematics"],
    "country": ["France", "Japan", "Brazil"],
    "capital": ["Paris", "Tokyo", "Brasília"],
    "adjective": ["golden", "serene", "bright"],
    "mood": ["calm", "excitement", "wonder"],
    "mood_cap": ["Calm", "Spark", "Wonder"],
    "product": ["notebook", "coffee brand", "app"],
    "product_cap": ["Note", "Brew", "App"],
    "sentence": [
        "Great product, very happy!",
        "Terrible service, never again.",
        "It was okay, nothing special.",
    ],
    "sentiment": ["positive", "negative", "neutral"],
    "text": [
        "Win a free prize now!!!",
        "Invoice attached for review.",
        "Click here to claim your reward!",
    ],
    "label": ["spam", "not spam", "spam"],
    "ticker": ["AAPL", "MSFT", "NVDA"],
    "period": ["52-week", "30-day", "90-day"],
    "price": ["180", "415", "820"],
    "avg": ["162", "390", "750"],
    "cmd": ["ls /tmp", "df -h", "uptime"],
    "output": ["file1.txt  file2.log", "80% used", "up 3 days"],
    "topic": ["octopuses", "the moon", "ancient Rome"],
    "fact": ["they have three hearts", "it has no atmosphere", "it lasted 1000 years"],
    "genre": ["sci-fi", "mystery", "historical fiction"],
    "title": ["Dune", "Gone Girl", "The Name of the Rose"],
    "author": ["Frank Herbert", "Gillian Flynn", "Umberto Eco"],
}


def _fill_template(template: str, rng: random.Random) -> str:
    result = template
    for key, values in _FILL.items():
        if "{" + key + "}" in result:
            result = result.replace("{" + key + "}", rng.choice(values))
    return result


def _compress(text: str) -> str:
    """50% token truncation (deterministic stand-in for real compression)."""
    tokens = text.split()
    half = max(1, len(tokens) // 2)
    return " ".join(tokens[:half]) + " [...]"


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_pairs(
    n_per_class: int = 10,
    seed: int = _SEED,
) -> list[dict]:
    """
    Generate `n_per_class` synthetic corpus pairs for each of the 8 classes.
    Returns list of {"request_class", "input", "reference", "compressed"}.
    """
    rng = random.Random(seed)
    pairs: list[dict] = []

    for req_class, templates in _TEMPLATES.items():
        for i in range(n_per_class):
            tpl = templates[i % len(templates)]
            inp = _fill_template(tpl["input"], rng)
            ref = _fill_template(tpl["reference"], rng)
            pairs.append({
                "request_class": req_class,
                "input": inp,
                "reference": ref,
                "compressed": _compress(inp),
            })

    return pairs
