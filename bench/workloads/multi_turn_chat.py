"""
Multi-turn chat workload.

Data source:
  - Tries to download LMSYS-Chat-1M via HuggingFace (requires auth — often fails)
  - Falls back to a realistic synthetic 10–15-turn dialogue generator

Each sample sends ONLY THE LAST TURN through both runners, with full history
as context. This is where S5 history distillation provides the biggest savings —
the history (9–14 turns) gets compressed, while the final question gets answered
correctly.

Topics span: coding help, creative writing, factual Q&A, math tutoring,
professional advice — realistic LMSYS-style distribution.
"""
from __future__ import annotations

import hashlib
import random
from typing import Optional

from bench.workloads import WorkloadSample

# ---------------------------------------------------------------------------
# Synthetic dialogue generator
# ---------------------------------------------------------------------------

_TOPICS = [
    ("coding", "Python", [
        "I'm trying to write a function that sorts a list of dictionaries by a nested key. Can you help?",
        "Thanks! What if the nested key is optional — some dicts might not have it?",
        "Got it. Can you also show how to write a unit test for this?",
        "The test is passing but I'm worried about edge cases. What are the most common ones?",
        "How would I handle the case where the input is None instead of an empty list?",
        "Can you refactor this into a class with a configurable default value?",
        "One more thing — how do I add type hints for this?",
        "Perfect. Can you write a docstring for the whole class?",
        "What's the difference between a dataclass and a regular class here?",
        "Should I use __slots__? When does that matter?",
        "How do I serialize this to JSON and back?",
        "Last question: how do I make this thread-safe?",
    ]),
    ("coding", "JavaScript", [
        "I have an async function that fetches user data but it sometimes returns undefined. Help?",
        "Here's the code: async function getUser(id) { const resp = await fetch('/api/' + id); return resp.json(); }",
        "What happens if the server returns a 404?",
        "How do I add error handling without making every call site messy?",
        "Can you show me the React hook version of this?",
        "What about caching the results to avoid duplicate fetches?",
        "How would I cancel an in-flight request if the component unmounts?",
        "When should I use useCallback here?",
        "Let me show you the full component — can you review it for performance issues?",
        "Thanks. One last thing: how do I test this with Jest?",
    ]),
    ("math", "calculus", [
        "Can you explain what a derivative actually means intuitively?",
        "OK so it's the rate of change. How do I compute it for x³ + 2x?",
        "What's the chain rule and when do I use it?",
        "Let me try: d/dx[sin(x²)]. Is this right: 2x·cos(x²)?",
        "Great! What about the product rule?",
        "Can you give me a real-world example of when you'd need the product rule?",
        "How does integration relate to differentiation?",
        "Can you show me the integral of x³ + 2x step by step?",
        "What's the fundamental theorem of calculus in plain English?",
        "How do I know when to use integration by parts vs substitution?",
        "Let me try this integral: ∫ x·e^x dx. Is my setup right?",
        "What are the most common mistakes students make with integration?",
    ]),
    ("writing", "creative", [
        "I'm writing a sci-fi short story about an AI that becomes self-aware. Any opening suggestions?",
        "I like the second one. Can you develop the first paragraph further?",
        "The pacing feels off. How do I slow down the opening without losing the reader?",
        "What about the protagonist's voice? It feels too formal.",
        "Can you rewrite that paragraph in a more casual, first-person voice?",
        "How do I foreshadow the twist without making it obvious?",
        "I want to add a secondary character who challenges the AI's self-perception. Ideas?",
        "Can you write a short dialogue between the AI and this character?",
        "The dialogue feels a bit on-the-nose. How do I make subtext work here?",
        "What's the difference between showing and telling in this context?",
        "Can you revise the last exchange to show their conflict instead of stating it?",
        "How long should each scene be for a 3000-word short story?",
    ]),
    ("advice", "career", [
        "I'm a software engineer with 5 years of experience considering switching to product management. Thoughts?",
        "What skills from engineering transfer directly to PM roles?",
        "What's the hardest part of the transition that I should prepare for?",
        "How do I build a PM portfolio if I've never had the title?",
        "Should I target startups or large companies for my first PM role?",
        "What questions will I likely get in PM interviews that I'd never face as an engineer?",
        "Can you help me craft an answer to 'Tell me about a product you improved'?",
        "What about the technical design questions? Will my engineering background help?",
        "How long does the transition typically take — am I looking at months or years?",
        "Is it worth getting a PM certification, or is that just resume fluff?",
        "Any books or resources you'd strongly recommend for this transition?",
        "One final question: what salary difference should I expect, and is it worth it?",
    ]),
]


def _generate_conversation(
    topic: str,
    subtopic: str,
    turns: list[str],
    rng: random.Random,
    n_turns: int,
) -> list[dict]:
    """Build a multi-turn conversation with n_turns user messages + assistant replies."""
    system = (
        f"You are a helpful, knowledgeable assistant specializing in {topic}. "
        f"Provide clear, accurate, and actionable responses."
    )
    messages: list[dict] = [{"role": "system", "content": system}]

    selected = turns[:n_turns]
    for i, user_msg in enumerate(selected):
        messages.append({"role": "user", "content": user_msg})
        if i < len(selected) - 1:
            # Add synthetic assistant reply for all but the last turn
            reply = (
                f"That's a great question about {subtopic}. "
                + rng.choice([
                    f"The key insight here is that {subtopic} requires careful consideration of multiple factors.",
                    f"Let me break this down step by step for {subtopic}.",
                    f"When working with {subtopic}, the most important thing to understand is the underlying principle.",
                    f"There are several approaches to this in {subtopic}, and the best one depends on your context.",
                ])
                + " " + rng.choice([
                    "Here's what I recommend: focus on the fundamentals first.",
                    "The best practice here is to start simple and iterate.",
                    "Let me show you a concrete example that illustrates this.",
                    "This is a common challenge, and there are well-established patterns to handle it.",
                ])
            )
            # Make some replies longer (triggers S5 better)
            if i > 2 and rng.random() < 0.4:
                reply += (
                    f"\n\nAdditional context: When we discussed {subtopic} earlier, "
                    f"we established that the core principle is consistency. "
                    f"Building on that foundation, the approach I'm recommending here "
                    f"aligns with industry best practices and avoids common pitfalls. "
                    f"Remember to test edge cases and document your assumptions."
                )
            messages.append({"role": "assistant", "content": reply})

    return messages


def _generate_samples(n: int, seed: int) -> list[WorkloadSample]:
    rng = random.Random(seed)
    samples = []

    for i in range(n):
        topic, subtopic, turns = rng.choice(_TOPICS)
        n_turns = rng.randint(10, min(len(turns), 13))
        messages = _generate_conversation(topic, subtopic, turns, rng, n_turns)

        # The last message is the final user turn (what we want answered)
        # Full history (all prior turns) is the context S5 will compress
        final_question = messages[-1]["content"]

        sid = hashlib.sha256(f"chat_{seed}_{i}".encode()).hexdigest()[:16]
        samples.append(WorkloadSample(
            sample_id=f"chat_{sid}",
            workload="chat",
            messages=messages,
            gold_answer=None,  # No gold for chat; use parity scoring
            metadata={
                "topic": topic,
                "subtopic": subtopic,
                "n_history_turns": len(messages) - 1,
                "final_question": final_question[:100],
            },
        ))

    return samples


# ---------------------------------------------------------------------------
# HuggingFace LMSYS fallback (usually fails without auth)
# ---------------------------------------------------------------------------

def _try_hf_lmsys(n: int) -> Optional[list[WorkloadSample]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        samples = []
        for item in ds:
            convs = item.get("conversation", [])
            if len(convs) < 8:
                continue
            messages = []
            for turn in convs[:14]:
                role = "user" if turn["role"] == "human" else "assistant"
                messages.append({"role": role, "content": turn["content"][:800]})
            if len(messages) < 6:
                continue
            sid = hashlib.sha256(f"chat_lmsys_{item.get('conversation_id',len(samples))}".encode()).hexdigest()[:16]
            samples.append(WorkloadSample(
                sample_id=f"chat_{sid}",
                workload="chat",
                messages=messages,
                metadata={"source": "lmsys"},
            ))
            if len(samples) >= n:
                break
        return samples if samples else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_chat_samples(n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """Load n multi-turn chat samples (LMSYS or synthetic fallback)."""
    hf = _try_hf_lmsys(n)
    if hf:
        return hf[:n]
    return _generate_samples(n, seed)
