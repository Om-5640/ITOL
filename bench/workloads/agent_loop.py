"""
Agent tool-loop workload.

Produces synthetic agent trajectories of 4–8 turns:
  user → assistant (tool_call) → tool_result → assistant (tool_call) → ... → final

Design:
  - 5 tool schemas per trajectory; not all called (S6 tool-schema pruning fires)
  - ~30% of tool results superseded by later turns (S6 tool-result expiry fires)
  - Final user message requests a summary of findings (triggers S5 history distillation)
  - Builds on calibration/synth_agent.py themes but creates full multi-turn trajectories
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from bench.workloads import WorkloadSample

# ---------------------------------------------------------------------------
# Tool schemas (5 per trajectory, not all used)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "name": "search_web",
        "description": "Search the web for up-to-date information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
            },
            "required": ["location"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "lookup_entity",
        "description": "Look up structured information about a named entity.",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "summarize_document",
        "description": "Summarize a long document to key points.",
        "parameters": {
            "type": "object",
            "properties": {
                "document": {"type": "string"},
                "max_words": {"type": "integer", "default": 100},
            },
            "required": ["document"],
        },
    },
]

# ---------------------------------------------------------------------------
# Trajectory templates
# ---------------------------------------------------------------------------

_TASKS = [
    ("research", "Research the latest developments in {topic} and provide a 3-point summary.",
     ["quantum computing", "renewable energy", "large language models", "CRISPR gene editing",
      "blockchain scalability", "autonomous vehicles", "protein folding", "fusion energy"]),
    ("analysis", "Analyze the performance of {company} over the last quarter and identify key trends.",
     ["Apple", "Google", "Microsoft", "Tesla", "Amazon", "Meta", "Nvidia", "OpenAI"]),
    ("planning", "Create a step-by-step plan to {goal} given the constraints: {constraint}.",
     [("reduce costs by 20%", "no layoffs"), ("launch a new product", "6-month timeline"),
      ("expand to a new market", "limited budget"), ("improve team productivity", "remote-first")]),
    ("calculation", "Calculate the total cost of {scenario} and break down the components.",
     ["a 5-year software subscription at $99/month with 15% annual price increases",
      "hiring 3 engineers at $120K salary with 30% benefits overhead for 2 years",
      "building a data center with $2M CapEx and $400K annual OpEx over 10 years"]),
]


def _make_tool_call(tool_name: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{hashlib.md5(json.dumps(args).encode()).hexdigest()[:8]}",
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(args)},
        }],
    }


def _make_tool_result(call_id: str, tool_name: str, content: str, superseded: bool = False) -> dict:
    result_content = content
    if superseded:
        result_content = f"[OUTDATED - superseded by later search] {content}"
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": result_content,
    }


def _generate_trajectory(task_desc: str, topic: str, rng: random.Random,
                          n_turns: int = 5) -> tuple[list[dict], list[str]]:
    """Build one agent trajectory. Returns (messages, key_entities)."""
    entities = [topic] + rng.sample(
        ["finding_1", "finding_2", "metric_A", "metric_B", "conclusion"], 3
    )

    system = (
        "You are an efficient research agent. Use tools to gather information, "
        "then synthesize findings into a comprehensive response. "
        "Be systematic and cite your sources."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": f"{task_desc}\n\nTopic: {topic}"})

    # Turn 1: initial search
    call_id_1 = f"call_{rng.randint(10000, 99999)}"
    messages.append({
        "role": "assistant",
        "content": f"I'll research {topic} using available tools.",
        "tool_calls": [{
            "id": call_id_1, "type": "function",
            "function": {"name": "search_web", "arguments": json.dumps({"query": f"{topic} latest developments"})},
        }],
    })
    # Superseded result (30% chance)
    superseded = rng.random() < 0.30
    result_content = (
        f"Found 5 results about {topic}. Key finding: {topic} has shown "
        f"significant advances in Q3 2026, with improvements in efficiency "
        f"and adoption rates reaching 45% in enterprise segments."
    )
    messages.append({
        "role": "tool", "tool_call_id": call_id_1, "name": "search_web",
        "content": f"[OUTDATED] {result_content}" if superseded else result_content,
    })

    # Turn 2: entity lookup
    call_id_2 = f"call_{rng.randint(10000, 99999)}"
    messages.append({
        "role": "assistant",
        "content": f"Let me get more structured information about {topic}.",
        "tool_calls": [{
            "id": call_id_2, "type": "function",
            "function": {"name": "lookup_entity", "arguments": json.dumps({
                "entity": topic, "fields": ["summary", "key_metrics", "recent_events"]
            })},
        }],
    })
    messages.append({
        "role": "tool", "tool_call_id": call_id_2, "name": "lookup_entity",
        "content": json.dumps({
            "entity": topic,
            "summary": f"{topic} is a rapidly evolving domain with broad applications.",
            "key_metrics": {"growth_rate": "23% YoY", "market_size": "$4.2B", "key_players": 12},
            "recent_events": [f"Major {topic} breakthrough announced in June 2026"],
        }),
    })

    # Turn 3: refinement search (supersedes turn 1 if that was marked outdated)
    if superseded and n_turns >= 4:
        call_id_3 = f"call_{rng.randint(10000, 99999)}"
        messages.append({
            "role": "assistant",
            "content": "The earlier search seems outdated. Let me search for more recent information.",
            "tool_calls": [{
                "id": call_id_3, "type": "function",
                "function": {"name": "search_web", "arguments": json.dumps({
                    "query": f"{topic} 2026 recent news", "max_results": 3
                })},
            }],
        })
        messages.append({
            "role": "tool", "tool_call_id": call_id_3, "name": "search_web",
            "content": (
                f"Recent results (2026): {topic} announced a partnership with 3 major "
                f"enterprises. Adoption up 60% since Q1. New regulatory framework "
                f"expected by Q4 2026."
            ),
        })

    # Final user message: request synthesis
    messages.append({
        "role": "user",
        "content": (
            f"Based on your research, please provide a comprehensive 3-point summary "
            f"of the key findings about {topic}, including the most important metrics "
            f"and what they mean for decision-makers."
        ),
    })

    return messages, entities


def _generate_samples(n: int, seed: int) -> list[WorkloadSample]:
    rng = random.Random(seed)
    samples = []

    for i in range(n):
        # Pick task type and topic
        task_type, task_tmpl, topics_pool = rng.choice(_TASKS)
        topic = rng.choice(topics_pool)
        if isinstance(topic, tuple):
            goal, constraint = topic
            task_desc = task_tmpl.format(goal=goal, constraint=constraint, company="", topic="")
            topic_label = f"{goal}"
        else:
            task_desc = task_tmpl.format(topic=topic, company=topic, goal="", constraint="")
            topic_label = topic

        n_turns = rng.randint(4, 8)
        messages, entities = _generate_trajectory(task_desc, topic_label, rng, n_turns)

        sid = hashlib.sha256(f"agent_{seed}_{i}".encode()).hexdigest()[:16]
        samples.append(WorkloadSample(
            sample_id=f"agent_{sid}",
            workload="agent",
            messages=messages,
            gold_entities=entities,
            metadata={
                "task_type": task_type,
                "topic": topic_label,
                "n_turns": len([m for m in messages if m["role"] == "user"]),
                "has_superseded_tools": any(
                    "[OUTDATED]" in str(m.get("content", "")) for m in messages
                ),
            },
        ))

    return samples


def load_agent_samples(n: int = 150, seed: int = 42) -> list[WorkloadSample]:
    """Generate n synthetic agent trajectory samples."""
    return _generate_samples(n, seed)
