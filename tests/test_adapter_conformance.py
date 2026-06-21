"""
§8.3 Adapter conformance — round-trip identity for all 6 adapters.

For every canned ICR and every adapter:
    body  = adapter.from_icr(icr)
    icr2  = adapter.to_icr(body, tenant_id=icr.tenant_id)

Assert:
    - message roles preserved (in order)
    - message text content preserved (in order)
    - system text preserved
    - model preserved

25 canned ICRs × 6 adapters = 150 parametrised assertions.
"""
from __future__ import annotations

import pytest

from itol.icr import ICR, Message, ContentBlock
from itol.adapters.openai_ import OpenAIAdapter
from itol.adapters.anthropic_ import AnthropicAdapter
from itol.adapters.mistral import MistralAdapter
from itol.adapters.groq import GroqAdapter
from itol.adapters.ollama import OllamaAdapter
from itol.adapters.cohere import CohereAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(
    model: str,
    msgs: list[tuple[str, str]],
    system: str = "",
) -> ICR:
    """Convenience factory for text-only ICRs."""
    return ICR.create(
        provider="test",
        model=model,
        messages=[Message.user(t) if r == "user" else Message.assistant(t) for r, t in msgs],
        system=[ContentBlock.text(system)] if system else None,
        tenant_id="default",
    )


def _system_text(icr: ICR) -> str:
    return "\n".join(b.text for b in icr.system if b.text)


def _msg_texts(icr: ICR) -> list[str]:
    return [m.text_content() for m in icr.messages]


def _msg_roles(icr: ICR) -> list[str]:
    return [m.role for m in icr.messages]


# ---------------------------------------------------------------------------
# 25 canned ICRs
# ---------------------------------------------------------------------------

CANNED_ICRS: list[ICR] = [
    # 1-5: single user message
    _make("gpt-4o",       [("user", "Hello, world!")]),
    _make("gpt-4o-mini",  [("user", "What is 2 + 2?")]),
    _make("command-r",    [("user", "Explain black holes in one paragraph.")]),
    _make("mistral-large",[("user", "List the capitals of all EU countries.")]),
    _make("llama3",       [("user", "Write a haiku about autumn.")]),

    # 6-10: system + single user message
    _make("gpt-4o",    [("user", "Summarise this doc.")], system="You are a concise summariser."),
    _make("gpt-4o",    [("user", "Translate to French.")], system="Translate everything the user says."),
    _make("command-r", [("user", "Debug this code.")],    system="You are an expert Python debugger."),
    _make("mistral-large", [("user", "Rate this essay.")], system="You give essay feedback scores 1-10."),
    _make("llama3",    [("user", "Is this sentence grammatical?")], system="You are a grammar checker."),

    # 11-15: 2-turn conversation (user, assistant, user)
    _make("gpt-4o", [("user", "What is photosynthesis?"), ("assistant", "It converts light to sugar."), ("user", "Give more detail.")]),
    _make("gpt-4o", [("user", "Tell me a joke."), ("assistant", "Why did the chicken cross?"), ("user", "I don't know, why?")]),
    _make("command-r", [("user", "Who wrote Hamlet?"), ("assistant", "William Shakespeare."), ("user", "When?")]),
    _make("mistral-large", [("user", "Start a story."), ("assistant", "Once upon a time..."), ("user", "Continue it.")]),
    _make("llama3", [("user", "What's the capital of Japan?"), ("assistant", "Tokyo."), ("user", "And of South Korea?")]),

    # 16-20: system + 2-turn conversation
    _make("gpt-4o", [("user", "How do I sort a list?"), ("assistant", "Use list.sort()."), ("user", "Show an example.")], system="You are a Python tutor."),
    _make("gpt-4o", [("user", "What time is it?"), ("assistant", "I cannot access real-time data."), ("user", "Estimate then.")], system="You answer helpfully."),
    _make("command-r", [("user", "Plan my day."), ("assistant", "Morning: exercise..."), ("user", "Add lunch.")], system="You are a productivity coach."),
    _make("mistral-large", [("user", "Write a tweet."), ("assistant", "Just shipped a new feature!"), ("user", "Make it longer.")], system="You write social media content."),
    _make("llama3", [("user", "Define entropy."), ("assistant", "Measure of disorder."), ("user", "Give an example.")], system="You are a physics professor."),

    # 21-25: 3-turn conversation (user, assistant, user, assistant, user)
    _make("gpt-4o", [("user", "A"), ("assistant", "B"), ("user", "C"), ("assistant", "D"), ("user", "E")]),
    _make("gpt-4o", [("user", "Start."), ("assistant", "OK."), ("user", "Next."), ("assistant", "Done."), ("user", "Thanks.")]),
    _make("command-r", [("user", "Q1?"), ("assistant", "A1."), ("user", "Q2?"), ("assistant", "A2."), ("user", "Q3?")]),
    _make("mistral-large", [("user", "Hello."), ("assistant", "Hi!"), ("user", "How are you?"), ("assistant", "Fine."), ("user", "Great.")]),
    _make("llama3", [("user", "1"), ("assistant", "2"), ("user", "3"), ("assistant", "4"), ("user", "5")]),
]

assert len(CANNED_ICRS) == 25, f"Expected 25 ICRs, got {len(CANNED_ICRS)}"


# ---------------------------------------------------------------------------
# Adapters under test
# ---------------------------------------------------------------------------

ADAPTERS = [
    OpenAIAdapter(),
    AnthropicAdapter(),
    MistralAdapter(),
    GroqAdapter(),
    OllamaAdapter(),
    CohereAdapter(),
]

ADAPTER_IDS = [a.name for a in ADAPTERS]


# ---------------------------------------------------------------------------
# Parametrised round-trip tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
@pytest.mark.parametrize("icr", CANNED_ICRS, ids=[f"icr{i+1}" for i in range(25)])
def test_round_trip_message_roles(icr: ICR, adapter) -> None:
    body = adapter.from_icr(icr)
    icr2 = adapter.to_icr(body, tenant_id=icr.tenant_id)
    assert _msg_roles(icr2) == _msg_roles(icr), (
        f"{adapter.name}: role mismatch\n  original: {_msg_roles(icr)}\n  got: {_msg_roles(icr2)}"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
@pytest.mark.parametrize("icr", CANNED_ICRS, ids=[f"icr{i+1}" for i in range(25)])
def test_round_trip_message_text(icr: ICR, adapter) -> None:
    body = adapter.from_icr(icr)
    icr2 = adapter.to_icr(body, tenant_id=icr.tenant_id)
    assert _msg_texts(icr2) == _msg_texts(icr), (
        f"{adapter.name}: text mismatch\n  original: {_msg_texts(icr)}\n  got: {_msg_texts(icr2)}"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
@pytest.mark.parametrize("icr", CANNED_ICRS, ids=[f"icr{i+1}" for i in range(25)])
def test_round_trip_system_text(icr: ICR, adapter) -> None:
    body = adapter.from_icr(icr)
    icr2 = adapter.to_icr(body, tenant_id=icr.tenant_id)
    assert _system_text(icr2) == _system_text(icr), (
        f"{adapter.name}: system mismatch\n  original: {_system_text(icr)!r}\n  got: {_system_text(icr2)!r}"
    )


@pytest.mark.parametrize("adapter", ADAPTERS, ids=ADAPTER_IDS)
@pytest.mark.parametrize("icr", CANNED_ICRS, ids=[f"icr{i+1}" for i in range(25)])
def test_round_trip_model(icr: ICR, adapter) -> None:
    body = adapter.from_icr(icr)
    icr2 = adapter.to_icr(body, tenant_id=icr.tenant_id)
    assert icr2.model == icr.model, (
        f"{adapter.name}: model mismatch — original {icr.model!r}, got {icr2.model!r}"
    )


# ---------------------------------------------------------------------------
# Cohere-specific: role mapping
# ---------------------------------------------------------------------------

def test_cohere_role_mapping_chatbot() -> None:
    """CHATBOT role in Cohere body → 'assistant' in ICR."""
    adapter = CohereAdapter()
    body = {
        "model": "command-r",
        "message": "What next?",
        "chat_history": [
            {"role": "USER", "message": "Hello"},
            {"role": "CHATBOT", "message": "Hi there!"},
        ],
    }
    icr = adapter.to_icr(body)
    assert icr.messages[0].role == "user"
    assert icr.messages[1].role == "assistant"
    assert icr.messages[2].role == "user"


def test_cohere_role_mapping_roundtrip() -> None:
    """ICR assistant role survives from_icr → chat_history → to_icr."""
    adapter = CohereAdapter()
    icr = _make("command-r", [("user", "Hi"), ("assistant", "Hello"), ("user", "Bye")])
    body = adapter.from_icr(icr)
    history = body["chat_history"]
    assert history[0]["role"] == "USER"
    assert history[1]["role"] == "CHATBOT"
    assert body["message"] == "Bye"
    icr2 = adapter.to_icr(body)
    assert _msg_roles(icr2) == ["user", "assistant", "user"]
    assert _msg_texts(icr2) == ["Hi", "Hello", "Bye"]


def test_cohere_preamble_roundtrip() -> None:
    """System text → preamble → system text round-trip."""
    adapter = CohereAdapter()
    icr = _make("command-r", [("user", "Go")], system="Be concise.")
    body = adapter.from_icr(icr)
    assert body["preamble"] == "Be concise."
    icr2 = adapter.to_icr(body)
    assert _system_text(icr2) == "Be concise."


# ---------------------------------------------------------------------------
# Adapter line-count assertions (§8.3 <30-min proof)
# ---------------------------------------------------------------------------

def test_mistral_adapter_is_concise() -> None:
    """Mistral adapter ≤ 20 source lines (proves <30-min new-provider claim)."""
    import inspect
    from itol.adapters.mistral import MistralAdapter
    src = inspect.getsource(MistralAdapter)
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(lines) <= 20, f"MistralAdapter has {len(lines)} non-blank/non-comment lines"


def test_groq_adapter_is_concise() -> None:
    import inspect
    from itol.adapters.groq import GroqAdapter
    src = inspect.getsource(GroqAdapter)
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(lines) <= 20, f"GroqAdapter has {len(lines)} non-blank/non-comment lines"


def test_ollama_adapter_is_concise() -> None:
    import inspect
    from itol.adapters.ollama import OllamaAdapter
    src = inspect.getsource(OllamaAdapter)
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert len(lines) <= 20, f"OllamaAdapter has {len(lines)} non-blank/non-comment lines"
