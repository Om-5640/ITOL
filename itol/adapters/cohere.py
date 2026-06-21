"""
Cohere adapter — §8.3.

Cohere's /v1/chat API uses a distinct request shape:
  - `message`      : str  — the current user turn
  - `chat_history` : list[{role: "USER"|"CHATBOT"|"SYSTEM", message: str}]
  - `preamble`     : str  — system prompt (optional)

Role mapping (ICR ↔ Cohere):
  ICR "user"      ↔ Cohere "USER"
  ICR "assistant" ↔ Cohere "CHATBOT"
  ICR "system"    ↔ Cohere "SYSTEM"  (rare; treated as system block in ICR)

This is the "worked non-trivial example" proving non-OpenAI providers fit in
<30 min of additional work beyond the base adapter pattern (§8.3).
"""
from __future__ import annotations

import uuid
from typing import Any

from itol.adapters.base import ProviderAdapter
from itol.icr import (
    ICR, ICRResponse, Message, ContentBlock, ContentType, ToolDef, UsageStats,
)

_ICR_TO_COHERE: dict[str, str] = {
    "user": "USER",
    "assistant": "CHATBOT",
    "system": "SYSTEM",
}
_COHERE_TO_ICR: dict[str, str] = {v: k for k, v in _ICR_TO_COHERE.items()}

_PASSTHROUGH_PARAMS = ("temperature", "max_tokens", "p", "k", "stop_sequences")


class CohereAdapter(ProviderAdapter):
    _name = "cohere"
    base_url = "https://api.cohere.ai/v1"

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # to_icr  (Cohere body → ICR)
    # ------------------------------------------------------------------

    def to_icr(self, body: dict[str, Any], *, tenant_id: str = "default") -> ICR:
        system_blocks: list[ContentBlock] = []
        preamble = body.get("preamble") or ""
        if preamble:
            system_blocks = [ContentBlock.text(preamble)]

        messages: list[Message] = []
        for entry in body.get("chat_history", []):
            cohere_role = (entry.get("role") or "USER").upper()
            text = entry.get("message") or ""
            icr_role = _COHERE_TO_ICR.get(cohere_role, "user")
            if icr_role == "system":
                # SYSTEM entries in chat_history → extend system blocks
                if text:
                    system_blocks.append(ContentBlock.text(text))
            else:
                messages.append(Message(
                    role=icr_role,  # type: ignore[arg-type]
                    content=[ContentBlock.text(text)] if text else [],
                ))

        # The current user message is always the last user turn
        current_msg = body.get("message") or ""
        if current_msg:
            messages.append(Message(role="user", content=[ContentBlock.text(current_msg)]))

        params = {k: body[k] for k in _PASSTHROUGH_PARAMS if k in body}

        return ICR(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            provider=self.name,
            model=body.get("model", "command-r"),
            system=system_blocks,
            messages=messages,
            tools=[],   # Cohere tool support is separate from basic chat round-trip
            params=params,
            raw=body,
        )

    # ------------------------------------------------------------------
    # from_icr  (ICR → Cohere body)
    # ------------------------------------------------------------------

    def from_icr(self, icr: ICR) -> dict[str, Any]:
        body: dict[str, Any] = {"model": icr.model}

        if icr.system:
            preamble = "\n".join(b.text for b in icr.system if b.text)
            if preamble:
                body["preamble"] = preamble

        # All messages except the last user turn go into chat_history
        chat_history: list[dict[str, str]] = []
        messages = icr.messages

        if not messages:
            body["message"] = ""
            body["chat_history"] = []
            body.update(icr.params)
            return body

        # Last message must be user turn (Cohere requirement)
        last = messages[-1]
        prior = messages[:-1]

        for msg in prior:
            cohere_role = _ICR_TO_COHERE.get(msg.role, "USER")
            text = "\n".join(
                b.text for b in msg.content
                if b.type is ContentType.TEXT and b.text
            )
            chat_history.append({"role": cohere_role, "message": text})

        last_text = "\n".join(
            b.text for b in last.content
            if b.type is ContentType.TEXT and b.text
        )

        body["message"] = last_text
        body["chat_history"] = chat_history
        body.update(icr.params)
        return body

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "none",
            "cache_read_discount": 0.0,
            "max_context": 128_000,
        }

    def parse_response(self, raw: dict[str, Any], request_id: str) -> ICRResponse:
        text = raw.get("text") or ""
        meta = raw.get("meta") or {}
        tokens = meta.get("tokens") or {}
        return ICRResponse(
            request_id=request_id,
            provider=self.name,
            model=raw.get("model", ""),
            content=[ContentBlock.text(text)] if text else [],
            usage=UsageStats(
                input_tokens=tokens.get("input_tokens", 0),
                output_tokens=tokens.get("output_tokens", 0),
            ),
            finish_reason=raw.get("finish_reason"),
            raw=raw,
        )
