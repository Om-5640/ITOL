"""
OpenAICompatibleAdapter — §8.3.

Mechanical to_icr/from_icr for any provider that speaks the OpenAI
chat-completions dialect (messages array, system-role message, tools array).
Subclass with just name + base_url + capabilities() override to add a new
OpenAI-compatible provider in <5 lines.
"""
from __future__ import annotations

import uuid
from typing import Any

from itol.adapters.base import ProviderAdapter
from itol.icr import (
    ICR, ICRResponse, Message, ContentBlock, ContentType, ToolDef, UsageStats,
)

# Params forwarded verbatim between ICR.params ↔ provider body.
_PASSTHROUGH_PARAMS = (
    "temperature", "max_tokens", "stop", "top_p", "stream",
    "n", "presence_penalty", "frequency_penalty", "seed", "logprobs",
)


class OpenAICompatibleAdapter(ProviderAdapter):
    """
    Base adapter for any OpenAI-dialect API.

    Subclasses set `name` and optionally `base_url`, then override
    `capabilities()` to declare provider-specific limits.
    """

    _name: str = "openai_compat"
    base_url: str = "https://api.openai.com/v1"

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # to_icr  (provider → ICR)
    # ------------------------------------------------------------------

    def to_icr(self, body: dict[str, Any], *, tenant_id: str = "default") -> ICR:
        system_blocks: list[ContentBlock] = []
        messages: list[Message] = []

        for m in body.get("messages", []):
            role = m.get("role", "user")
            content = m.get("content") or ""
            blocks = self._parse_content(content)

            if role == "system":
                system_blocks.extend(blocks)
            else:
                messages.append(Message(role=role, content=blocks))

        tools: list[ToolDef] = []
        for t in body.get("tools", []):
            fn = t.get("function", t)
            tools.append(ToolDef(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            ))

        params = {k: body[k] for k in _PASSTHROUGH_PARAMS if k in body}

        return ICR(
            request_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            provider=self.name,
            model=body.get("model", ""),
            system=system_blocks,
            messages=messages,
            tools=tools,
            params=params,
            raw=body,
        )

    # ------------------------------------------------------------------
    # from_icr  (ICR → provider)
    # ------------------------------------------------------------------

    def from_icr(self, icr: ICR) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        if icr.system:
            sys_text = "\n".join(b.text for b in icr.system if b.text)
            if sys_text:
                messages.append({"role": "system", "content": sys_text})

        for msg in icr.messages:
            content = self._serialize_content(msg.content)
            messages.append({"role": msg.role, "content": content})

        body: dict[str, Any] = {"model": icr.model, "messages": messages}
        body.update(icr.params)

        if icr.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in icr.tools
            ]
        return body

    # ------------------------------------------------------------------
    # capabilities / parse_response
    # ------------------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "prefix",
            "cache_read_discount": 0.10,
            "max_context": 128_000,
        }

    def parse_response(self, raw: dict[str, Any], request_id: str) -> ICRResponse:
        choice = (raw.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        usage = raw.get("usage") or {}
        return ICRResponse(
            request_id=request_id,
            provider=self.name,
            model=raw.get("model", ""),
            content=[ContentBlock.text(text)] if text else [],
            usage=UsageStats(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_content(content: Any) -> list[ContentBlock]:
        if isinstance(content, str):
            return [ContentBlock.text(content)] if content else []
        if isinstance(content, list):
            blocks: list[ContentBlock] = []
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                t = blk.get("type", "text")
                if t == "text":
                    text = blk.get("text", "")
                    if text:
                        blocks.append(ContentBlock.text(text))
                elif t == "image_url":
                    url = (blk.get("image_url") or {}).get("url", "")
                    blocks.append(ContentBlock(type=ContentType.IMAGE_URL, image_url=url))
            return blocks
        return []

    @staticmethod
    def _serialize_content(blocks: list[ContentBlock]) -> str:
        return "\n".join(b.text for b in blocks if b.type is ContentType.TEXT and b.text)
