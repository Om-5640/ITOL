"""
Anthropic adapter — §8.3.

Anthropic's /v1/messages format differs from OpenAI:
  - system is a top-level string field, not a role in messages
  - roles are "user" | "assistant" only (no "system" in messages)
  - content may be a string or list of {type, text} blocks
"""
from __future__ import annotations

import uuid
from typing import Any

from itol.adapters.base import ProviderAdapter
from itol.icr import (
    ICR, ICRResponse, Message, ContentBlock, ContentType, ToolDef, UsageStats,
)

_PASSTHROUGH_PARAMS = ("temperature", "max_tokens", "stop_sequences", "top_p", "stream")


class AnthropicAdapter(ProviderAdapter):
    _name = "anthropic"
    base_url = "https://api.anthropic.com/v1"

    @property
    def name(self) -> str:
        return self._name

    def to_icr(self, body: dict[str, Any], *, tenant_id: str = "default") -> ICR:
        # System: top-level string or list of content blocks
        system_blocks: list[ContentBlock] = []
        raw_system = body.get("system")
        if isinstance(raw_system, str) and raw_system:
            system_blocks = [ContentBlock.text(raw_system)]
        elif isinstance(raw_system, list):
            for blk in raw_system:
                if blk.get("type") == "text" and blk.get("text"):
                    system_blocks.append(ContentBlock.text(blk["text"]))

        messages: list[Message] = []
        for m in body.get("messages", []):
            role = m.get("role", "user")
            content = m.get("content") or ""
            if isinstance(content, str):
                blocks = [ContentBlock.text(content)] if content else []
            elif isinstance(content, list):
                blocks = [
                    ContentBlock.text(blk["text"])
                    for blk in content
                    if blk.get("type") == "text" and blk.get("text")
                ]
            else:
                blocks = []
            messages.append(Message(role=role, content=blocks))

        tools: list[ToolDef] = []
        for t in body.get("tools", []):
            tools.append(ToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("input_schema", {}),
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

    def from_icr(self, icr: ICR) -> dict[str, Any]:
        body: dict[str, Any] = {"model": icr.model}

        if icr.system:
            sys_text = "\n".join(b.text for b in icr.system if b.text)
            body["system"] = sys_text

        body["messages"] = [
            {
                "role": msg.role,
                "content": "\n".join(
                    b.text for b in msg.content
                    if b.type is ContentType.TEXT and b.text
                ),
            }
            for msg in icr.messages
        ]

        body.update(icr.params)

        if icr.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in icr.tools
            ]
        return body

    def capabilities(self) -> dict[str, Any]:
        return {
            "native_prompt_cache": "prefix",
            "cache_read_discount": 0.10,
            "max_context": 200_000,
        }

    def parse_response(self, raw: dict[str, Any], request_id: str) -> ICRResponse:
        content_blocks: list[ContentBlock] = []
        for blk in raw.get("content", []):
            if blk.get("type") == "text" and blk.get("text"):
                content_blocks.append(ContentBlock.text(blk["text"]))
        usage = raw.get("usage") or {}
        return ICRResponse(
            request_id=request_id,
            provider=self.name,
            model=raw.get("model", ""),
            content=content_blocks,
            usage=UsageStats(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
            ),
            finish_reason=raw.get("stop_reason"),
            raw=raw,
        )
