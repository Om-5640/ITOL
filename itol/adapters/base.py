"""
ProviderAdapter ABC — §8.3.

Every concrete adapter must implement to_icr/from_icr so any provider fits the
ITOL pipeline with no pipeline-core changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from itol.icr import ICR, ICRResponse


class ProviderAdapter(ABC):
    """Abstract base for all provider adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider slug, e.g. 'openai', 'mistral', 'cohere'."""

    @abstractmethod
    def to_icr(self, body: dict[str, Any], *, tenant_id: str = "default") -> ICR:
        """Translate a provider-native request body into an ICR."""

    @abstractmethod
    def from_icr(self, icr: ICR) -> dict[str, Any]:
        """Translate an (optionally mutated) ICR back to a provider-native body."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """
        Return provider capability metadata consumed by the pipeline.

        Expected keys (all optional — pipeline uses safe defaults when absent):
            native_prompt_cache : "prefix" | "none"
            cache_read_discount : float   (fraction, e.g. 0.1 = 10 % of normal price)
            max_context         : int     (context window tokens)
        """

    @abstractmethod
    def parse_response(self, raw: dict[str, Any], request_id: str) -> ICRResponse:
        """Parse a provider-native response dict into an ICRResponse."""
