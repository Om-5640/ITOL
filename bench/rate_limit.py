"""
Async token-bucket rate limiter per (provider, model).

Enforces both RPS and TPM constraints. On 429: exponential backoff,
max 5 retries. Every rate-limit hit is counted for report disclosure.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    rps: float                 # max requests per second
    tpm: int                   # max tokens per minute
    retry_max: int = 5

    # Internal state
    _req_tokens: float = field(default=0.0, init=False, repr=False)
    _tok_tokens: float = field(default=0.0, init=False, repr=False)
    _last_req_refill: float = field(default_factory=time.monotonic, init=False, repr=False)
    _last_tok_refill: float = field(default_factory=time.monotonic, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _throttle_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._req_tokens = self.rps
        self._tok_tokens = float(self.tpm)
        self._last_req_refill = time.monotonic()
        self._last_tok_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens_estimate: int = 200) -> None:
        """Block until both RPS and TPM buckets have capacity."""
        async with self._lock:
            now = time.monotonic()

            # Refill request bucket (capacity = rps, refill rate = rps/sec)
            elapsed_req = now - self._last_req_refill
            self._req_tokens = min(self.rps, self._req_tokens + elapsed_req * self.rps)
            self._last_req_refill = now

            # Refill token bucket (capacity = tpm, refill rate = tpm/60 per sec)
            elapsed_tok = now - self._last_tok_refill
            self._tok_tokens = min(
                float(self.tpm),
                self._tok_tokens + elapsed_tok * (self.tpm / 60.0)
            )
            self._last_tok_refill = now

            # Wait for request slot
            if self._req_tokens < 1.0:
                wait = (1.0 - self._req_tokens) / self.rps
                self._throttle_count += 1
                logger.debug("RPS throttle: sleeping %.2fs", wait)
                await asyncio.sleep(wait)
                self._req_tokens = 0.0
            else:
                self._req_tokens -= 1.0

            # Wait for token budget
            if self._tok_tokens < tokens_estimate:
                deficit = tokens_estimate - self._tok_tokens
                wait = deficit / (self.tpm / 60.0)
                self._throttle_count += 1
                logger.debug("TPM throttle: sleeping %.2fs for %d tokens", wait, tokens_estimate)
                await asyncio.sleep(wait)
                self._tok_tokens = 0.0
            else:
                self._tok_tokens -= tokens_estimate

    def on_429(self, attempt: int) -> float:
        """Compute exponential backoff delay (seconds) for a 429 response."""
        self._throttle_count += 1
        delay = min(2.0 ** attempt * 2.0, 60.0)
        logger.warning("429 received (attempt %d) — backing off %.1fs", attempt, delay)
        return delay

    @property
    def throttle_count(self) -> int:
        return self._throttle_count


# Per-provider singleton limiters (created lazily per asyncio event loop)
_limiters: dict[str, RateLimiter] = {}


def get_limiter(provider_name: str) -> RateLimiter:
    """Return (or create) the RateLimiter for a provider."""
    if provider_name not in _limiters:
        from bench.config import PROVIDERS
        cfg = PROVIDERS.get(provider_name)
        if cfg is None:
            # Permissive default for unknown providers
            _limiters[provider_name] = RateLimiter(rps=10.0, tpm=1_000_000)
        else:
            _limiters[provider_name] = RateLimiter(rps=cfg.rps, tpm=cfg.tpm,
                                                    retry_max=cfg.retry_max)
    return _limiters[provider_name]


def reset_limiters() -> None:
    """Reset all limiters (useful between test runs)."""
    _limiters.clear()


async def call_with_retry(
    fn,
    *args,
    provider_name: str,
    tokens_estimate: int = 200,
    **kwargs,
):
    """
    Call an async function with rate-limiting and 429 retry logic.

    fn must be an async callable that returns (response_dict, status_code).
    Raises RuntimeError after retry_max failures.
    """
    limiter = get_limiter(provider_name)
    retry_max = limiter.retry_max

    for attempt in range(retry_max + 1):
        await limiter.acquire(tokens_estimate)
        result, status = await fn(*args, **kwargs)

        if status == 429:
            if attempt >= retry_max:
                raise RuntimeError(f"Rate limit persists after {retry_max} retries for {provider_name}")
            delay = limiter.on_429(attempt)
            await asyncio.sleep(delay)
            continue

        return result, status

    raise RuntimeError(f"Exhausted retries for {provider_name}")
