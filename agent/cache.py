"""
Response cache for the Hybrid Token-Efficient Routing Agent.

Caches model responses keyed by normalized prompts so that identical
or near-identical tasks can be answered for FREE (zero remote tokens).

Supports:
  - Exact match caching (hash-based, O(1) lookup)
  - Normalized matching (lowercased, stripped, punctuation-normalized)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agent.models import ExecutionResult, Route, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Tracking cache performance."""
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass
class CacheEntry:
    """A single cached response."""
    output: str
    route_used: Route
    confidence: float


class ResponseCache:
    """
    In-memory response cache.

    Keys are normalized prompt hashes for fast O(1) lookups.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._cache: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    @staticmethod
    def _normalize(prompt: str) -> str:
        """Normalize prompt for cache key generation."""
        text = prompt.lower().strip()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove trailing punctuation variations
        text = text.rstrip("?.! ")
        return text

    @staticmethod
    def _hash(text: str) -> str:
        """Generate a hash key from normalized text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get(self, prompt: str) -> Optional[ExecutionResult]:
        """Look up a cached response. Returns None on miss."""
        if not self._enabled:
            self.stats.misses += 1
            return None

        normalized = self._normalize(prompt)
        key = self._hash(normalized)

        entry = self._cache.get(key)
        if entry is not None:
            self.stats.hits += 1
            logger.info("Cache HIT for prompt (key=%s)", key)
            return ExecutionResult(
                output=entry.output,
                route_used=Route.LOCAL,  # Cached = free = local
                token_usage=TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
                confidence=entry.confidence,
                latency_ms=0.0,
                fallback_triggered=False,
            )

        self.stats.misses += 1
        return None

    def put(self, prompt: str, result: ExecutionResult) -> None:
        """Store a response in the cache."""
        if not self._enabled:
            return

        # Don't cache error responses
        if result.output.startswith("[ERROR]"):
            return

        normalized = self._normalize(prompt)
        key = self._hash(normalized)

        self._cache[key] = CacheEntry(
            output=result.output,
            route_used=result.route_used,
            confidence=result.confidence,
        )
        logger.debug("Cached response for prompt (key=%s)", key)

    @property
    def size(self) -> int:
        return len(self._cache)
