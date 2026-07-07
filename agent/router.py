"""
Routing brain for the Hybrid Token-Efficient Routing Agent.

Uses a zero-cost heuristic classifier to decide whether each task
should be handled by the local model (free) or the remote Fireworks AI
model (costly).  No LLM inference is used for routing itself.

Signals analyzed:
  1. Prompt length (longer → harder)
  2. Keyword complexity markers (code, math, reasoning, etc.)
  3. Structural complexity (nested questions, multi-step instructions)
  4. Output format requests (JSON, tables, lists)
  5. Domain detection (translation, creative writing, factual Q&A)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agent.config import RouterConfig
from agent.models import (
    Route,
    RoutingDecision,
    Task,
    TaskDifficulty,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal keywords — curated lists for heuristic classification
# ---------------------------------------------------------------------------

# High-complexity indicators → favour remote
# Using single words so they match naturally inside longer prompts.
_COMPLEX_KEYWORDS: set[str] = {
    # Reasoning & logic
    "reason", "reasoning", "explain", "analyze", "analyse", "evaluate",
    "compare", "contrast", "critique", "prove", "proof", "derive",
    "deduce", "infer", "justify", "argue", "hypothesis",
    # Math
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "probability", "statistics", "algebra", "geometry", "theorem",
    "mathematical", "formula", "irrational",
    # Code & engineering
    "implement", "debug", "refactor", "algorithm", "function",
    "class", "optimize", "complexity", "backpropagation", "neural",
    "api", "endpoint", "authentication", "schema", "architect",
    "design", "system",
    # Multi-step markers
    "step", "first", "then", "finally", "multi-step",
    "strategy", "plan", "outline",
    # Creative / long-form
    "essay", "story", "compose", "draft", "comprehensive",
    "detailed", "in-depth", "thorough", "elaborate",
    # Generalization signals
    "generalize", "generalise", "extend", "apply",
}

# Low-complexity indicators → favour local
_SIMPLE_KEYWORDS: set[str] = {
    "translate", "summarize", "summarise", "define", "what is",
    "who is", "when was", "where is", "yes or no", "true or false",
    "short answer", "brief", "one word",
    "hello", "hi", "thanks", "thank you", "greet",
    "capital", "name",
    # Additional patterns for local routing
    "how many", "how old", "how far", "how long",
    "what color", "what colour", "what year", "what day",
    "meaning of", "definition of", "synonym", "antonym",
    "spell", "abbreviation", "acronym",
    "convert", "temperature", "currency",
    "largest", "smallest", "tallest", "fastest",
    "president", "founder", "inventor", "author",
    "continent", "country", "city", "planet",
    "simple", "basic", "quick", "easy",
}

# Structural output requests (moderate+ complexity)
_STRUCTURED_OUTPUT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bjson\b", re.IGNORECASE),
    re.compile(r"\btable\b", re.IGNORECASE),
    re.compile(r"\bcsv\b", re.IGNORECASE),
    re.compile(r"\bmarkdown\b", re.IGNORECASE),
    re.compile(r"\byaml\b", re.IGNORECASE),
    re.compile(r"\bxml\b", re.IGNORECASE),
]

# Multi-part question patterns
_MULTI_PART_PATTERN = re.compile(
    r"(\d+[\.\)]\s)|(\b(and also|additionally|furthermore|moreover)\b)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Heuristic router
# ---------------------------------------------------------------------------

class HeuristicRouter:
    """
    Zero-overhead router that classifies task complexity using fast
    string heuristics.  No model inference is performed.

    The complexity score is a weighted combination of multiple signals,
    each normalized to [0, 1].  The final score is compared against
    `config.complexity_threshold` to decide local vs. remote.
    """

    # Signal weights (sum to ~1.0 for interpretability)
    _WEIGHTS: dict[str, float] = {
        "length": 0.10,
        "complex_keywords": 0.30,
        "simple_keywords": 0.15,  # negative signal (boosted for stronger local bias)
        "structured_output": 0.10,
        "multi_part": 0.10,
        "question_depth": 0.05,
        "sentence_count": 0.20,
    }

    def __init__(self, config: RouterConfig) -> None:
        self._config = config

    def route(self, task: Task) -> RoutingDecision:
        """Classify a task and decide where to route it."""
        signals = self._compute_signals(task.prompt)
        score = self._aggregate(signals)
        difficulty = self._score_to_difficulty(score)
        chosen_route = (
            Route.REMOTE
            if score >= self._config.complexity_threshold
            else Route.LOCAL
        )

        reason = self._build_reason(chosen_route, score, difficulty, signals)

        decision = RoutingDecision(
            route=chosen_route,
            complexity_score=round(score, 4),
            difficulty=difficulty,
            reason=reason,
            signals=signals,
        )

        logger.debug(
            "Router decision for task %s: %s (score=%.3f)",
            task.id, chosen_route.value, score,
        )
        return decision

    # ----- signal computation -----

    def _compute_signals(self, prompt: str) -> dict[str, Any]:
        """Compute individual heuristic signals from the prompt."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        return {
            "length": self._length_signal(word_count),
            "complex_keywords": self._keyword_signal(prompt_lower, _COMPLEX_KEYWORDS),
            "simple_keywords": self._keyword_signal(prompt_lower, _SIMPLE_KEYWORDS),
            "structured_output": self._structured_output_signal(prompt),
            "multi_part": self._multi_part_signal(prompt),
            "question_depth": self._question_depth_signal(prompt),
            "sentence_count": self._sentence_count_signal(prompt),
            "word_count": word_count,  # informational, not scored
        }

    @staticmethod
    def _length_signal(word_count: int) -> float:
        """Longer prompts tend to be more complex.  Sigmoid-ish curve."""
        if word_count <= 10:
            return 0.1
        if word_count <= 30:
            return 0.3
        if word_count <= 80:
            return 0.5
        if word_count <= 150:
            return 0.7
        return 0.9

    @staticmethod
    def _keyword_signal(prompt_lower: str, keywords: set[str]) -> float:
        """Fraction of keyword set that appears in the prompt (capped at 1)."""
        hits = sum(1 for kw in keywords if kw in prompt_lower)
        # Normalize: 2+ hits is a strong signal
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _structured_output_signal(prompt: str) -> float:
        """Does the prompt request structured output?"""
        hits = sum(1 for pat in _STRUCTURED_OUTPUT_PATTERNS if pat.search(prompt))
        return min(hits / 2.0, 1.0)

    @staticmethod
    def _multi_part_signal(prompt: str) -> float:
        """Does the prompt contain multiple sub-questions or steps?"""
        matches = _MULTI_PART_PATTERN.findall(prompt)
        return min(len(matches) / 3.0, 1.0)

    @staticmethod
    def _question_depth_signal(prompt: str) -> float:
        """Count question marks as a rough proxy for complexity."""
        q_count = prompt.count("?")
        if q_count <= 1:
            return 0.1
        if q_count <= 3:
            return 0.5
        return 0.9

    @staticmethod
    def _sentence_count_signal(prompt: str) -> float:
        """More sentences usually means a more complex, multi-part request."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', prompt)
        count = len([s for s in sentences if s.strip()])
        if count <= 1:
            return 0.05
        if count <= 2:
            return 0.2
        if count <= 4:
            return 0.5
        if count <= 6:
            return 0.75
        return 1.0

    # ----- aggregation -----

    def _aggregate(self, signals: dict[str, Any]) -> float:
        """Weighted combination of signals into a single complexity score."""
        score = 0.0
        for name, weight in self._WEIGHTS.items():
            value = signals.get(name, 0.0)
            if name == "simple_keywords":
                # Simple keywords are a *negative* signal: high match → low complexity
                score += weight * (1.0 - value)
            else:
                score += weight * value
        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_to_difficulty(score: float) -> TaskDifficulty:
        if score < 0.35:
            return TaskDifficulty.SIMPLE
        if score < 0.65:
            return TaskDifficulty.MODERATE
        return TaskDifficulty.COMPLEX

    @staticmethod
    def _build_reason(
        route: Route,
        score: float,
        difficulty: TaskDifficulty,
        signals: dict[str, Any],
    ) -> str:
        top_signals = sorted(
            [(k, v) for k, v in signals.items() if isinstance(v, float)],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        signal_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals)
        return (
            f"Complexity {score:.2f} ({difficulty.value}) → {route.value}. "
            f"Top signals: {signal_str}"
        )
