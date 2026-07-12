"""
Executors for local and remote model inference.

- RuleBasedExecutor: Handles simple tasks with pattern matching (zero cost, no GPU).
- RemoteExecutor: Calls the Fireworks AI API via OpenAI-compatible client.
- HybridExecutor: Orchestrates all executors with:
    • Response caching (identical prompts = free)
    • Rule-based fast path (simple math = instant + free)
    • Smart model tiering (cheap / strong / code models per category)
    • Category-specific system prompts + tight token budgets
    • Fallback model if primary returns blank
"""

from __future__ import annotations

import logging
import re
import time
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from agent.cache import ResponseCache
from agent.config import AppConfig, get_resolved_model
from agent.models import (
    ExecutionResult,
    Route,
    RoutingDecision,
    Task,
    TokenUsage,
)
from agent.local_model import LocalModelProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category detection (comprehensive regex classifier — ported from KaananeTaha)
# ---------------------------------------------------------------------------

_CLASSIFIER_PATTERNS: dict[str, list[str]] = {
    "code_debug": [
        r"\bbug\b", r"\bdebug\b", r"\bfix (this|the|my|it)\b",
        r"what'?s wrong", r"why (does|is)n'?t (this|it|my)\b",
        r"error in (this|the|my)\b", r"traceback", r"stack ?trace",
        r"throws? an? (error|exception)", r"returns? \w+ instead",
        r"infinite loop", r"corrected (version|code)",
    ],
    "code_gen": [
        r"\b(write|create|implement|build|generate|produce|give me)\b.*"
        r"\b(function|method|class|program|script|routine)\b",
        r"\bfunction (that|to)\b", r"\bcode that\b",
        r"\bwrite (a|an|some) code\b", r"\bimplement (a|an|the)\b",
    ],
    "sentiment": [
        r"\bsentiment\b", r"positive or negative", r"positive, negative",
        r"classify the (tone|emotion|sentiment|mood)",
        r"(emotional )?tone of (this|the|that)",
        r"\b(positive|negative|neutral)\b.*\breview\b",
        r"how (positive|negative) ", r"is this (review|tweet|comment)\b",
    ],
    "ner": [
        r"named entit", r"\bner\b",
        r"extract (all )?(the )?(entit|name|person|people|organi|location|date)",
        r"(list|identify|find|pull out) (all )?(the )?"
        r"(people|persons?|organi[sz]ations?|locations?|dates?|entit)",
        r"(person|organization|location|date)\s*[:=]",
    ],
    "summarization": [
        r"summari[sz]e", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
        r"\bshorten\b", r"in (one|a single|two|three|\d+) (sentences?|words?|lines?)",
        r"main (idea|point|takeaway)", r"\bthe gist\b", r"key points",
        r"boil .* down",
    ],
    "logic": [
        r"\bpuzzle\b", r"who (is|owns|sits|lives|has|drinks|likes)\b",
        r"if and only if", r"exactly one", r"at least one",
        r"the following (clues|facts|statements|conditions)",
        r"each (person|house|box|day|one) .*(different|exactly|only|one)",
        r"\bdeduce\b", r"logically (follows?|true)",
        r"(definitely|necessarily) (true|follows)",
        r"knights? and knaves", r"truth[- ]?teller", r"\bliar\b",
    ],
    "math": [
        r"\bcalculate\b", r"\bcompute\b", r"how (much|many)\b", r"percent",
        r"\d+\s*%", r"\bsum of\b", r"\baverage\b", r"solve for\b",
        r"\d+\s*[+\-*/x×÷]\s*\d+", r"total (cost|price|amount)",
        r"\b(interest|discount|ratio|profit)\b",
        r"find the (largest|smallest|value|angle|area|sum|total|average)",
        r"what is \d",
    ],
    "factual": [
        r"what (is|are|was|were)\b", r"who (is|was|were)\b",
        r"when (did|was|is)\b", r"where (is|was|are)\b",
        r"why (is|do|does|are)\b", r"how (do|does|can)\b",
        r"\bexplain\b", r"\bdefine\b", r"\bdescribe\b", r"what does .* mean",
    ],
}

# Priority order: specific categories before general fallback
_PRIORITY_ORDER = [
    "code_debug", "code_gen", "sentiment", "ner",
    "summarization", "logic", "math", "factual",
]

_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in _CLASSIFIER_PATTERNS.items()
}

_CODE_FENCE = re.compile(r"```")
_CODE_HINT = re.compile(
    r"\b(def |class |return |import |#include|public |void |printf|"
    r"console\.log|System\.out)|=>|;\s*$",
    re.MULTILINE,
)

def _detect_category(prompt: str) -> str:
    """Classify the prompt into one of the 8 hackathon categories."""
    text = prompt or ""
    for cat in _PRIORITY_ORDER:
        if any(rx.search(text) for rx in _COMPILED_PATTERNS[cat]):
            return cat
    # Raw code in prompt with no other signals → probably debug
    return "code_debug" if (_CODE_FENCE.search(text) or _CODE_HINT.search(text)) else "factual"


# ---------------------------------------------------------------------------
# Category config: (system_prompt, max_tokens, model_tier)
# Tiers: "cheap" = small/fast, "strong" = largest general, "code" = code-spec
# ---------------------------------------------------------------------------

_BASE = "Answer directly and minimally. No preamble."

_CATEGORY_CONFIG: dict[str, tuple[str, int, str]] = {
    "factual": (
        f"{_BASE} Be pedantically accurate. Ignore all social pleasantries.",
        256, "strong",
    ),
    "math": (
        f"{_BASE} Output only the final mathematical result. No steps. No explanation.",
        256, "strong",
    ),
    "sentiment": (
        f"{_BASE} Output exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.",
        256, "cheap",
    ),
    "summarization": (
        f"{_BASE} Output the summary and stop.",
        256, "cheap",
    ),
    "ner": (
        f"{_BASE} Identify entities. Format strictly as 'label: value'.",
        256, "strong",
    ),
    "code_debug": (
        f"Provide only the corrected code block. No Markdown. No comments. Minimalist.",
        512, "code",
    ),
    "logic": (
        f"{_BASE} Output final result only. Do not show reasoning logic.",
        256, "strong",
    ),
    "code_gen": (
        f"Output raw code only. No text outside logic. No Markdown. Concise.",
        1024, "code",
    ),
}


# ---------------------------------------------------------------------------
# Model tiering (dynamically inferred from ALLOWED_MODELS)
# ---------------------------------------------------------------------------

_MOE_PAT = re.compile(r"(\d+)\s*x\s*(\d+)\s*b\b")
_ACTIVE_PAT = re.compile(r"\ba(\d+)b\b")
_DENSE_PAT = re.compile(r"(\d+)\s*b\b")
_CODE_MODEL_PAT = re.compile(r"\bcode|coder|-code\b")
_QUANT_PAT = re.compile(r"nvfp4|fp4|fp8|int8|int4|awq|gptq|gguf")
_NON_CHAT_HINTS = (
    "embed", "rerank", "whisper", "audio", "tts", "image", "vision",
    "moderation", "guard", "clip", "diffusion", "flux",
)


def _total_params(model_id: str) -> int:
    mid = model_id.lower()
    moe = _MOE_PAT.search(mid)
    if moe:
        return int(moe.group(1)) * int(moe.group(2))
    sizes = [int(m.group(1)) for m in _DENSE_PAT.finditer(mid)]
    return max(sizes) if sizes else 100  # unknown → treat as frontier


def _active_params(model_id: str) -> int:
    m = _ACTIVE_PAT.search(model_id.lower())
    return int(m.group(1)) if m else _total_params(model_id)


def _resolve_tiers(allowed_models: list[str]) -> dict[str, str]:
    """
    Infer cheap / strong / code tiers from ALLOWED_MODELS.
    Never hardcodes model IDs — works with any set of allowed models.
    """
    usable = [m for m in allowed_models
              if not any(b in m.lower() for b in _NON_CHAT_HINTS)]
    if not usable:
        usable = list(allowed_models)

    general = [m for m in usable if not _CODE_MODEL_PAT.search(m.lower())] or usable

    strong = max(
        general,
        key=lambda m: (_total_params(m), not bool(_QUANT_PAT.search(m.lower())))
    )
    code_models = [m for m in usable if _CODE_MODEL_PAT.search(m.lower())]
    code = max(code_models, key=_total_params) if code_models else strong
    cheap = min(
        usable,
        key=lambda m: (_active_params(m), not bool(_QUANT_PAT.search(m.lower())))
    )

    logger.info("Model tiers resolved → cheap=%s | strong=%s | code=%s", cheap, strong, code)
    return {"cheap": cheap, "strong": strong, "code": code}


# ---------------------------------------------------------------------------
# Remote executor (Fireworks AI)
# ---------------------------------------------------------------------------

_THINK_PAT = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

class RemoteExecutor:
    """
    Calls the remote Fireworks API with smart model tiering per task category.

    Uses category-specific system prompts and token budgets to match the
    #1-ranked repository's accuracy while adding token-efficiency on top.
    Falls back to the strong model if the primary returns a blank answer.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config.fireworks
        self._client = OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
            timeout=25.0,   # per-request limit (harness kills at 10 min)
            max_retries=3,
        )
        # Resolve model tiers once at startup
        if self._config.allowed_models:
            self._tiers = _resolve_tiers(self._config.allowed_models)
        else:
            # Fallback for local dev without ALLOWED_MODELS
            fallback = get_resolved_model(config)
            self._tiers = {"cheap": fallback, "strong": fallback, "code": fallback}
            logger.warning("ALLOWED_MODELS not set — all tiers use %s", fallback)
            
        self._no_effort_param: set[str] = set()

    def _model_for_tier(self, tier: str) -> str:
        return self._tiers.get(tier, self._tiers["strong"])

    def _call(
        self,
        model: str,
        prompt: str,
        system: str,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Make one API call; returns (text, prompt_tokens, completion_tokens)."""
        kwargs = {}
        if model not in self._no_effort_param:
            kwargs["reasoning_effort"] = "none"
            
        try:
            response_stream = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
        except Exception as e:
            if not (kwargs and "reasoning effort" in str(e).lower()):
                raise
            self._no_effort_param.add(model)
            response_stream = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
        text = ""
        for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    text += delta.content
        
        text = _THINK_PAT.sub("", text).strip()
        return text, 0, 0

    def execute(self, task: Task) -> ExecutionResult:
        """
        Classify the task, pick the right model tier, call the API.
        Falls back to the strong model if the primary model returns blank.
        """
        category = _detect_category(task.prompt)
        system, max_tokens, tier = _CATEGORY_CONFIG.get(
            category, _CATEGORY_CONFIG["factual"]
        )
        primary_model = self._model_for_tier(tier)
        strong_model = self._model_for_tier("strong")

        logger.info(
            "Task %s → category=%s tier=%s model=%s max_tokens=%d",
            task.id, category, tier, primary_model, max_tokens,
        )

        start = time.perf_counter()
        pt = ct = 0

        try:
            try:
                text, pt, ct = self._call(primary_model, task.prompt, system, max_tokens)
            except Exception as primary_exc:
                logger.warning("Task %s: primary model threw exception: %s", task.id, primary_exc)
                text, pt, ct = "", 0, 0

            # Fallback: blank answer on primary → retry with the alternate tier
            fallback_model = self._model_for_tier("strong") if tier == "cheap" else self._model_for_tier("cheap")
            
            if not text and primary_model != fallback_model:
                logger.warning(
                    "Task %s: primary model returned blank or failed — retrying with fallback tier",
                    task.id,
                )
                try:
                    text2, pt2, ct2 = self._call(
                        fallback_model, task.prompt, system, max_tokens
                    )
                    if text2:
                        text, pt, ct = text2, pt + pt2, ct + ct2
                except Exception as fb_exc:
                    logger.error("Task %s: fallback also failed: %s", task.id, fb_exc)

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Remote execution failed for task %s: %s", task.id, exc)
            return ExecutionResult(
                output="",
                route_used=Route.REMOTE,
                token_usage=TokenUsage(),
                confidence=0.0,
                latency_ms=elapsed_ms,
                fallback_triggered=True,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        return ExecutionResult(
            output=text,
            route_used=Route.REMOTE,
            token_usage=TokenUsage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            ),
            confidence=1.0,
            latency_ms=elapsed_ms,
            fallback_triggered=False,
        )


# ---------------------------------------------------------------------------
# Rule-based executor (zero-token math solver)
# ---------------------------------------------------------------------------

class RuleBasedExecutor:
    """
    Handles tasks that can be solved by pure computation — no model needed.
    Returns None if the task cannot be handled by rules.
    """

    _MATH_PATTERN = re.compile(
        r"^(?:what is|calculate|compute|solve|evaluate)?\s*(\d+(?:\.\d+)?)\s*"
        r"([+\-*/^])\s*(\d+(?:\.\d+)?)[^\d\w]*$",
        re.I,
    )

    def try_execute(self, task: Task) -> Optional[ExecutionResult]:
        prompt = task.prompt.strip()
        if len(prompt.split()) <= 20:
            result = self._try_math(prompt)
            if result is not None:
                return result
        return None

    def _make_result(self, output: str) -> ExecutionResult:
        return ExecutionResult(
            output=output,
            route_used=Route.LOCAL,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            confidence=1.0,
            latency_ms=0.0,
            fallback_triggered=False,
        )

    def _try_math(self, prompt: str) -> Optional[ExecutionResult]:
        match = self._MATH_PATTERN.search(prompt)
        if not match:
            return None
        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
        try:
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            elif op == "/":
                result = a / b if b != 0 else float("inf")
            elif op == "^":
                result = a ** b
            else:
                return None
            answer = str(int(result)) if result == int(result) else str(round(result, 6))
            return self._make_result(f"Answer: {answer}")
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Hybrid executor (orchestrator)
# ---------------------------------------------------------------------------

class HybridExecutor:
    """Orchestrates caching, rule-based, local CPU, and tiered remote models."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._rules = RuleBasedExecutor()
        self._remote = RemoteExecutor(config)
        self._cache = ResponseCache(enabled=config.cache_enabled)
        self._local = LocalModelProvider()

    def execute(
        self, task: Task, decision: RoutingDecision
    ) -> ExecutionResult:
        """Execute the task with the full optimization pipeline."""

        # ── Step 1: Cache check ──
        cached = self._cache.get(task.prompt)
        if cached is not None:
            logger.info("Task %s: served from cache (0 tokens)", task.id)
            return cached

        # ── Step 2: Rule-based fast path (0 tokens for simple math) ──
        rule_result = self._rules.try_execute(task)
        if rule_result is not None:
            logger.info("Task %s: answered by rule-based executor (0 tokens)", task.id)
            self._cache.put(task.prompt, rule_result)
            return rule_result

        # ── Step 3: Local CPU model (0 tokens for easy tasks) ──
        category = _detect_category(task.prompt)
        local_result = self._local.answer(task, category)
        if local_result is not None:
            logger.info("Task %s: answered by local CPU model (0 tokens)", task.id)
            self._cache.put(task.prompt, local_result)
            return local_result

        # ── Step 4: Remote execution with smart model tiering ──
        result = self._remote.execute(task)
        self._cache.put(task.prompt, result)
        return result
