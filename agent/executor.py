"""
Executors for local and remote model inference.

- RuleBasedExecutor: Handles simple tasks with pattern matching (zero cost, no GPU).
- LocalExecutor:  Runs a HuggingFace model in-process (zero scored tokens).
- RemoteExecutor: Calls the Fireworks AI API via OpenAI-compatible client.
- HybridExecutor: Orchestrates all three with:
    • Response caching (identical prompts = free)
    • Rule-based fast path (simple tasks = instant + free)
    • Prompt compression (fewer remote tokens)
    • Dynamic token budgeting (tight max_tokens per task)
    • Cascading fallback (rules → local model → verify → remote)
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from openai import OpenAI

from agent.budget import estimate_token_budget
from agent.cache import ResponseCache
from agent.compressor import compress_prompt
from agent.config import AppConfig, get_resolved_model
from agent.models import (
    ExecutionResult,
    Route,
    RoutingDecision,
    Task,
    TokenUsage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Remote executor (Fireworks AI)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Category detection & system prompts (hackathon-specific)
# ---------------------------------------------------------------------------

_CATEGORY_PROMPTS: dict[str, str] = {
    "sentiment": "Return ONLY the sentiment label (positive, negative, or neutral). Do not include any other text or justification.",
    "ner": "Extract all named entities. Format as a comma-separated list of entities.",
    "summarization": "Summarize the text concisely.",
    "code_debug": "Return ONLY the corrected code. No explanations.",
    "code_gen": "Return ONLY the raw code. Do not use markdown formatting or explanations.",
    "math": "Return ONLY the final numerical answer. Do not show your steps or any text.",
    "logic": "Return ONLY the final conclusion.",
    "factual": "Answer accurately and concisely. Return ONLY the answer, with no conversational filler.",
}


def _detect_category(prompt: str) -> str:
    """Detect the hackathon task category from the prompt text."""
    p = prompt.lower()

    # Code debugging — check early (code prompts may contain other keywords)
    if any(kw in p for kw in ["debug", "fix the bug", "bug in", "what is wrong with this code",
                               "find the error", "fix this code", "buggy",
                               "incorrect output", "doesn't work"]):
        return "code_debug"

    # Code generation — check early
    if any(kw in p for kw in ["write a function", "write a program", "implement a",
                               "write code", "write a python", "write a class",
                               "write a script", "generate code", "create a function",
                               "write a method", "code to"]):
        return "code_gen"

    # Sentiment classification
    if any(kw in p for kw in ["sentiment", "positive or negative", "classify the feeling",
                               "is this positive", "is this negative", "tone of",
                               "classify the sentiment", "sentiment analysis"]):
        return "sentiment"

    # Named entity recognition — specific keywords only
    if any(kw in p for kw in ["named entit", "extract entit", " ner ",
                               "identify the entit", "extract the names",
                               "person, org", "people, places", "entities in"]):
        return "ner"

    # Text summarisation
    if any(kw in p for kw in ["summarise", "summarize", "summary of", "condense",
                               "in one sentence", "in a few words", "tldr",
                               "brief overview", "shorten this"]):
        return "summarization"

    # Mathematical reasoning
    if any(kw in p for kw in ["calculate", "compute", "what is the value",
                               "how much does", "how many", "percentage", "profit",
                               "ratio", "solve for", "equation", "total cost",
                               "interest rate", "probability of"]):
        return "math"

    # Logical / deductive reasoning
    if any(kw in p for kw in ["logic", "deduc", "if all", "must be true",
                               "which of the following", "constraint",
                               "puzzle", "who lives in", "who has which",
                               "given that", "therefore", "conclude",
                               "can we conclude"]):
        return "logic"

    return "factual"


class RemoteExecutor:
    """Calls the remote API through the OpenAI-compatible SDK."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config.fireworks
        self._resolved_model = get_resolved_model(config)
        self._client = OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
        )

    def execute(
        self,
        task: Task,
        max_tokens_override: Optional[int] = None,
        compress: bool = True,
    ) -> ExecutionResult:
        """Send the task to the remote model and return the result."""
        prompt = task.prompt

        # Compress the prompt to save tokens
        if compress:
            prompt = compress_prompt(prompt)

        max_tokens = max_tokens_override or self._config.max_tokens

        # Category-aware system prompt for better accuracy
        category = _detect_category(task.prompt)
        system_prompt = _CATEGORY_PROMPTS.get(category, "Answer accurately and concisely.")

        logger.info(
            "Remote execution for task %s via %s (max_tokens=%d, category=%s)",
            task.id, self._resolved_model, max_tokens, category,
        )
        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=self._resolved_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=max_tokens,
            )

            elapsed_ms = (time.perf_counter() - start) * 1000
            usage = response.usage
            output_text = response.choices[0].message.content or ""

            return ExecutionResult(
                output=output_text.strip(),
                route_used=Route.REMOTE,
                token_usage=TokenUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                confidence=1.0,
                latency_ms=elapsed_ms,
                fallback_triggered=False,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Remote execution failed for task %s: %s", task.id, exc)
            return ExecutionResult(
                output=f"[ERROR] Remote model call failed: {exc}",
                route_used=Route.REMOTE,
                token_usage=TokenUsage(),
                confidence=0.0,
                latency_ms=elapsed_ms,
                fallback_triggered=False,
            )


# ---------------------------------------------------------------------------
# Rule-based executor (dynamic computation only — NO hardcoded answers)
# ---------------------------------------------------------------------------

class RuleBasedExecutor:
    """
    Handles tasks that can be solved by pure computation — no model needed.

    IMPORTANT: Per hackathon rules, we must NOT hardcode factual answers.
    Only dynamic computation (math) is allowed here.
    Returns None if the task can't be handled by rules.
    """

    # Math expressions: "What is 7 + 15?", "Calculate 3 * 4"
    _MATH_PATTERN = re.compile(
        r"^(?:what is|calculate|compute|solve|evaluate)?\s*(\d+(?:\.\d+)?)\s*"
        r"([\+\-\*\/\^])\s*(\d+(?:\.\d+)?)[^\d\w]*$",
        re.I,
    )

    def try_execute(self, task: Task) -> Optional[ExecutionResult]:
        """
        Try to answer the task with pure computation.
        Returns None if not possible.
        """
        prompt = task.prompt.strip()

        # Only attempt math on short prompts that are clearly math questions.
        # Long prompts with numbers embedded (e.g., summarization tasks) should
        # NOT be intercepted here.
        word_count = len(prompt.split())
        if word_count <= 20:
            result = self._try_math(prompt)
            if result is not None:
                return result

        return None

    def _make_result(self, output: str) -> ExecutionResult:
        """Create a zero-cost execution result."""
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
            # Format as int if whole number
            answer = str(int(result)) if result == int(result) else str(result)
            return self._make_result(answer)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Local executor (HuggingFace transformers)
# ---------------------------------------------------------------------------

class LocalExecutor:
    """
    Runs a local HuggingFace model for inference.

    The model is lazily loaded on first call to avoid slow container
    startup when the local path isn't needed.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config.local_model
        self._model: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._load_failed: bool = False

    def _load_model(self) -> None:
        """Lazy-load the model and tokenizer."""
        if self._model is not None:
            return
        if self._load_failed:
            return

        logger.info("Loading local model: %s", self._config.model_name)
        start = time.perf_counter()

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "auto": "auto",
            }
            torch_dtype = dtype_map.get(self._config.torch_dtype, "auto")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_name,
                torch_dtype=torch_dtype,
                device_map=self._config.device,
                trust_remote_code=True,
            )

            elapsed = (time.perf_counter() - start) * 1000
            logger.info("Local model loaded in %.0fms", elapsed)

        except Exception as exc:
            logger.error("Failed to load local model: %s", exc)
            self._load_failed = True
            raise

    def execute(
        self,
        task: Task,
        max_new_tokens_override: Optional[int] = None,
    ) -> ExecutionResult:
        """Run the task through the local model."""
        self._load_model()
        logger.info("Local execution for task %s", task.id)
        start = time.perf_counter()

        max_new_tokens = max_new_tokens_override or self._config.max_new_tokens

        try:
            import torch

            tokenizer = self._tokenizer
            model = self._model

            # Build chat-style prompt
            messages = [{"role": "user", "content": task.prompt}]

            # Use chat template if available, else fall back to raw prompt
            if hasattr(tokenizer, "apply_chat_template"):
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = f"User: {task.prompt}\nAssistant:"

            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=max(self._config.temperature, 0.01),
                    do_sample=self._config.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Decode only the generated tokens (not the prompt)
            generated_ids = outputs.sequences[0][prompt_len:]
            output_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            completion_tokens = len(generated_ids)

            # Estimate confidence from output logits
            confidence = self._estimate_confidence(outputs)

            elapsed_ms = (time.perf_counter() - start) * 1000

            return ExecutionResult(
                output=output_text.strip(),
                route_used=Route.LOCAL,
                token_usage=TokenUsage(
                    prompt_tokens=prompt_len,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_len + completion_tokens,
                ),
                confidence=confidence,
                latency_ms=elapsed_ms,
                fallback_triggered=False,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Local execution failed for task %s: %s", task.id, exc)
            return ExecutionResult(
                output=f"[ERROR] Local model failed: {exc}",
                route_used=Route.LOCAL,
                confidence=0.0,
                latency_ms=elapsed_ms,
                fallback_triggered=False,
            )

    @staticmethod
    def _estimate_confidence(outputs: object) -> float:
        """
        Estimate confidence from generation scores.

        Uses the mean of the top-1 softmax probabilities across generated
        tokens as a rough confidence proxy.
        """
        try:
            import torch

            if not hasattr(outputs, "scores") or not outputs.scores:
                return 0.5

            confidences = []
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                top_prob = probs.max().item()
                confidences.append(top_prob)

            if not confidences:
                return 0.5

            return sum(confidences) / len(confidences)

        except Exception:
            return 0.5


# ---------------------------------------------------------------------------
# Output verifier (for cascading execution)
# ---------------------------------------------------------------------------

class OutputVerifier:
    """
    Lightweight output verification — checks if a local model response
    looks "good enough" without using another LLM.

    This enables cascading: local → verify → escalate only if bad.
    """

    @staticmethod
    def verify(task: Task, result: ExecutionResult) -> bool:
        """
        Returns True if the output passes basic quality checks.
        Returns False if it should be escalated to the remote model.
        """
        output = result.output.strip()

        # 1. Empty or error output → fail
        if not output or output.startswith("[ERROR]"):
            return False

        # 2. Too short for the prompt complexity
        prompt_words = len(task.prompt.split())
        output_words = len(output.split())

        # If prompt is non-trivial but output is suspiciously tiny → fail
        if prompt_words > 15 and output_words < 2:
            return False

        # Ratio check: output should be at least ~5% of prompt for complex tasks
        if prompt_words > 30 and output_words < max(3, prompt_words * 0.05):
            return False

        # 3. Repetition detection — if the output repeats itself excessively
        if OutputVerifier._has_excessive_repetition(output):
            return False

        # 4. Confidence-based (already handled elsewhere, but double-check)
        if result.confidence < 0.10:
            return False

        # 5. Gibberish detection — high ratio of non-alphanumeric chars
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in output) / max(len(output), 1)
        if alnum_ratio < 0.5:
            return False

        return True

    @staticmethod
    def _has_excessive_repetition(text: str) -> bool:
        """Check if text contains excessive repetition (degenerate output)."""
        words = text.split()
        if len(words) < 10:
            return False

        # Check if any single word repeats more than 40% of the time
        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / len(words) > 0.4:
            return True

        # Check for repeated n-grams (3-word sequences)
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            trigram_counts = Counter(trigrams)
            most_common_tri = trigram_counts.most_common(1)[0][1]
            if most_common_tri > 3 and most_common_tri / len(trigrams) > 0.3:
                return True

        return False


# ---------------------------------------------------------------------------
# Hybrid executor (orchestrator with all optimizations)
# ---------------------------------------------------------------------------

class HybridExecutor:
    """
    Orchestrates local and remote execution with full optimization stack:

    1. Cache check → instant free response if hit
    2. Route decision → local or remote
    3. If local: execute → verify output → fallback to remote if bad
    4. If remote: compress prompt → dynamic token budget → execute
    5. Cache the result for future hits
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._rules = RuleBasedExecutor()
        self._local = LocalExecutor(config)
        self._remote = RemoteExecutor(config)
        self._cache = ResponseCache(enabled=config.cache_enabled)
        self._verifier = OutputVerifier()
        self._fallback_threshold = config.router.confidence_fallback_threshold
        self._compression_enabled = config.compression_enabled

    @property
    def cache(self) -> ResponseCache:
        """Expose cache for stats reporting."""
        return self._cache

    def execute(
        self, task: Task, decision: RoutingDecision
    ) -> ExecutionResult:
        """Execute the task with full optimization pipeline."""

        # ── Step 1: Cache check ──
        cached = self._cache.get(task.prompt)
        if cached is not None:
            logger.info("Task %s: served from cache (0 tokens)", task.id)
            return cached

        # Compute dynamic token budget
        token_budget = estimate_token_budget(task, decision)

        # ── Step 2: Rule-based fast path (instant, free, no GPU) ──
        rule_result = self._rules.try_execute(task)
        if rule_result is not None:
            logger.info("Task %s: answered by rule-based executor (0 tokens)", task.id)
            self._cache.put(task.prompt, rule_result)
            return rule_result

        # ── Step 3: Route to remote ──
        if decision.route == Route.REMOTE:
            result = self._remote.execute(
                task,
                max_tokens_override=token_budget,
                compress=self._compression_enabled,
            )
            self._cache.put(task.prompt, result)
            return result

        # ── Step 4: Local model execution with cascading verification ──
        try:
            local_result = self._local.execute(
                task,
                max_new_tokens_override=token_budget,
            )
        except Exception as exc:
            logger.warning(
                "Task %s: local model unavailable (%s) — routing to remote",
                task.id, exc,
            )
            result = self._remote.execute(
                task,
                max_tokens_override=token_budget,
                compress=self._compression_enabled,
            )
            self._cache.put(task.prompt, result)
            return result

        # ── Step 5: Verify local output quality ──
        confidence_ok = local_result.confidence >= self._fallback_threshold
        verification_ok = self._verifier.verify(task, local_result)

        if confidence_ok and verification_ok:
            # Local result is good — cache it and return (0 scored tokens!)
            self._cache.put(task.prompt, local_result)
            return local_result

        # ── Step 6: Escalate to remote (fallback) ──
        reason = []
        if not confidence_ok:
            reason.append(
                f"confidence {local_result.confidence:.2f} < {self._fallback_threshold:.2f}"
            )
        if not verification_ok:
            reason.append("output verification failed")

        logger.warning(
            "Task %s: local output rejected (%s) — falling back to remote",
            task.id,
            ", ".join(reason),
        )

        remote_result = self._remote.execute(
            task,
            max_tokens_override=token_budget,
            compress=self._compression_enabled,
        )
        remote_result.fallback_triggered = True
        self._cache.put(task.prompt, remote_result)
        return remote_result
