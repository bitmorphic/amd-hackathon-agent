"""
Dynamic token budget estimator.

Estimates the optimal max_tokens for a given task based on its
complexity, type, and the hackathon task category.  Setting tight
budgets on remote calls saves scored tokens without hurting accuracy.

KEY INSIGHT: The hackathon uses an LLM-Judge for accuracy, so answers
must be complete enough to demonstrate understanding. But tokens are
the scoring metric, so we cap verbosity.
"""

from __future__ import annotations

import re
from agent.models import RoutingDecision, Task, TaskDifficulty


# ---------------------------------------------------------------------------
# Category detection (mirrors executor._detect_category)
# ---------------------------------------------------------------------------

def _detect_budget_category(prompt: str) -> str:
    """Detect task category for budget allocation."""
    p = prompt.lower()

    # Code (debug + generation) — needs most tokens for complete code
    if any(kw in p for kw in ["write a function", "write a program", "implement a",
                               "write code", "write a python", "write a class",
                               "write a script", "generate code", "create a function",
                               "write a method", "code to",
                               "debug", "fix the bug", "bug in", "fix this code",
                               "what is wrong", "find the error", "buggy"]):
        return "code"

    # Sentiment — short label + brief justification
    if any(kw in p for kw in ["sentiment", "positive or negative", "classify the",
                               "is this positive", "is this negative", "tone of"]):
        return "sentiment"

    # NER — structured extraction
    if any(kw in p for kw in ["named entit", "extract entit", " ner ",
                               "identify the entit", "extract the names",
                               "person, org", "entities in"]):
        return "ner"

    # Summarisation — typically 1-3 sentences
    if any(kw in p for kw in ["summarise", "summarize", "summary of", "condense",
                               "in one sentence", "in a few words", "tldr"]):
        return "summarization"

    # Math — step-by-step + answer
    if any(kw in p for kw in ["calculate", "compute", "what is the value",
                               "original price", "compound interest",
                               "how much", "how many", "percentage", "profit",
                               "total distance", "total cost", "interest rate",
                               "probability of", "discount"]):
        return "math"

    # Simple arithmetic (just a number)
    if re.search(r"\d+\s*[\+\-\*\/\^]\s*\d+", p):
        return "simple_math"

    # Logic — step-by-step reasoning
    if any(kw in p for kw in ["logic", "deduc", "if all", "must be true",
                               "can we conclude", "constraint",
                               "puzzle", "who has", "given that"]):
        return "logic"

    # Explanation
    if any(kw in p for kw in ["explain", "describe", "how does", "what is",
                               "what causes", "what are", "concept of"]):
        return "explanation"

    return "general"


# ---------------------------------------------------------------------------
# Budget allocation per category
# ---------------------------------------------------------------------------

# These budgets balance accuracy (LLM-Judge) vs token efficiency (scoring).
# Each is tuned to be the minimum needed for a quality answer.
_CATEGORY_BUDGETS: dict[str, int] = {
    "code":          400,   # Complete functions with docstrings
    "sentiment":     80,    # Label + 1-2 sentence justification
    "ner":           150,   # Structured entity list
    "summarization": 100,   # 1-3 concise sentences
    "math":          200,   # Step-by-step + final answer
    "simple_math":   30,    # Just the number
    "logic":         200,   # Step-by-step reasoning + conclusion
    "explanation":   200,   # 2-4 sentences
    "general":       150,   # Default
}


def estimate_token_budget(task: Task, decision: RoutingDecision) -> int:
    """
    Estimate optimal max_tokens for a task.

    Returns a tight budget that's just enough for a quality answer
    that will pass the LLM-Judge accuracy gate while minimizing
    scored tokens.
    """
    prompt = task.prompt
    word_count = len(prompt.split())
    category = _detect_budget_category(prompt)

    # Get base budget from category
    base_budget = _CATEGORY_BUDGETS.get(category, 150)

    # Adjust for complexity
    if decision.complexity_score >= 0.6:
        base_budget = int(base_budget * 1.3)  # 30% more for complex tasks

    # Adjust for long prompts (they usually need longer answers)
    if word_count > 50 and category not in ("simple_math", "sentiment"):
        base_budget = max(base_budget, 250)

    # Cap at reasonable maximum
    return min(base_budget, 600)
