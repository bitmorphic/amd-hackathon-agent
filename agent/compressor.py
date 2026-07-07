"""
Prompt compressor for reducing remote token usage.

Applies a series of text transformations to strip unnecessary tokens
from prompts before sending them to the remote model, without losing
semantic meaning.  Every token saved here is a token NOT counted in
the final hackathon score.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filler phrases that can be removed without losing meaning
# ---------------------------------------------------------------------------
_FILLER_PHRASES: list[tuple[re.Pattern[str], str]] = [
    # Politeness fluff
    (re.compile(r"\bcould you (please |kindly )?", re.I), ""),
    (re.compile(r"\bwould you (please |kindly )?", re.I), ""),
    (re.compile(r"\bcan you (please |kindly )?", re.I), ""),
    (re.compile(r"\bplease\b", re.I), ""),
    (re.compile(r"\bkindly\b", re.I), ""),
    (re.compile(r"\bi would like you to\b", re.I), ""),
    (re.compile(r"\bi want you to\b", re.I), ""),
    (re.compile(r"\bi need you to\b", re.I), ""),
    (re.compile(r"\bi'm curious about\b", re.I), ""),
    (re.compile(r"\bcan you tell me\b", re.I), ""),
    (re.compile(r"\bi'd like to know\b", re.I), ""),
    (re.compile(r"\bcould you help me with\b", re.I), ""),
    (re.compile(r"\bdo you know\b", re.I), ""),
    (re.compile(r"\bi was wondering\b", re.I), ""),
    (re.compile(r"\bi'm looking for\b", re.I), ""),
    # Verbose instruction prefixes
    (re.compile(r"\bprovide me with\b", re.I), "give"),
    (re.compile(r"\bprovide a\b", re.I), "give a"),
    (re.compile(r"\bin a detailed manner\b", re.I), "in detail"),
    (re.compile(r"\bas detailed as possible\b", re.I), "in detail"),
    (re.compile(r"\bwith a comprehensive\b", re.I), "with a full"),
    (re.compile(r"\bin a comprehensive manner\b", re.I), "fully"),
    (re.compile(r"\bmake sure to\b", re.I), ""),
    (re.compile(r"\bensure that you\b", re.I), ""),
    (re.compile(r"\bdon'?t forget to\b", re.I), ""),
    (re.compile(r"\bit is important to\b", re.I), ""),
    (re.compile(r"\bit would be great if\b", re.I), ""),
    (re.compile(r"\bfor me\b", re.I), ""),
    (re.compile(r"\bfor this task\b", re.I), ""),
    (re.compile(r"\bin your (own )?words\b", re.I), ""),
    (re.compile(r"\bas an ai\b", re.I), ""),
    (re.compile(r"\bas a language model\b", re.I), ""),
    (re.compile(r"\bif possible\b", re.I), ""),
    (re.compile(r"\bwhen possible\b", re.I), ""),
    # Trailing pleasantries
    (re.compile(r"\bthanks? ?(you|in advance)?\.?\s*$", re.I), ""),
    (re.compile(r"\bi appreciate your help\.?\s*$", re.I), ""),
    (re.compile(r"\bany help (is|would be) appreciated\.?\s*$", re.I), ""),
]

# Repeated whitespace cleanup
_MULTI_SPACE = re.compile(r"  +")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def compress_prompt(prompt: str) -> str:
    """
    Compress a prompt to reduce token count while preserving meaning.

    Returns the compressed prompt.  The caller should use this for
    remote model calls to minimize scored tokens.
    """
    original_len = len(prompt.split())
    result = prompt

    # 1. Remove filler phrases
    for pattern, replacement in _FILLER_PHRASES:
        result = pattern.sub(replacement, result)

    # 2. Collapse whitespace
    result = _MULTI_SPACE.sub(" ", result)
    result = _MULTI_NEWLINE.sub("\n\n", result)
    result = result.strip()

    # 3. Remove leading/trailing punctuation artifacts from removals
    result = re.sub(r"^\s*[,;]\s*", "", result)
    result = re.sub(r"\s+([,;.])", r"\1", result)

    compressed_len = len(result.split())
    saved = original_len - compressed_len

    if saved > 0:
        logger.debug(
            "Prompt compressed: %d → %d words (saved %d, %.0f%%)",
            original_len, compressed_len, saved,
            (saved / original_len * 100) if original_len else 0,
        )

    return result


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimate (words × 1.3).

    This avoids needing a real tokenizer just for estimation.
    Good enough for budgeting decisions.
    """
    return int(len(text.split()) * 1.3)
