"""
Configuration management for the Hybrid Token-Efficient Routing Agent.

All settings are loaded from environment variables (with .env file support)
so the container can be configured at runtime without code changes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root (if it exists)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class FireworksConfig:
    """Settings for the remote Fireworks AI API."""
    api_key: str = field(
        default_factory=lambda: os.getenv("FIREWORKS_API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_MODEL",
            "accounts/fireworks/models/llama-v3.1-8b-instruct",
        )
    )
    allowed_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv("ALLOWED_MODELS", "").split(",")
            if m.strip()
        ]
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("FIREWORKS_TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("FIREWORKS_MAX_TOKENS", "512"))
    )


@dataclass(frozen=True)
class LocalModelConfig:
    """Settings for the local model."""
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_MODEL_NAME", "google/gemma-2-2b-it"
        )
    )
    device: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_DEVICE", "auto")
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("LOCAL_MODEL_MAX_NEW_TOKENS", "512")
        )
    )
    temperature: float = field(
        default_factory=lambda: float(
            os.getenv("LOCAL_MODEL_TEMPERATURE", "0.1")
        )
    )
    torch_dtype: str = field(
        default_factory=lambda: os.getenv("LOCAL_MODEL_DTYPE", "auto")
    )


@dataclass(frozen=True)
class RouterConfig:
    """Thresholds that control routing decisions."""
    # Complexity score above this sends the task to the remote model
    complexity_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ROUTER_COMPLEXITY_THRESHOLD", "0.6")
        )
    )
    # If local model confidence is below this, fallback to remote
    confidence_fallback_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("ROUTER_CONFIDENCE_FALLBACK_THRESHOLD", "0.2")
        )
    )


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration — aggregates all sub-configs."""
    fireworks: FireworksConfig = field(default_factory=FireworksConfig)
    local_model: LocalModelConfig = field(default_factory=LocalModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    compression_enabled: bool = field(
        default_factory=lambda: os.getenv("COMPRESSION_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )

    def validate(self) -> list[str]:
        """Return a list of configuration warnings/errors."""
        issues: list[str] = []
        if not self.fireworks.api_key:
            issues.append(
                "FIREWORKS_API_KEY is not set — remote model calls will fail."
            )
        if not (0.0 <= self.router.complexity_threshold <= 1.0):
            issues.append(
                "ROUTER_COMPLEXITY_THRESHOLD must be between 0.0 and 1.0."
            )
        if not (0.0 <= self.router.confidence_fallback_threshold <= 1.0):
            issues.append(
                "ROUTER_CONFIDENCE_FALLBACK_THRESHOLD must be between 0.0 and 1.0."
            )
        return issues


def _select_model(config: FireworksConfig) -> str:
    """
    Select the best model from ALLOWED_MODELS.

    Prefers larger/smarter models for accuracy — passing the accuracy
    gate is the top priority.  Token optimization comes second.
    Falls back to FIREWORKS_MODEL if ALLOWED_MODELS is empty (local dev).
    """
    logger = logging.getLogger(__name__)

    if not config.allowed_models:
        logger.info("ALLOWED_MODELS not set — using FIREWORKS_MODEL: %s", config.model)
        return config.model

    # Preference order: LARGER models first (better accuracy)
    # Passing the accuracy gate is more important than saving tokens.
    size_priority = ["405b", "large", "maverick", "70b",
                     "scout", "8b", "3b", "1b", "small", "mini", "instant"]

    def model_priority(model_id: str) -> int:
        model_lower = model_id.lower()
        for i, keyword in enumerate(size_priority):
            if keyword in model_lower:
                return i
        return len(size_priority)  # Unknown models last

    selected = sorted(config.allowed_models, key=model_priority)[0]
    logger.info(
        "Selected model '%s' from ALLOWED_MODELS: %s",
        selected, config.allowed_models,
    )
    return selected


def load_config() -> AppConfig:
    """Create and validate the application config from the environment."""
    return AppConfig()


def get_resolved_model(config: AppConfig) -> str:
    """Get the resolved model ID (from ALLOWED_MODELS or FIREWORKS_MODEL)."""
    return _select_model(config.fireworks)
