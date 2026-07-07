"""
Data models and schemas for the Hybrid Token-Efficient Routing Agent.

All structured data flows through these types, keeping the rest of the
codebase decoupled from serialization details.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Route(str, Enum):
    """Where a task gets executed."""
    LOCAL = "local"
    REMOTE = "remote"


class TaskDifficulty(str, Enum):
    """Coarse difficulty bucket used by the router."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """A single task the agent must complete."""
    id: str = Field(..., description="Unique task identifier")
    prompt: str = Field(..., description="The user prompt / instruction")
    expected_answer: Optional[str] = Field(
        None, description="Ground-truth answer (for evaluation only)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to the task",
    )


# ---------------------------------------------------------------------------
# Router output
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    """The router's verdict on where to send a task."""
    route: Route
    complexity_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Estimated complexity (0 = trivial, 1 = very hard)",
    )
    difficulty: TaskDifficulty
    reason: str = Field(
        ..., description="Human-readable explanation of the routing decision"
    )
    signals: dict[str, Any] = Field(
        default_factory=dict,
        description="Individual signal scores that fed into the decision",
    )


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Token counts for a single model call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def scored_tokens(self) -> int:
        """Tokens that count toward the hackathon score.
        
        Local tokens = 0.  Remote tokens = total_tokens.
        The caller is responsible for checking the route before summing.
        """
        return self.total_tokens


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

class ExecutionResult(BaseModel):
    """Result from running a task through a model (local or remote)."""
    output: str = Field(..., description="Model-generated answer")
    route_used: Route
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Self-assessed confidence in the answer (local model only)",
    )
    latency_ms: float = Field(
        0.0, description="Wall-clock time for the model call in milliseconds"
    )
    fallback_triggered: bool = Field(
        False,
        description="True if this result came from a confidence-based fallback",
    )


# ---------------------------------------------------------------------------
# Agent-level response (aggregates routing + execution)
# ---------------------------------------------------------------------------

class AgentResponse(BaseModel):
    """Complete response for a single task — routing decision + execution."""
    task_id: str
    routing: RoutingDecision
    result: ExecutionResult
    timestamp: float = Field(default_factory=time.time)

    @property
    def scored_tokens(self) -> int:
        """Tokens that count toward the final hackathon score."""
        if self.result.route_used == Route.LOCAL:
            return 0
        return self.result.token_usage.total_tokens
