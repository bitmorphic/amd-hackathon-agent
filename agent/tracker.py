"""
Token usage tracker and statistics reporter.

Collects per-task metrics (route chosen, tokens, latency) and produces
a rich summary report showing total scored tokens, routing distribution,
cache performance, and potential savings.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

from agent.models import AgentResponse, Route

logger = logging.getLogger(__name__)

# ANSI color codes (safe on most terminals)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"
_BG_GREEN = "\033[42m"
_BG_RED = "\033[41m"
_BG_YELLOW = "\033[43m"


def _supports_color() -> bool:
    """Check if the terminal supports ANSI colors."""
    if sys.platform == "win32":
        return True  # Modern Windows terminals support ANSI
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# Disable colors if not supported
if not _supports_color():
    _RESET = _BOLD = _DIM = ""
    _GREEN = _YELLOW = _RED = _CYAN = _MAGENTA = _WHITE = ""
    _BG_GREEN = _BG_RED = _BG_YELLOW = ""


@dataclass
class TaskStat:
    """Metrics for a single task execution."""
    task_id: str
    route: Route
    scored_tokens: int
    total_tokens: int
    latency_ms: float
    fallback_triggered: bool
    confidence: float
    token_budget: int = 0
    cached: bool = False


@dataclass
class TrackerSummary:
    """Aggregate statistics across all tasks."""
    total_tasks: int = 0
    local_tasks: int = 0
    remote_tasks: int = 0
    fallback_tasks: int = 0
    cached_tasks: int = 0
    total_scored_tokens: int = 0
    total_local_tokens: int = 0
    total_remote_tokens: int = 0
    avg_latency_ms: float = 0.0
    avg_confidence_local: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "local_tasks": self.local_tasks,
            "remote_tasks": self.remote_tasks,
            "fallback_tasks": self.fallback_tasks,
            "cached_tasks": self.cached_tasks,
            "local_pct": (
                f"{self.local_tasks / self.total_tasks * 100:.1f}%"
                if self.total_tasks else "N/A"
            ),
            "total_scored_tokens": self.total_scored_tokens,
            "total_local_tokens_free": self.total_local_tokens,
            "total_remote_tokens_counted": self.total_remote_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_confidence_local": round(self.avg_confidence_local, 3),
            "cache_hit_rate": f"{self.cache_hit_rate * 100:.1f}%",
        }


class UsageTracker:
    """Accumulates per-task stats and computes aggregate summaries."""

    def __init__(self) -> None:
        self._stats: list[TaskStat] = []

    def record(
        self,
        response: AgentResponse,
        token_budget: int = 0,
        cached: bool = False,
    ) -> None:
        """Record metrics from a completed agent response."""
        stat = TaskStat(
            task_id=response.task_id,
            route=response.result.route_used,
            scored_tokens=response.scored_tokens,
            total_tokens=response.result.token_usage.total_tokens,
            latency_ms=response.result.latency_ms,
            fallback_triggered=response.result.fallback_triggered,
            confidence=response.result.confidence,
            token_budget=token_budget,
            cached=cached,
        )
        self._stats.append(stat)

        # Live task log
        route_color = _GREEN if stat.route == Route.LOCAL else _YELLOW
        cache_tag = f" {_CYAN}[CACHED]{_RESET}" if stat.cached else ""
        fallback_tag = f" {_RED}[FALLBACK]{_RESET}" if stat.fallback_triggered else ""

        print(
            f"  {_DIM}{'>'}{_RESET} {_WHITE}{stat.task_id:<16}{_RESET} "
            f"{route_color}{stat.route.value:<8}{_RESET} "
            f"scored={_BOLD}{stat.scored_tokens:<6}{_RESET} "
            f"latency={stat.latency_ms:>7.0f}ms "
            f"conf={stat.confidence:.2f}"
            f"{cache_tag}{fallback_tag}",
            flush=True,
        )

    def summarize(self, cache_hit_rate: float = 0.0) -> TrackerSummary:
        """Compute aggregate statistics."""
        if not self._stats:
            return TrackerSummary()

        local_stats = [s for s in self._stats if s.route == Route.LOCAL]
        remote_stats = [s for s in self._stats if s.route == Route.REMOTE]
        fallback_stats = [s for s in self._stats if s.fallback_triggered]
        cached_stats = [s for s in self._stats if s.cached]

        total_latency = sum(s.latency_ms for s in self._stats)
        local_confidences = [
            s.confidence for s in local_stats if s.confidence > 0
        ]

        return TrackerSummary(
            total_tasks=len(self._stats),
            local_tasks=len(local_stats),
            remote_tasks=len(remote_stats),
            fallback_tasks=len(fallback_stats),
            cached_tasks=len(cached_stats),
            total_scored_tokens=sum(s.scored_tokens for s in self._stats),
            total_local_tokens=sum(s.total_tokens for s in local_stats),
            total_remote_tokens=sum(s.total_tokens for s in remote_stats),
            avg_latency_ms=total_latency / len(self._stats),
            avg_confidence_local=(
                sum(local_confidences) / len(local_confidences)
                if local_confidences
                else 0.0
            ),
            cache_hit_rate=cache_hit_rate,
        )

    def print_report(self, cache_hit_rate: float = 0.0) -> None:
        """Print a rich, colored summary to stdout."""
        summary = self.summarize(cache_hit_rate)
        data = summary.to_dict()

        # Token savings estimate (vs all-remote baseline)
        # Assume average remote call uses ~200 tokens
        baseline_estimate = summary.total_tasks * 200
        saved = baseline_estimate - summary.total_scored_tokens
        savings_pct = (saved / baseline_estimate * 100) if baseline_estimate else 0

        print()
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print(f"  {_BOLD}{_WHITE}  HYBRID ROUTING AGENT — EXECUTION REPORT{_RESET}")
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print()

        # Routing distribution bar
        local_pct = summary.local_tasks / summary.total_tasks * 100 if summary.total_tasks else 0
        remote_pct = 100 - local_pct
        bar_width = 40
        local_bar = int(bar_width * local_pct / 100)
        remote_bar = bar_width - local_bar

        print(f"  {_BOLD}Routing Distribution{_RESET}")
        print(
            f"  {_BG_GREEN}{_WHITE}{' ' * local_bar}{_RESET}"
            f"{_BG_RED}{_WHITE}{' ' * remote_bar}{_RESET}"
            f"  {_GREEN}LOCAL {local_pct:.0f}%{_RESET} | "
            f"{_YELLOW}REMOTE {remote_pct:.0f}%{_RESET}"
        )
        print()

        # Key metrics
        metrics = [
            ("Total Tasks", str(summary.total_tasks), _WHITE),
            ("Local (free)", str(summary.local_tasks), _GREEN),
            ("Remote (counted)", str(summary.remote_tasks), _YELLOW),
            ("Fallbacks", str(summary.fallback_tasks), _RED),
            ("Cache Hits", str(summary.cached_tasks), _CYAN),
            ("", "", ""),  # spacer
            ("Scored Tokens", str(summary.total_scored_tokens), _BOLD + _WHITE),
            ("Free Local Tokens", str(summary.total_local_tokens), _GREEN),
            ("Estimated Savings", f"~{savings_pct:.0f}% vs all-remote", _GREEN),
            ("", "", ""),  # spacer
            ("Avg Latency", f"{summary.avg_latency_ms:.0f}ms", _WHITE),
            ("Cache Hit Rate", data["cache_hit_rate"], _CYAN),
        ]

        for label, value, color in metrics:
            if not label:
                print()
                continue
            print(f"  {_DIM}{label:<28}{_RESET} {color}{value}{_RESET}")

        print()
        print(f"  {_BOLD}{_CYAN}{'=' * 58}{_RESET}")
        print()

    def export_json(self) -> str:
        """Export all per-task stats as JSON."""
        return json.dumps(
            [
                {
                    "task_id": s.task_id,
                    "route": s.route.value,
                    "scored_tokens": s.scored_tokens,
                    "total_tokens": s.total_tokens,
                    "latency_ms": round(s.latency_ms, 1),
                    "fallback_triggered": s.fallback_triggered,
                    "confidence": round(s.confidence, 3),
                    "token_budget": s.token_budget,
                    "cached": s.cached,
                }
                for s in self._stats
            ],
            indent=2,
        )
