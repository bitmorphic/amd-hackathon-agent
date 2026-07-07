"""
Hybrid Token-Efficient Routing Agent — Entry Point

Reads tasks from a JSON file (or stdin), runs each through the
optimized router → executor pipeline with caching, compression,
dynamic budgeting, and cascading fallback.

Usage:
    python main.py --tasks tasks.json
    python main.py --tasks tasks.json --output results.json
    echo '{"id":"1","prompt":"Hello"}' | python main.py --stdin
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from agent.budget import estimate_token_budget
from agent.config import load_config
from agent.executor import HybridExecutor
from agent.models import AgentResponse, Task
from agent.router import HeuristicRouter
from agent.tracker import UsageTracker

# ANSI codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_WHITE = "\033[97m"
_MAGENTA = "\033[95m"


def setup_logging(level: str) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner() -> None:
    """Print a startup banner."""
    print()
    print(f"  {_BOLD}{_CYAN}╔══════════════════════════════════════════════════════╗{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_BOLD}{_WHITE}Hybrid Token-Efficient Routing Agent{_RESET}                {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_DIM}AMD Developer Hackathon: ACT II — Track 1{_RESET}          {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}╠══════════════════════════════════════════════════════╣{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_GREEN}✓{_RESET} Heuristic router     {_DIM}(zero overhead){_RESET}            {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_GREEN}✓{_RESET} Response caching     {_DIM}(identical = free){_RESET}         {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_GREEN}✓{_RESET} Prompt compression   {_DIM}(fewer remote tokens){_RESET}      {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_GREEN}✓{_RESET} Dynamic budgeting    {_DIM}(tight max_tokens){_RESET}         {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}║{_RESET}  {_GREEN}✓{_RESET} Cascading fallback   {_DIM}(local → verify → remote){_RESET} {_BOLD}{_CYAN}║{_RESET}")
    print(f"  {_BOLD}{_CYAN}╚══════════════════════════════════════════════════════╝{_RESET}")
    print()


def load_tasks(source: str | None, use_stdin: bool) -> list[Task]:
    """Load tasks from a JSON file or stdin."""
    if use_stdin:
        raw = sys.stdin.read().strip()
    elif source:
        raw = Path(source).read_text(encoding="utf-8")
    else:
        print("Error: provide --tasks <file> or --stdin", file=sys.stderr)
        sys.exit(1)

    data = json.loads(raw)

    # Support both a single task object and a list of tasks
    if isinstance(data, dict):
        data = [data]

    tasks = []
    for i, item in enumerate(data):
        if "id" not in item:
            item["id"] = str(i + 1)
        tasks.append(Task(**item))

    return tasks


def run_agent(tasks: list[Task]) -> tuple[list[AgentResponse], UsageTracker, float]:
    """Run the full agent pipeline on a list of tasks."""
    config = load_config()

    # Log config warnings
    issues = config.validate()
    for issue in issues:
        logging.warning("Config: %s", issue)

    router = HeuristicRouter(config.router)
    executor = HybridExecutor(config)
    tracker = UsageTracker()

    responses: list[AgentResponse] = []

    print(f"  {_BOLD}Processing {len(tasks)} task(s)...{_RESET}")
    print(f"  {_DIM}{'─' * 75}{_RESET}")

    for task in tasks:
        # Step 1: Route
        decision = router.route(task)

        # Step 2: Compute dynamic token budget
        token_budget = estimate_token_budget(task, decision)

        # Step 3: Execute (with caching, compression, verification, fallback)
        result = executor.execute(task, decision)

        # Step 4: Package response
        response = AgentResponse(
            task_id=task.id,
            routing=decision,
            result=result,
        )
        responses.append(response)

        # Step 5: Track (with cache awareness)
        cached = result.token_usage.total_tokens == 0 and result.latency_ms == 0
        tracker.record(response, token_budget=token_budget, cached=cached)

    print(f"  {_DIM}{'─' * 75}{_RESET}")

    # Get cache hit rate
    cache_hit_rate = executor.cache.stats.hit_rate

    return responses, tracker, cache_hit_rate


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hybrid Token-Efficient Routing Agent",
    )
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Path to a JSON file containing tasks",
    )
    parser.add_argument(
        "--stdin", action="store_true",
        help="Read tasks from stdin (JSON)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write results to this JSON file (default: stdout)",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress banner and live output (JSON only)",
    )
    args = parser.parse_args()

    # Setup
    config = load_config()
    setup_logging(args.log_level or config.log_level)

    if not args.quiet:
        print_banner()

    # Load tasks
    tasks = load_tasks(args.tasks, args.stdin)

    if not args.quiet:
        print(f"  {_MAGENTA}Loaded {len(tasks)} task(s){_RESET}")
        print()

    # Run
    responses, tracker, cache_hit_rate = run_agent(tasks)

    # Output results
    results_data = [resp.model_dump(mode="json") for resp in responses]
    results_json = json.dumps(results_data, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(results_json, encoding="utf-8")
        if not args.quiet:
            print(f"\n  {_GREEN}Results written to {args.output}{_RESET}")
    elif args.quiet:
        print(results_json)

    # Print summary report
    if not args.quiet:
        tracker.print_report(cache_hit_rate=cache_hit_rate)


if __name__ == "__main__":
    main()
