"""
Local evaluation harness for the Hybrid Token-Efficient Routing Agent.

Runs the agent against test tasks with known answers, measures accuracy
and token usage, and outputs a detailed report with per-task breakdowns.

Usage:
    python -m eval.evaluate --tasks eval/test_tasks.json
    python -m eval.evaluate --tasks eval/test_tasks.json --report results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path so we can import agent modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.budget import estimate_token_budget
from agent.config import load_config
from agent.executor import HybridExecutor
from agent.models import AgentResponse, Route, Task
from agent.router import HeuristicRouter
from agent.tracker import UsageTracker

# ANSI
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    import re as _re
    text = text.lower().strip()
    # Remove common punctuation (keep alphanumeric and spaces)
    text = _re.sub(r"[^\w\s]", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text


# Common number words for normalization
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def _normalize_numbers(text: str) -> str:
    """Replace number words with digits for comparison."""
    # Handle compound numbers like "twenty-two" → "22"
    import re as _re
    text = _re.sub(r"twenty[- ](\w+)", lambda m: str(20 + int(_NUMBER_WORDS.get(m.group(1), "0"))), text)
    text = _re.sub(r"thirty[- ](\w+)", lambda m: str(30 + int(_NUMBER_WORDS.get(m.group(1), "0"))), text)
    text = _re.sub(r"forty[- ](\w+)", lambda m: str(40 + int(_NUMBER_WORDS.get(m.group(1), "0"))), text)
    text = _re.sub(r"fifty[- ](\w+)", lambda m: str(50 + int(_NUMBER_WORDS.get(m.group(1), "0"))), text)
    for word, digit in _NUMBER_WORDS.items():
        text = text.replace(word, digit)
    return text


def simple_accuracy_check(output: str, expected: str) -> float:
    """
    Basic accuracy check: does the output contain the expected answer?

    Returns 1.0 for match, 0.0 for miss.  This is a rough heuristic —
    for real evaluation you'd want an LLM-as-judge or exact-match
    depending on the task type.
    """
    output_norm = _normalize_text(output)
    expected_norm = _normalize_text(expected)

    # Exact containment match (normalized)
    if expected_norm in output_norm:
        return 1.0

    # Number-normalized containment
    output_nums = _normalize_numbers(output_norm)
    expected_nums = _normalize_numbers(expected_norm)
    if expected_nums in output_nums:
        return 1.0

    # Check if individual key words are present
    expected_words = set(expected_norm.split())
    output_words = set(output_norm.split())
    # Remove common stop words for cleaner matching
    stop_words = {"a", "an", "the", "is", "are", "was", "were", "of", "in", "to", "and", "or", "for", "it", "that"}
    expected_words -= stop_words
    output_words -= stop_words
    if len(expected_words) > 0:
        overlap = len(expected_words & output_words) / len(expected_words)
        if overlap >= 0.6:
            return overlap

    return 0.0


def run_evaluation(tasks_path: str) -> dict:
    """Run the full evaluation pipeline."""
    config = load_config()
    router = HeuristicRouter(config.router)
    executor = HybridExecutor(config)
    tracker = UsageTracker()

    # Load tasks
    raw = Path(tasks_path).read_text(encoding="utf-8")
    task_data = json.loads(raw)
    tasks = [Task(**t) for t in task_data]

    results = []
    correct = 0
    total_with_answers = 0

    print(f"\n  {_BOLD}{_CYAN}EVALUATION HARNESS{_RESET}")
    print(f"  {_DIM}{'─' * 60}{_RESET}")

    for task in tasks:
        # Route
        decision = router.route(task)

        # Budget
        token_budget = estimate_token_budget(task, decision)

        # Execute
        result = executor.execute(task, decision)

        # Package response
        response = AgentResponse(
            task_id=task.id,
            routing=decision,
            result=result,
        )

        cached = result.token_usage.total_tokens == 0 and result.latency_ms == 0
        tracker.record(response, token_budget=token_budget, cached=cached)

        # Check accuracy
        accuracy = 0.0
        if task.expected_answer:
            total_with_answers += 1
            accuracy = simple_accuracy_check(result.output, task.expected_answer)
            if accuracy >= 0.7:
                correct += 1

        results.append({
            "task_id": task.id,
            "prompt_preview": task.prompt[:80] + ("..." if len(task.prompt) > 80 else ""),
            "route": decision.route.value,
            "complexity_score": decision.complexity_score,
            "difficulty": decision.difficulty.value,
            "token_budget": token_budget,
            "scored_tokens": response.scored_tokens,
            "confidence": result.confidence,
            "accuracy": round(accuracy, 2),
            "fallback": result.fallback_triggered,
            "output_preview": result.output[:120] + ("..." if len(result.output) > 120 else ""),
        })

    # Summary
    cache_hit_rate = executor.cache.stats.hit_rate
    summary = tracker.summarize(cache_hit_rate)
    overall_accuracy = correct / total_with_answers if total_with_answers else 0.0

    report = {
        "summary": {
            **summary.to_dict(),
            "accuracy": f"{overall_accuracy * 100:.1f}%",
            "correct": correct,
            "total_with_answers": total_with_answers,
        },
        "tasks": results,
    }

    return report


def print_eval_report(report: dict) -> None:
    """Print a detailed evaluation report."""
    summary = report["summary"]

    print(f"\n  {_BOLD}{_WHITE}--- Evaluation Summary ---{_RESET}")

    accuracy_color = _GREEN if float(summary["accuracy"].rstrip("%")) >= 70 else _RED
    print(f"  {_BOLD}Accuracy:{_RESET}          {accuracy_color}{summary['accuracy']}{_RESET} ({summary['correct']}/{summary['total_with_answers']})")
    print(f"  {_BOLD}Scored Tokens:{_RESET}     {_YELLOW}{summary['total_scored_tokens']}{_RESET}")
    print(f"  {_BOLD}Local (free):{_RESET}      {_GREEN}{summary['local_tasks']}{_RESET} tasks")
    print(f"  {_BOLD}Remote (counted):{_RESET}  {_YELLOW}{summary['remote_tasks']}{_RESET} tasks")
    print(f"  {_BOLD}Fallbacks:{_RESET}         {_RED}{summary['fallback_tasks']}{_RESET}")
    print(f"  {_BOLD}Cache Hits:{_RESET}        {_CYAN}{summary['cached_tasks']}{_RESET}")

    print(f"\n  {_BOLD}{_WHITE}--- Per-Task Results ---{_RESET}")
    print(f"  {_DIM}{'Task':<16} {'Route':<8} {'Score':>6} {'Budget':>7} {'Tokens':>7} {'Acc':>5} {'Status'}{_RESET}")
    print(f"  {_DIM}{'─' * 65}{_RESET}")

    for t in report["tasks"]:
        status = f"{_GREEN}PASS{_RESET}" if t["accuracy"] >= 0.7 else f"{_RED}FAIL{_RESET}"
        route_color = _GREEN if t["route"] == "local" else _YELLOW
        fallback_tag = f" {_RED}FB{_RESET}" if t["fallback"] else ""

        print(
            f"  {_WHITE}{t['task_id']:<16}{_RESET} "
            f"{route_color}{t['route']:<8}{_RESET} "
            f"{t['complexity_score']:>5.2f}  "
            f"{t['token_budget']:>6}  "
            f"{t['scored_tokens']:>6}  "
            f"{t['accuracy']:>4.1f}  "
            f"{status}{fallback_tag}"
        )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Hybrid Routing Agent",
    )
    parser.add_argument(
        "--tasks", type=str, required=True,
        help="Path to test tasks JSON file",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Write evaluation report to this JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_evaluation(args.tasks)
    print_eval_report(report)

    # Save report if requested
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"  {_GREEN}Report saved to {args.report}{_RESET}\n")


if __name__ == "__main__":
    main()
