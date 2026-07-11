"""
Container entry point for the AMD Hackathon submission.

Reads tasks from /input/tasks.json, processes them through the
optimized hybrid agent, and writes results to /output/results.json.

This is the ENTRYPOINT for the Docker container used in evaluation.
For local development, use main.py instead.

Container contract:
  - Input:  /input/tasks.json  → [{"task_id": "t1", "prompt": "..."}, ...]
  - Output: /output/results.json → [{"task_id": "t1", "answer": "..."}, ...]
  - Exit code 0 on success, non-zero on failure
  - Maximum runtime: 10 minutes
  - Env vars FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
    are injected by the evaluation harness at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = os.getenv("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "/output/results.json")
MAX_RUNTIME_SECONDS = 330  # 5.5 min safety margin (evaluator kills at 6 min)


def setup_logging() -> None:
    """Configure structured logging."""
    level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,  # Keep stdout clean for output
    )


def main() -> int:
    """Main container entrypoint. Returns exit code."""
    setup_logging()
    logger = logging.getLogger("run")

    start_time = time.monotonic()

    # ── Step 1: Load configuration ──
    try:
        from agent.config import load_config, get_resolved_model
        from agent.executor import HybridExecutor
        from agent.models import Task
        from agent.router import HeuristicRouter
        from agent.budget import estimate_token_budget

        config = load_config()
        resolved_model = get_resolved_model(config)
        logger.info("Config loaded. Model: %s", resolved_model)
        logger.info("Base URL: %s", config.fireworks.base_url)
        logger.info("ALLOWED_MODELS: %s", config.fireworks.allowed_models or "(not set)")
        logger.info("Cache: %s | Compression: %s",
                     config.cache_enabled, config.compression_enabled)
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # ── Step 2: Read input tasks ──
    try:
        input_path = Path(INPUT_PATH)
        logger.info("Reading tasks from %s", input_path)

        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return 1

        raw = json.loads(input_path.read_text(encoding="utf-8"))

        # Normalize task format: support various key names
        tasks = []
        for item in raw:
            task_id = (item.get("task_id") or item.get("id")
                       or item.get("taskId") or item.get("task") or "")
            prompt = (item.get("prompt") or item.get("question")
                      or item.get("input") or item.get("text")
                      or item.get("query") or "")
            if task_id and prompt:
                tasks.append(Task(id=str(task_id), prompt=prompt))

        logger.info("Loaded %d tasks", len(tasks))
        if not tasks:
            logger.error("No valid tasks found in input file")
            return 1

    except Exception as exc:
        logger.error("Failed to read input tasks: %s", exc)
        return 1

    # ── Step 3: Initialize pipeline ──
    try:
        router = HeuristicRouter(config.router)
        executor = HybridExecutor(config)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc)
        return 1

    # ── Step 4: Process tasks ──
    from concurrent.futures import ThreadPoolExecutor

    import threading
    
    deadline = start_time + MAX_RUNTIME_SECONDS
    results = []
    
    total_tokens = 0
    token_lock = threading.Lock()

    def _process_task(task: Task):
        nonlocal total_tokens
        try:
            decision = router.route(task)
            result = executor.execute(task, decision)
            
            with token_lock:
                total_tokens += result.token_usage.total_tokens
                
            logger.info(
                "Task %s: route=%s tokens=%d latency=%.0fms category=%s",
                task.id,
                result.route_used.value,
                result.token_usage.total_tokens,
                result.latency_ms,
                decision.difficulty.value,
            )
            return {"task_id": task.id, "answer": result.output}
        except Exception as exc:
            logger.error("Task %s failed: %s", task.id, exc)
            return {"task_id": task.id, "answer": ""}

    pool = ThreadPoolExecutor(max_workers=8)
    futures = [pool.submit(_process_task, task) for task in tasks]

    for i, fut in enumerate(futures):
        try:
            # Ensure we leave enough time to write results.json
            timeout = max(0.1, deadline - time.monotonic())
            results.append(fut.result(timeout=timeout))
        except Exception as exc:
            logger.error("Task %s timed out or failed: %s", tasks[i].id, exc)
            results.append({"task_id": tasks[i].id, "answer": ""})
            
    # Do NOT wait for stuck threads, just cancel pending and move on so we can write output
    pool.shutdown(wait=False, cancel_futures=True)

    # ── Step 5: Write output ──
    try:
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Results written to %s (%d tasks)", output_path, len(results))
    except Exception as exc:
        logger.error("Failed to write output: %s", exc)
        return 1

    # ── Summary ──
    total_elapsed = time.monotonic() - start_time
    if hasattr(executor, 'cache'):
        logger.info("Cache hits: %d", executor.cache.stats.hits)
    logger.info("Total runtime: %.1fs | Tasks: %d | Total Tokens Used: %d", total_elapsed, len(results), total_tokens)

    return 0


if __name__ == "__main__":
    sys.exit(main())
