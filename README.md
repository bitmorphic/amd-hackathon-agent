# Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II — Track 1: General-Purpose AI Agent**

A token-efficient AI agent that handles all 8 capability categories using a hybrid routing architecture that minimizes API token usage while maintaining high answer quality.

## Architecture

```
Input Tasks (/input/tasks.json)
        │
        ▼
  ┌─────────────────┐
  │  Task Router     │ ── Complexity scoring (heuristic)
  └──────┬──────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Rules  │ │ Remote Model │ ── Category-aware prompts
│ (math) │ │ (Fireworks)  │    Dynamic token budgeting
│ 0 tok  │ │              │    Prompt compression
└────────┘ └──────────────┘
    │              │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ Response      │ ── Semantic caching
    │ Cache         │    Dedup identical prompts
    └──────┬───────┘
           ▼
  /output/results.json
```

## Key Optimizations

| Optimization | How it saves tokens |
|---|---|
| **Category-aware system prompts** | Tailored prompts per task type = more accurate, shorter answers |
| **Dynamic token budgeting** | Each task gets the minimum `max_tokens` needed (30-400) |
| **Prompt compression** | Removes redundant whitespace, filler words before API call |
| **Response caching** | Identical prompts served from cache (0 tokens) |
| **Rule-based math** | Simple arithmetic computed locally (0 tokens) |
| **ALLOWED_MODELS selection** | Picks the smallest allowed model for efficiency |

## Capability Categories

| # | Category | Strategy |
|---|----------|----------|
| 1 | Factual knowledge | Remote model with concise system prompt |
| 2 | Mathematical reasoning | Rule-based for simple math, remote for word problems |
| 3 | Sentiment classification | Remote with "label first, then justify" prompt |
| 4 | Text summarisation | Remote with length-constrained prompt |
| 5 | Named entity recognition | Remote with structured extraction prompt |
| 6 | Code debugging | Remote with "identify bug, show fix" prompt |
| 7 | Logical reasoning | Remote with "step-by-step, state conclusion" prompt |
| 8 | Code generation | Remote with "write correct code only" prompt |

## Container Contract

```
Input:   /input/tasks.json  → [{"task_id": "t1", "prompt": "..."}, ...]
Output:  /output/results.json → [{"task_id": "t1", "answer": "..."}, ...]
Exit:    0 on success, non-zero on failure
Runtime: < 10 minutes
```

## Environment Variables

| Variable | Description |
|---|---|
| `FIREWORKS_API_KEY` | Provided by harness at eval time |
| `FIREWORKS_BASE_URL` | Must route ALL API calls through this |
| `ALLOWED_MODELS` | Comma-separated list of permitted model IDs |

## Build & Run

```bash
# Build Docker image
docker build -t amd-routing-agent .

# Run (hackathon evaluation)
docker run \
  -v ./input:/input \
  -v ./output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS=$ALLOWED_MODELS \
  amd-routing-agent

# Local development
pip install -r requirements.txt
INPUT_PATH=input/tasks.json OUTPUT_PATH=output/results.json python run.py
```

## Project Structure

```
├── run.py              # Container entry point (reads /input, writes /output)
├── main.py             # Local dev entry point (CLI)
├── Dockerfile          # Multi-stage build, <1GB image
├── agent/
│   ├── config.py       # Env var config + ALLOWED_MODELS resolver
│   ├── executor.py     # Rule-based + remote executors with category prompts
│   ├── router.py       # Complexity-based routing heuristics
│   ├── budget.py       # Category-aware dynamic token budgets
│   ├── cache.py        # Semantic response caching
│   ├── compressor.py   # Prompt compression (whitespace + filler removal)
│   ├── models.py       # Data models (Task, ExecutionResult, etc.)
│   └── tracker.py      # Token usage tracking
├── eval/
│   ├── test_tasks.json # Test tasks covering all 8 categories
│   └── evaluate.py     # Local evaluation harness
└── requirements.txt    # Python dependencies
```

## Scoring Strategy

1. **Pass the accuracy gate** — category-aware system prompts maximize LLM-Judge scores
2. **Minimize tokens** — tight dynamic budgets + prompt compression + caching
3. **No hardcoded answers** — all factual answers come from the model
4. **No local model dependency** — works with or without GPU
