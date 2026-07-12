<div align="center">
  <img src="assets/banner.png" alt="AMD Developer Hackathon" width="80%">
  <br><br>
  <img src="assets/tera_icon_transparent.png" alt="TERA Logo" width="180">
  <br><br>
  
  <h3>TERA (Token-Efficient Routing Agent)</h3>
  <b>AMD Developer Hackathon: ACT II — Track 1: Hybrid Token-Efficient Routing Agent</b>
  <br><br>

  <img src="https://img.shields.io/badge/AMD-000000?style=for-the-badge&logo=amd&logoColor=white" alt="AMD" />
  <img src="https://img.shields.io/badge/Google_DeepMind-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google DeepMind" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Fireworks_AI-FF6B6B?style=for-the-badge&logo=fireworks&logoColor=white" alt="Fireworks AI" />
  <img src="https://img.shields.io/badge/Llama_3-0466C8?style=for-the-badge&logo=meta&logoColor=white" alt="Llama 3" />
</div>

<br/>

An ultra-fast, token-efficient AI agent designed to conquer all 8 hackathon capability categories. TERA uses a **Regex-Based Classifier** and **Smart Model Tiering** to achieve maximum accuracy while mathematically minimizing token costs.

## 🔥 The Winning Architecture

To achieve an 85%+ accuracy score while maintaining the lowest possible token usage, TERA relies on a 5-pillar architecture:

### 1. ⚡ 8-Worker Concurrency
The container utilizes a `ThreadPoolExecutor` with 8 parallel workers. This allows TERA to process all tasks in the harness in < 15 seconds, completely eliminating timeout crashes.

### 2. 🧠 Comprehensive Regex Classifier
Instead of blindly sending prompts to an LLM, TERA runs a lightning-fast regex pass (80+ patterns) over every prompt to classify it into one of the 8 hackathon tracks (e.g., `math`, `code_gen`, `sentiment`). 

### 3. 🎯 Smart Model Tiering
TERA dynamically analyzes the injected `ALLOWED_MODELS` list at runtime and sorts them into tiers:
- **`cheap` tier** (Smallest model) → Used for easy tasks (Sentiment, NER, Summarization)
- **`strong` tier** (Largest model) → Used for hard tasks (Math, Logic, Factual)
- **`code` tier** (Code specific) → Used for Code Gen & Debugging

*Result: We use maximum intelligence only when needed, saving thousands of tokens on easier tasks.*

### 4. 📝 Judge-Aligned System Prompts & Budgets
Once classified, the task is given a highly specific system prompt designed exactly for the Hackathon LLM-Judge. 
- *Example (Logic):* "Deduce in brief numbered steps, then 'Answer: <value>' on its own line."
Each category also gets a strict `max_tokens` budget to prevent runaway generation.

### 5. 🛡️ Fallback Safety Net
If the primary model API fails or returns a blank string, TERA automatically catches the error and retries the prompt using the `strong` model tier, ensuring an answer is always provided.

## 🚀 Execution Flow

```text
Input Tasks (/input/tasks.json)
        │
        ▼ (8 Concurrent Workers)
  ┌───────────────────────┐
  │ 1. Response Cache     │ ── Instant 0-token answers for duplicates
  └──────┬────────────────┘
         │
  ┌──────▼────────────────┐
  │ 2. Regex Classifier   │ ── Classifies into 1 of 8 tracks
  └──────┬────────────────┘
         │
  ┌──────▼────────────────┐
  │ 3. Rule-Based Math    │ ── Solves simple arithmetic locally (0 tokens)
  └──────┬────────────────┘
         │
  ┌──────▼───────────────────────────┐
  │ 4. Smart Tiered Remote Execution │ ── cheap / strong / code models
  └──────┬───────────────────────────┘    + Category system prompts
         │
  ┌──────▼────────────────┐
  │ 5. Fallback Mechanism │ ── Retries on blank/failed answers
  └──────┬────────────────┘
         │
  /output/results.json
```

## 🛠️ Build & Run

```bash
# Build Docker image
docker build -t amd-routing-agent .

# Run (Simulating the Hackathon Evaluation Harness)
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY \
  -e FIREWORKS_BASE_URL=$FIREWORKS_BASE_URL \
  -e ALLOWED_MODELS=$ALLOWED_MODELS \
  amd-routing-agent

# Local fast testing
pip install -r requirements.txt
export INPUT_PATH=eval/test_tasks.json
export OUTPUT_PATH=eval/test_results.json
python run.py
```
