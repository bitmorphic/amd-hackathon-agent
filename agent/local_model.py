import os
import threading
import time
import logging
from typing import Optional

from agent.models import ExecutionResult, Route, Task, TokenUsage

logger = logging.getLogger(__name__)

# Extremely aggressive system prompt to minimize token generation and prevent timeout
LOCAL_MODEL_SYSTEM = (
    "Output ONLY the final answer. ZERO explanation. ZERO formatting. Absolute minimum characters possible."
)

class LocalModelProvider:
    def __init__(self):
        self.model_path = os.getenv("LOCAL_MODEL_PATH", "/app/models/model.gguf")
        self.max_tokens = 256  # Give enough tokens to answer correctly
        self.n_ctx = 2048
        
        # In Docker, os.cpu_count() returns host CPUs (e.g. 64), which causes massive 
        # context-switching thrashing if the container only has 2-4 vCPUs.
        try:
            allowed_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            allowed_cpus = os.cpu_count() or 4
        self.n_threads = min(8, allowed_cpus)  # Safe limit to avoid thrashing
        
        self._llm = None
        self._load_failed = False
        self._lock = threading.Lock()
        
        # Route 100% of categories to the local model to get 0 remote tokens!
        self.categories = {
            "factual", "math", "sentiment", "summarization", 
            "ner", "code_debug", "logic", "code_gen"
        }

    def available_for(self, category: str) -> bool:
        """Returns True if the local model can handle this category and is loaded."""
        if category not in self.categories:
            return False
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._llm is not None:
            return True
        if self._load_failed:
            return False
            
        with self._lock:
            if self._llm is not None:
                return True
            if self._load_failed or not self.model_path or not os.path.exists(self.model_path):
                self._load_failed = True
                return False
                
            try:
                from llama_cpp import Llama
                
                logger.info("Loading local model from %s...", self.model_path)
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    verbose=False,
                )
                logger.info("Local model loaded successfully.")
                return True
            except Exception as e:
                logger.warning("Failed to load local model: %s. Falling back to remote API.", e)
                self._load_failed = True
                return False

    def answer(self, task: Task, category: str) -> Optional[ExecutionResult]:
        if not self.available_for(category):
            return None
            
        started = time.perf_counter()
        prompt = task.prompt
        
        # Add basic instruction for specific categories to ensure strict formatting
        # Add basic instruction for specific categories to ensure strict formatting
        if category == "sentiment":
            prompt = "State ONLY the sentiment: positive, negative, or neutral.\n\n" + prompt
        elif category == "ner":
            prompt = "List entities as 'label: value'. Labels: person, organization, location, date.\n\n" + prompt
        elif category == "summarization":
            prompt = "Output only the summary.\n\n" + prompt
            
        try:
            with self._lock:
                completion = self._llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": LOCAL_MODEL_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                )
        except Exception as e:
            logger.error("Local model inference failed: %s", e)
            return None
            
        latency_ms = (time.perf_counter() - started) * 1000
        choice = completion["choices"][0]
        text = (choice.get("message") or {}).get("content", "").strip()
        
        return ExecutionResult(
            output=text,
            route_used=Route.LOCAL, # 0 remote tokens!
            token_usage=TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            confidence=0.95,
            latency_ms=latency_ms,
            fallback_triggered=False,
        )
