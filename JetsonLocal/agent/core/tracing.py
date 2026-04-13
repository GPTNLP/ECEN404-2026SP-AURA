"""
LangSmith tracing for AURA.

Captures token usage and output speed for every Ollama LLM call.

Enable by adding to JetsonLocal/.env:
  LANGCHAIN_TRACING_V2=true
  LANGSMITH_API_KEY=<your key from smith.langchain.com>
  LANGCHAIN_PROJECT=AURA          # optional — groups all runs under one project

When disabled (default), this module is a complete no-op with zero overhead.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

# ─── Feature flag ─────────────────────────────────────────────────────────────
# Recognised by both legacy (LANGCHAIN_TRACING_V2) and new (LANGSMITH_TRACING)
# environment variable names.
TRACING_ENABLED: bool = (
    os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    or os.getenv("LANGSMITH_TRACING", "").lower() == "true"
)

# Optional: override the LangSmith project name (defaults to env var or "AURA")
LS_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "AURA"))

# ─── Lazy client ──────────────────────────────────────────────────────────────
_ls_client = None


def _client():
    """Return a cached LangSmith Client, or None if unavailable."""
    global _ls_client
    if _ls_client is not None:
        return _ls_client
    if not TRACING_ENABLED:
        return None
    try:
        from langsmith import Client  # type: ignore
        _ls_client = Client()
        return _ls_client
    except Exception as exc:
        print(f"[LangSmith] Client init failed — tracing disabled: {exc}")
        # Flip the flag so we don't retry on every call
        global TRACING_ENABLED
        TRACING_ENABLED = False
        return None


# ─── OllamaTrace context object ───────────────────────────────────────────────

class OllamaTrace:
    """
    Lightweight context object for a single Ollama /api/generate call.

    Usage
    ─────
    trace = OllamaTrace.start_llm(model="llama3.2:3b", prompt=p, system=s)
    raw_response = <call ollama>
    trace.finish(raw_response)          # extracts token counts + speed and sends to LangSmith
    """

    __slots__ = ("_run_tree", "_started_at")

    def __init__(self, run_tree, started_at: float):
        self._run_tree = run_tree
        self._started_at = started_at

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def start_llm(
        cls,
        *,
        model: str,
        prompt: str,
        system: str,
        call_type: str = "answer",   # "answer" | "extract" | "keywords" | "intent"
    ) -> "OllamaTrace":
        """Open a new LangSmith LLM run.  Returns a no-op trace if disabled."""
        started_at = time.monotonic()
        if not TRACING_ENABLED:
            return cls(None, started_at)

        rt = None
        try:
            from langsmith.run_trees import RunTree  # type: ignore
            rt = RunTree(
                name=f"ollama/{model}",
                run_type="llm",
                project_name=LS_PROJECT,
                inputs={
                    "model":     model,
                    "call_type": call_type,
                    "system":    system[:500] if system else "",
                    "prompt":    prompt[:2000],
                },
            )
            rt.post()
        except Exception as exc:
            print(f"[LangSmith] Failed to open run: {exc}")
            rt = None

        return cls(rt, started_at)

    @classmethod
    def start_embed(
        cls,
        *,
        model: str,
        batch_size: int,
    ) -> "OllamaTrace":
        """Open a LangSmith 'embedding' run for a batch embed call."""
        started_at = time.monotonic()
        if not TRACING_ENABLED:
            return cls(None, started_at)

        rt = None
        try:
            from langsmith.run_trees import RunTree  # type: ignore
            rt = RunTree(
                name=f"ollama/embed/{model}",
                run_type="embedding",
                project_name=LS_PROJECT,
                inputs={"model": model, "batch_size": batch_size},
            )
            rt.post()
        except Exception as exc:
            print(f"[LangSmith] Failed to open embed run: {exc}")
            rt = None

        return cls(rt, started_at)

    # ── finish ────────────────────────────────────────────────────────────────

    def finish(self, ollama_response: Dict[str, Any], response_text: str = "") -> None:
        """
        Close the LangSmith run and attach Ollama's built-in metrics.

        Ollama /api/generate fields used:
          prompt_eval_count    — input tokens
          eval_count           — output tokens
          eval_duration        — generation time in nanoseconds
          prompt_eval_duration — prompt processing time in nanoseconds
          total_duration       — wall-clock total in nanoseconds
        """
        if self._run_tree is None:
            return

        try:
            tokens_in   = int(ollama_response.get("prompt_eval_count", 0))
            tokens_out  = int(ollama_response.get("eval_count", 0))
            eval_ns     = int(ollama_response.get("eval_duration", 0))
            prompt_ns   = int(ollama_response.get("prompt_eval_duration", 0))
            total_ns    = int(ollama_response.get("total_duration", 0))

            eval_s      = eval_ns   / 1e9 if eval_ns   > 0 else 0.0
            prompt_s    = prompt_ns / 1e9 if prompt_ns > 0 else 0.0
            total_s     = total_ns  / 1e9 if total_ns  > 0 else time.monotonic() - self._started_at

            tokens_per_sec = round(tokens_out / eval_s, 2) if eval_s > 0 else 0.0

            self._run_tree.end(
                outputs={
                    "response": response_text[:4000],
                    "token_usage": {
                        "prompt_tokens":     tokens_in,
                        "completion_tokens": tokens_out,
                        "total_tokens":      tokens_in + tokens_out,
                    },
                    "performance": {
                        "tokens_per_second":    tokens_per_sec,
                        "generation_time_s":    round(eval_s, 3),
                        "prompt_proc_time_s":   round(prompt_s, 3),
                        "total_latency_s":      round(total_s, 3),
                    },
                }
            )
            self._run_tree.patch()
        except Exception as exc:
            print(f"[LangSmith] Failed to close run: {exc}")

    def finish_embed(self, batch_size: int, duration_s: float) -> None:
        """Close an embedding run."""
        if self._run_tree is None:
            return
        try:
            self._run_tree.end(
                outputs={
                    "texts_embedded": batch_size,
                    "duration_s":     round(duration_s, 3),
                }
            )
            self._run_tree.patch()
        except Exception as exc:
            print(f"[LangSmith] Failed to close embed run: {exc}")

    def error(self, exc: Exception) -> None:
        """Mark the run as failed."""
        if self._run_tree is None:
            return
        try:
            self._run_tree.end(error=str(exc))
            self._run_tree.patch()
        except Exception:
            pass
