import os
import asyncio
from ai.lightrag_local import OllamaClient
from core.config import DEFAULT_MODEL, EMBEDDING_MODEL

# Separate lightweight model for intent classification.
# llama3.2:1b is ~2x faster than 3b for single-word outputs.
# Set AURA_INTENT_MODEL=llama3.2:3b to reuse the main model.
_INTENT_MODEL = os.getenv("AURA_INTENT_MODEL", "llama3.2:1b")

_SYSTEM = (
    "Classify the user input. "
    "Reply with exactly one word: MOVEMENT for physical commands "
    "(move, go, stop, forward, backward, left, right, turn, walk), "
    "QUESTION for everything else."
)


async def parse_intent(user_msg: str) -> str:
    """Classifies user input as 'MOVEMENT' or 'QUESTION' via local LLM."""
    client = OllamaClient("http://127.0.0.1:11434", EMBEDDING_MODEL, _INTENT_MODEL)
    prompt = f"Input: '{user_msg[:200]}'"
    try:
        res = await client.generate(
            prompt=prompt,
            system=_SYSTEM,
            timeout_s=8.0,
            num_predict=5,    # only need one word
            num_ctx=256,      # short context = fast prefill
            temperature=0.0,  # deterministic
            fast=True,
        )
        return "MOVEMENT" if "MOVEMENT" in res.upper() else "QUESTION"
    except Exception as e:
        print(f"[LLM] Intent parsing fallback: {e}")
        return "QUESTION"
