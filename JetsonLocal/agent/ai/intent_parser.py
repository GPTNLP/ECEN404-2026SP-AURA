import asyncio
from ai.lightrag_local import OllamaClient
from core.config import DEFAULT_MODEL, EMBEDDING_MODEL

async def parse_intent(user_msg: str) -> str:
    """Classifies user input as 'MOVEMENT' or 'QUESTION' via local LLM."""
    client = OllamaClient("http://127.0.0.1:11434", EMBEDDING_MODEL, DEFAULT_MODEL)
    system_prompt = "You are an intent classifier. Classify user input as 'MOVEMENT' or 'QUESTION'. Reply with exactly one word."
    prompt = f"Input: '{user_msg}'"
    try:
        res = await client.generate(prompt, system=system_prompt, timeout_s=5.0)
        return "MOVEMENT" if "MOVEMENT" in res.upper() else "QUESTION"
    except Exception as e:
        print(f"[LLM] Intent parsing fallback due to error: {e}")
        return "QUESTION"