"""
LightRAG - Graph-enhanced RAG for AURA.

Based on: LightRAG: Simple and Fast Retrieval-Augmented Generation
(Guo et al., 2024)  https://arxiv.org/abs/2410.05779

Key differences from naive chunk-vector RAG (as described in paper):
  1. Graph-based indexing: LLM extracts entities + relationships during insert
  2. Dual-level retrieval:
       Low-level  — entity name matching → entity descriptions + 1-hop relations
       High-level — vector search on entity description embeddings
  3. Hybrid fusion: graph context + chunk BM25 fed together into the LLM
  4. GPU acceleration: num_gpu=99 on every Ollama call, batch /api/embed endpoint

Backward compatible: existing databases (meta.json / faiss.index) still load
and work; graph enrichment is additive and stored in graph.json.
"""

from __future__ import annotations

import os
import re
import json
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

import faiss
from rank_bm25 import BM25Okapi

# ─── Tunables ────────────────────────────────────────────────────────────────

# Ollama connectivity
AURA_OLLAMA_URL       = os.getenv("AURA_OLLAMA_URL", "http://127.0.0.1:11434")
AURA_OLLAMA_TIMEOUT_S = float(os.getenv("AURA_OLLAMA_TIMEOUT_S", "180"))
# 2h keep-alive: prevents model eviction from VRAM between interactions.
# Without this, a 30-min gap causes a GPU→CPU fallback on the next call.
AURA_KEEP_ALIVE    = os.getenv("AURA_KEEP_ALIVE", "2h")
AURA_NUM_GPU       = int(os.getenv("AURA_NUM_GPU", "99"))   # offload all layers to Jetson GPU
# Lightweight model for keyword extraction — runs in parallel with the embed call.
# Using the 1b intent model here (not 3b) means: (a) it finishes faster so the
# parallel phase takes ~600ms instead of ~2s, and (b) the 3b model's KV cache is
# NOT evicted between keyword extraction and answer generation, so Ollama can reuse
# any cached prefix KV from the previous query.
AURA_FAST_LLM      = os.getenv("AURA_INTENT_MODEL", "llama3.2:1b")

# Answer generation
# num_ctx=4096: with 4800-char chunks and MAX_CTX_CHARS=12000, actual prompts are
# ~3200 tokens (context + query + system). llama3.2:1b KV cache at q8_0 is ~200 MB
# at 4096 tokens — well within the Jetson's 8 GB alongside camera/YOLO.
AURA_NUM_PREDICT   = int(os.getenv("AURA_NUM_PREDICT", "512"))
AURA_NUM_CTX       = int(os.getenv("AURA_NUM_CTX", "4096"))
AURA_TEMPERATURE   = float(os.getenv("AURA_TEMPERATURE", "0.2"))
AURA_NUM_THREAD    = int(os.getenv("AURA_NUM_THREAD", "0"))  # 0 = auto

# Graph extraction (build-time only, not on the query path)
# Default OFF: each chunk requires one LLM call (~20s on Jetson), making a 30-chunk
# document take 10+ minutes. Enable with AURA_GRAPH_EXTRACT=true only when build
# time is not a concern and the richer entity graph is needed.
AURA_GRAPH_EXTRACT       = os.getenv("AURA_GRAPH_EXTRACT", "false").lower() == "true"
AURA_GRAPH_TIMEOUT_S     = float(os.getenv("AURA_GRAPH_TIMEOUT_S", "90"))
AURA_GRAPH_NUM_PREDICT   = int(os.getenv("AURA_GRAPH_NUM_PREDICT", "768"))
AURA_GRAPH_NUM_CTX       = int(os.getenv("AURA_GRAPH_NUM_CTX", "3072"))

# Retrieval
# MAX_CTX_CHARS=12000: ~3000 tokens; fits 2-3 full 4800-char chunks in context.
# With AURA_NUM_CTX=4096, total prompt (system+context+query+answer) stays within budget.
MAX_CTX_CHARS       = int(os.getenv("AURA_MAX_CTX_CHARS", "12000"))
DEFAULT_TOP_K       = int(os.getenv("AURA_TOP_K", "8"))
BM25_REBUILD_EVERY  = int(os.getenv("AURA_BM25_REBUILD_EVERY", "50"))
AURA_LOCAL_TOP_K    = int(os.getenv("AURA_LOCAL_TOP_K", "4"))   # entity matches
AURA_GLOBAL_TOP_K   = int(os.getenv("AURA_GLOBAL_TOP_K", "4"))  # entity FAISS hits

# Chunking — reference LightRAG uses chunk_token_size=1200 TOKENS.
# nomic-embed-text averages ~4 chars/token for English technical text, so
# 1200 tokens ≈ 4800 chars per chunk. The previous default of 1200 chars
# was only ~300 tokens — 4× too small, causing far too many tiny chunks and
# degrading retrieval quality. This was the primary cause of "1 chunk" results
# for multi-page academic PDFs.
AURA_INSERT_CHUNK_SIZE    = int(os.getenv("AURA_INSERT_CHUNK_SIZE", "4800"))
# Overlap: reference uses 100 tokens → ~400 chars
AURA_INSERT_CHUNK_OVERLAP = int(os.getenv("AURA_INSERT_CHUNK_OVERLAP", "400"))
# Minimum chunk length: filters stray punctuation / arXiv stamps from PDF
# extraction. Kept low so dense tables and short-paragraph PDFs aren't dropped.
AURA_MIN_CHUNK_CHARS      = int(os.getenv("AURA_MIN_CHUNK_CHARS", "30"))

# Batch embedding
AURA_EMBED_BATCH_SIZE = int(os.getenv("AURA_EMBED_BATCH_SIZE", "8"))

# Semantic query cache
# Tier-1 (exact hit): cosine sim ≥ EXACT_THRESH → return cached answer immediately.
# Tier-2 (context reuse): sim ≥ CTX_THRESH → skip FAISS/BM25, reuse cached passages.
#   Because the same passages produce the same prompt prefix, Ollama's KV cache
#   fires automatically and prefill drops from ~3000 tokens to ~30 tokens.
AURA_CACHE_EXACT_THRESH = float(os.getenv("AURA_CACHE_EXACT_THRESH", "0.93"))
AURA_CACHE_CTX_THRESH   = float(os.getenv("AURA_CACHE_CTX_THRESH",   "0.82"))
AURA_CACHE_MAX_ENTRIES  = int(os.getenv("AURA_CACHE_MAX_ENTRIES",    "64"))
AURA_CACHE_TTL_S        = float(os.getenv("AURA_CACHE_TTL_S",        "3600"))  # 1 h


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class QueryParam:
    """Controls retrieval mode for aquery()."""
    mode: str = "hybrid"   # "local" | "global" | "hybrid"
    top_k: int = DEFAULT_TOP_K


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj):
    _safe_mkdir(os.path.dirname(path))
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) + 1e-12
    return (v / norm).astype(np.float32)


# ─── Semantic query cache ─────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    q_norm:     np.ndarray          # unit-normalized query embedding
    chunk_hits: List[Dict[str, Any]]
    sources:    List[str]
    answer:     str
    created_at: float               # time.time()


class _QueryCache:
    """
    Two-tier LRU semantic cache keyed by cosine similarity on query embeddings.

    Tier 1 (exact, sim ≥ EXACT_THRESH): return cached answer immediately.
    Tier 2 (context, sim ≥ CTX_THRESH): return cached chunk_hits/sources so
        the caller can skip retrieval. The same passages → same prompt prefix →
        Ollama KV cache fires → prefill ~30 tokens instead of ~3000.
    """

    def __init__(self):
        self._entries: List[_CacheEntry] = []

    def _evict(self):
        now = time.time()
        # Remove expired entries
        self._entries = [e for e in self._entries if now - e.created_at < AURA_CACHE_TTL_S]
        # LRU eviction when still over capacity
        if len(self._entries) > AURA_CACHE_MAX_ENTRIES:
            self._entries = self._entries[-AURA_CACHE_MAX_ENTRIES:]

    def lookup(self, q_norm: np.ndarray) -> Tuple[str, Optional[_CacheEntry]]:
        """
        Returns ("exact", entry) | ("context", entry) | ("miss", None).
        """
        self._evict()
        if not self._entries:
            return "miss", None
        # Stack all entry vectors and do a single matrix dot product
        mat = np.stack([e.q_norm for e in self._entries])   # (N, D)
        sims = mat @ q_norm                                   # (N,)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        entry = self._entries[best_idx]
        if best_sim >= AURA_CACHE_EXACT_THRESH:
            # Promote to most-recent (LRU touch)
            self._entries.pop(best_idx)
            self._entries.append(entry)
            return "exact", entry
        if best_sim >= AURA_CACHE_CTX_THRESH:
            return "context", entry
        return "miss", None

    def store(self, q_norm: np.ndarray, chunk_hits, sources, answer: str):
        self._entries.append(_CacheEntry(
            q_norm=q_norm,
            chunk_hits=chunk_hits,
            sources=sources,
            answer=answer,
            created_at=time.time(),
        ))
        self._evict()

    def clear(self):
        self._entries.clear()


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    out: List[str] = []
    word: List[str] = []
    for ch in s:
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                out.append("".join(word))
                word = []
    if word:
        out.append("".join(word))
    return out


def _chunk_text_chars(text: str, max_chars: int, overlap: int) -> List[str]:
    """Pure character-based sliding-window split (used for oversized paragraphs)."""
    chunks: List[str] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        step = max_chars - overlap
        i += max(1, step)  # guard against zero/negative step
    return chunks


def _chunk_text(
    text: str,
    max_chars: int = AURA_INSERT_CHUNK_SIZE,
    overlap: int = AURA_INSERT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Paragraph-aware chunking (follows the reference LightRAG approach).

    Groups consecutive paragraphs into chunks up to max_chars, keeping
    paragraph boundaries intact wherever possible. A single paragraph that
    exceeds max_chars alone is split with the character-based fallback.
    Overlap is achieved by carrying the last paragraph of the previous
    chunk into the start of the next one.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Split into paragraphs (double newline boundary)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if para_len > max_chars:
            # Flush accumulated paragraphs first
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            # Split the oversized paragraph with character overlap
            for sub in _chunk_text_chars(para, max_chars, overlap):
                chunks.append(sub)
            continue

        sep = 2 if current else 0  # "\n\n" separator cost
        if current_len + sep + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            # Carry the last paragraph as overlap into the next chunk
            carry = current[-1]
            current = [carry]
            current_len = len(carry)

        current.append(para)
        current_len += (2 if len(current) > 1 else 0) + para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _parse_json_from_llm(text: str) -> dict:
    """
    Robustly extract the outermost JSON object from LLM output.
    Handles markdown code fences, surrounding prose, and minor syntax issues.
    """
    text = (text or "").strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Walk chars to find outermost balanced { }
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1  # reset, try next occurrence

    return {}


# ─── Prompt templates ─────────────────────────────────────────────────────────
# Using string concatenation to avoid .format() conflicts with JSON { } chars.

_GRAPH_EXTRACT_SYSTEM = (
    "You are a knowledge graph extractor. "
    "Output only valid JSON. No explanations, no markdown fences."
)

_GRAPH_EXTRACT_PREFIX = """\
Extract entities and relationships from the text below.
Return ONLY a JSON object with this exact structure (no extra text):

{
  "entities": [
    {"name": "LightRAG", "type": "concept", "description": "a graph-enhanced RAG system combining vector and graph retrieval"}
  ],
  "relations": [
    {"src": "LightRAG", "tgt": "FAISS", "description": "uses FAISS for vector similarity search", "keywords": ["retrieval", "index"], "strength": 8}
  ],
  "high_level_keywords": ["information retrieval", "knowledge graphs"]
}

entity types: concept, person, organization, equipment, process, location
strength: 1-10 integer (importance of the relationship)
If nothing meaningful is found, return: {"entities": [], "relations": [], "high_level_keywords": []}

Text:
"""

_KEYWORD_EXTRACT_SYSTEM = "You are a keyword extractor. Output only valid JSON."

_KEYWORD_EXTRACT_PREFIX = """\
Given the user query, extract two types of keywords.
Output JSON:
{"high_level_keywords": ["overarching theme", "..."], "low_level_keywords": ["specific entity", "..."]}

Query: """


# ─── Ollama client ────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, llm_model: str):
        self.base_url = (base_url or AURA_OLLAMA_URL).rstrip("/")
        self.embed_model = embed_model
        self.llm_model = llm_model

    def _post_json(
        self, path: str, payload: Dict[str, Any], timeout_s: float
    ) -> Dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def embed_batch(self, texts: List[str], timeout_s: float = 120.0) -> List[np.ndarray]:
        """
        Embed a batch of texts.  Tries the Ollama 0.1.26+ /api/embed endpoint
        (which accepts a list and returns all embeddings in one round-trip), then
        falls back to individual /api/embeddings calls.
        """
        if not texts:
            return []

        # Try new batch endpoint
        try:
            payload = {
                "model": self.embed_model,
                "input": texts,
                "options": {"num_gpu": AURA_NUM_GPU},
            }
            out = await asyncio.to_thread(
                self._post_json, "/api/embed", payload, timeout_s
            )
            embeddings = out.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(texts):
                return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception:
            pass

        # Fallback: one call per text
        results: List[np.ndarray] = []
        per_call = max(10.0, timeout_s / len(texts))
        for text in texts:
            try:
                out = await asyncio.to_thread(
                    self._post_json,
                    "/api/embeddings",
                    {
                        "model": self.embed_model,
                        "prompt": text,
                        "options": {"num_gpu": AURA_NUM_GPU},
                    },
                    per_call,
                )
                emb = out.get("embedding")
                if isinstance(emb, list) and emb:
                    results.append(np.array(emb, dtype=np.float32))
                    continue
            except Exception as e:
                raise RuntimeError(
                    f"Ollama embed failed at {self.base_url}. "
                    f"Is Ollama running? ({e})"
                )
            raise RuntimeError(f"Ollama embed returned empty for: {text[:60]!r}")
        return results

    async def embed(self, text: str, timeout_s: float = 60.0) -> np.ndarray:
        results = await self.embed_batch([text], timeout_s=timeout_s)
        return results[0]

    def _make_options(
        self,
        num_predict: int,
        num_ctx: int,
        temperature: float,
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "num_gpu": AURA_NUM_GPU,  # offload all layers to Jetson GPU
        }
        if AURA_NUM_THREAD > 0:
            opts["num_thread"] = AURA_NUM_THREAD
        # mirostat=0 disables adaptive perplexity-targeting sampling.  At
        # temperature=0.2 the distribution is already sharp enough that mirostat
        # adds overhead (~5-10% per token) without measurable quality gain.
        opts["mirostat"] = 0
        return opts

    async def generate(
        self,
        prompt: str,
        system: str = "",
        timeout_s: float = AURA_OLLAMA_TIMEOUT_S,
        num_predict: int = AURA_NUM_PREDICT,
        num_ctx: int = AURA_NUM_CTX,
        temperature: float = AURA_TEMPERATURE,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """
        Generate a response.  When on_token is provided, streams the response
        token-by-token and calls on_token(token) as each arrives — allows the
        caller to pipe tokens to TTS sentence-by-sentence instead of waiting
        for the full answer.  Returns the complete answer string regardless.
        """
        streaming = on_token is not None
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "prompt": prompt,
            "system": system,
            "stream": streaming,
            "keep_alive": AURA_KEEP_ALIVE,
            "options": self._make_options(num_predict, num_ctx, temperature),
        }

        if not streaming:
            try:
                out = await asyncio.to_thread(
                    self._post_json, "/api/generate", payload, timeout_s
                )
            except Exception as e:
                raise RuntimeError(
                    f"Ollama generate failed at {self.base_url}. "
                    f"Is Ollama running? ({e})"
                )
            txt = out.get("response")
            if not isinstance(txt, str):
                raise RuntimeError("Ollama generate returned no response text.")
            return txt.strip()

        # Streaming path: run the blocking HTTP stream in a thread, feed tokens
        # back to the event loop via a queue so on_token() can be awaited safely.
        loop = asyncio.get_running_loop()
        token_q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=1024)

        def _stream_worker() -> None:
            try:
                with requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=timeout_s,
                ) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except Exception:
                            continue
                        tok = obj.get("response", "")
                        if tok:
                            loop.call_soon_threadsafe(token_q.put_nowait, tok)
                        if obj.get("done", False):
                            break
            except Exception:
                pass
            finally:
                loop.call_soon_threadsafe(token_q.put_nowait, None)  # sentinel

        worker = loop.run_in_executor(None, _stream_worker)
        tokens: List[str] = []
        while True:
            tok = await token_q.get()
            if tok is None:
                break
            tokens.append(tok)
            await on_token(tok)
        await worker
        return "".join(tokens).strip()


# ─── LightRAG ────────────────────────────────────────────────────────────────

class LightRAG:
    """
    Graph-enhanced RAG store.  Dual-level retrieval following the LightRAG paper:

      Local  (low-level)  — entity name matching → entity descriptions + 1-hop relations
      Global (high-level) — vector similarity on entity description embeddings
      Hybrid              — both + BM25 on raw chunks

    File layout in working_dir
    ─────────────────────────────────────────────────────
    meta.json            chunk metadata list
    db.json              cloud-sync summary
    entities.json        entity export for cloud sync
    graph.json           knowledge graph: entities + relations   [graph feature]
    embeddings.npy       chunk embedding matrix
    faiss.index          chunk FAISS index
    entity_list.json     entity rows parallel to entity FAISS   [graph feature]
    entity_emb.npy       entity description embedding matrix    [graph feature]
    entity_faiss.index   entity FAISS index                     [graph feature]
    """

    def __init__(
        self,
        working_dir: str,
        llm_model_name: str,
        embed_model_name: str,
        ollama_base_url: str = "",
    ):
        self.working_dir = os.path.abspath(working_dir)
        _safe_mkdir(self.working_dir)

        # ── chunk store ──────────────────────────────────────────────────
        self.meta_path       = os.path.join(self.working_dir, "meta.json")
        self.db_json_path    = os.path.join(self.working_dir, "db.json")
        self.emb_path        = os.path.join(self.working_dir, "embeddings.npy")
        self.index_path      = os.path.join(self.working_dir, "faiss.index")

        # ── graph store ──────────────────────────────────────────────────
        self.graph_path         = os.path.join(self.working_dir, "graph.json")
        self.entity_list_path   = os.path.join(self.working_dir, "entity_list.json")
        self.entity_emb_path    = os.path.join(self.working_dir, "entity_emb.npy")
        self.entity_index_path  = os.path.join(self.working_dir, "entity_faiss.index")

        # ── cloud sync export ────────────────────────────────────────────
        self.entities_path = os.path.join(self.working_dir, "entities.json")

        self.client = OllamaClient(
            base_url=ollama_base_url or AURA_OLLAMA_URL,
            embed_model=embed_model_name,
            llm_model=llm_model_name,
        )
        # Separate lightweight client for keyword extraction — uses 1b model so
        # the 3b answer model's KV cache is never touched during retrieval prep.
        self._fast_client = OllamaClient(
            base_url=ollama_base_url or AURA_OLLAMA_URL,
            embed_model=embed_model_name,
            llm_model=AURA_FAST_LLM,
        )

        self._cache = _QueryCache()

        # chunk runtime state
        self._rows: List[Dict[str, Any]] = _load_json(self.meta_path, default=[])
        self._emb: Optional[np.ndarray] = None
        self._index: Optional[faiss.Index] = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []
        self._inserts_since_bm25 = 0

        # graph runtime state
        # schema: {
        #   "entities": { "Name": {"type": str, "description": str, "sources": [...]} },
        #   "relations": [{"src", "tgt", "description", "keywords", "strength", "source"}],
        #   "chunk_keywords": [["kw", ...], ...]   # parallel to self._rows
        # }
        self._graph: Dict[str, Any] = {
            "entities": {}, "relations": [], "chunk_keywords": []
        }

        # entity FAISS runtime state
        self._entity_rows: List[Dict[str, Any]] = []
        self._entity_emb: Optional[np.ndarray] = None
        self._entity_index: Optional[faiss.Index] = None
        self._entity_index_dirty = False

        self._load_store()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_store(self):
        # Chunk embeddings + FAISS
        if os.path.exists(self.emb_path):
            try:
                self._emb = np.load(self.emb_path)
            except Exception:
                self._emb = None

        if (
            self._emb is not None
            and self._emb.ndim == 2
            and self._emb.shape[0] == len(self._rows)
        ):
            dim = int(self._emb.shape[1])
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._emb.astype(np.float32))
        else:
            self._emb = None
            self._index = None

        self._bm25_tokens = [_tokenize(r.get("text", "")) for r in self._rows]
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None
        self._inserts_since_bm25 = 0

        # Graph
        raw_graph = _load_json(
            self.graph_path,
            default={"entities": {}, "relations": [], "chunk_keywords": []},
        )
        self._graph = {
            "entities":       raw_graph.get("entities", {}) if isinstance(raw_graph.get("entities"), dict) else {},
            "relations":      raw_graph.get("relations", []) if isinstance(raw_graph.get("relations"), list) else [],
            "chunk_keywords": raw_graph.get("chunk_keywords", []) if isinstance(raw_graph.get("chunk_keywords"), list) else [],
        }

        # Entity FAISS
        self._entity_rows = _load_json(self.entity_list_path, default=[])
        if os.path.exists(self.entity_emb_path):
            try:
                self._entity_emb = np.load(self.entity_emb_path)
            except Exception:
                self._entity_emb = None

        if (
            self._entity_emb is not None
            and self._entity_emb.ndim == 2
            and self._entity_emb.shape[0] == len(self._entity_rows)
            and self._entity_rows
        ):
            dim = int(self._entity_emb.shape[1])
            self._entity_index = faiss.IndexFlatIP(dim)
            self._entity_index.add(self._entity_emb.astype(np.float32))
        else:
            self._entity_emb = None
            self._entity_index = None

    def flush(self):
        if self._bm25_tokens:
            self._bm25 = BM25Okapi(self._bm25_tokens)

        _save_json(self.meta_path, self._rows)
        _save_json(
            self.db_json_path,
            {
                "chunk_count":    len(self._rows),
                "entity_count":   len(self._graph.get("entities", {})),
                "relation_count": len(self._graph.get("relations", [])),
                "updated_at_epoch":    time.time(),
                "updated_at_readable": time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": self._rows,
            },
        )

        # Export entities for cloud sync
        ents = self._graph.get("entities", {})
        _save_json(
            self.entities_path,
            {
                "count": len(ents),
                "entities": [{"name": n, **info} for n, info in ents.items()],
            },
        )

        _save_json(self.graph_path, self._graph)

        if self._emb is not None:
            np.save(self.emb_path, self._emb.astype(np.float32))
        if self._index is not None:
            faiss.write_index(self._index, self.index_path)

        _save_json(self.entity_list_path, self._entity_rows)
        if self._entity_emb is not None:
            np.save(self.entity_emb_path, self._entity_emb.astype(np.float32))
        if self._entity_index is not None:
            faiss.write_index(self._entity_index, self.entity_index_path)

    def reset(self):
        self._rows = []
        self._emb = None
        self._index = None
        self._bm25 = None
        self._bm25_tokens = []
        self._inserts_since_bm25 = 0
        self._graph = {"entities": {}, "relations": [], "chunk_keywords": []}
        self._entity_rows = []
        self._entity_emb = None
        self._entity_index = None
        self._entity_index_dirty = False

        for p in [
            self.meta_path, self.db_json_path, self.entities_path,
            self.emb_path, self.index_path, self.graph_path,
            self.entity_list_path, self.entity_emb_path, self.entity_index_path,
        ]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        return {
            "chunk_count":    len(self._rows),
            "entity_count":   len(self._graph.get("entities", {})),
            "relation_count": len(self._graph.get("relations", [])),
            "vdb_path":       self.working_dir,
            "graph_enabled":  AURA_GRAPH_EXTRACT,
        }

    # ── graph mutation helpers ────────────────────────────────────────────────

    def _merge_entity(self, name: str, etype: str, description: str, source: str):
        """Insert or merge an entity into the graph."""
        ents = self._graph["entities"]
        if name not in ents:
            ents[name] = {"type": etype, "description": description, "sources": [source]}
        else:
            existing = ents[name]
            if description and description not in existing.get("description", ""):
                existing["description"] = existing.get("description", "") + "; " + description
            if source not in existing.get("sources", []):
                existing.setdefault("sources", []).append(source)

    def _add_relation(
        self,
        src: str,
        tgt: str,
        description: str,
        keywords: List[str],
        strength: int,
        source: str,
    ):
        """Add a relation, merging if src+tgt already exists."""
        rels = self._graph["relations"]
        for rel in rels:
            if rel.get("src") == src and rel.get("tgt") == tgt:
                rel["strength"] = max(rel.get("strength", 1), strength)
                if description and description not in rel.get("description", ""):
                    rel["description"] = rel.get("description", "") + "; " + description
                for kw in keywords:
                    if kw not in rel.get("keywords", []):
                        rel.setdefault("keywords", []).append(kw)
                return
        rels.append({
            "src": src, "tgt": tgt,
            "description": description,
            "keywords": keywords,
            "strength": max(1, min(10, int(strength))),
            "source": source,
        })

    # ── graph extraction (build-time) ─────────────────────────────────────────

    async def _extract_graph(self, text: str, source: str = "") -> Dict[str, Any]:
        """
        Call LLM to extract entities, relations, and high-level keywords.
        Errors are non-fatal: returns {} so insert continues.
        """
        prompt = _GRAPH_EXTRACT_PREFIX + text[:2000]
        try:
            raw = await self.client.generate(
                prompt=prompt,
                system=_GRAPH_EXTRACT_SYSTEM,
                timeout_s=AURA_GRAPH_TIMEOUT_S,
                num_predict=AURA_GRAPH_NUM_PREDICT,
                num_ctx=AURA_GRAPH_NUM_CTX,
                temperature=0.0,   # deterministic extraction

            )
            result = _parse_json_from_llm(raw)
            if not result:
                print(f"[LightRAG]  graph extraction returned empty JSON (non-fatal)")
            return result
        except Exception as e:
            print(f"[LightRAG]  graph extraction skipped (non-fatal): {e}")
            return {}

    async def _rebuild_entity_index(self):
        """Embed all entity descriptions and rebuild the entity FAISS index."""
        entities = self._graph.get("entities", {})
        if not entities:
            return

        names = list(entities.keys())
        # Paper uses "<name>: <description>" as the embedding text for each entity
        texts = [
            f"{name}: {info.get('description', '')}"
            for name, info in entities.items()
        ]

        n = len(texts)
        print(f"[LightRAG]  rebuilding entity index: {n} entities (batch size {AURA_EMBED_BATCH_SIZE})...")
        all_embs: List[np.ndarray] = []
        for i in range(0, n, AURA_EMBED_BATCH_SIZE):
            batch = texts[i : i + AURA_EMBED_BATCH_SIZE]
            embs = await self.client.embed_batch(batch)
            all_embs.extend(embs)
            if n > AURA_EMBED_BATCH_SIZE:
                print(f"[LightRAG]  entity embed {min(i + len(batch), n)}/{n}")

        emb_matrix = np.vstack([_normalize(e).reshape(1, -1) for e in all_embs]).astype(np.float32)

        self._entity_rows = [
            {"name": name, "text": text, "info": entities[name]}
            for name, text in zip(names, texts)
        ]
        self._entity_emb = emb_matrix

        dim = int(emb_matrix.shape[1])
        self._entity_index = faiss.IndexFlatIP(dim)
        self._entity_index.add(emb_matrix)
        self._entity_index_dirty = False

        print(f"[LightRAG]  entity index ready — {n} entities")

    # ── insert ────────────────────────────────────────────────────────────────

    async def _insert_one_chunk(self, text: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        source = meta.get("source", f"chunk_{_now_ms()}")

        # Embed and store chunk
        emb = _normalize(await self.client.embed(text))
        _id = f"chunk_{_now_ms()}_{len(self._rows)}"
        self._rows.append({"id": _id, "text": text, "meta": meta})

        if self._emb is None:
            self._emb = emb.reshape(1, -1)
            self._index = faiss.IndexFlatIP(int(self._emb.shape[1]))
            self._index.add(self._emb)
        else:
            self._emb = np.vstack([self._emb, emb.reshape(1, -1)])
            self._index.add(emb.reshape(1, -1))   # type: ignore[union-attr]

        self._bm25_tokens.append(_tokenize(text))
        self._inserts_since_bm25 += 1
        if self._inserts_since_bm25 >= BM25_REBUILD_EVERY:
            self._bm25 = BM25Okapi(self._bm25_tokens)
            self._inserts_since_bm25 = 0

        # Graph extraction
        if AURA_GRAPH_EXTRACT:
            extracted = await self._extract_graph(text, source=source)

            for ent in extracted.get("entities", []):
                name = str(ent.get("name", "")).strip()
                if name:
                    self._merge_entity(
                        name=name,
                        etype=str(ent.get("type", "concept")),
                        description=str(ent.get("description", "")),
                        source=source,
                    )

            for rel in extracted.get("relations", []):
                src = str(rel.get("src", "")).strip()
                tgt = str(rel.get("tgt", "")).strip()
                if src and tgt:
                    self._add_relation(
                        src=src, tgt=tgt,
                        description=str(rel.get("description", "")),
                        keywords=[str(k) for k in rel.get("keywords", [])],
                        strength=int(rel.get("strength", 5)),
                        source=source,
                    )

            # Align chunk_keywords list length
            kw_list = self._graph["chunk_keywords"]
            while len(kw_list) < len(self._rows):
                kw_list.append([])
            kw_list[len(self._rows) - 1] = [
                str(k) for k in extracted.get("high_level_keywords", [])
            ]

            self._entity_index_dirty = True

    async def ainsert(self, text: str, meta: Optional[Dict[str, Any]] = None):
        raw_chunks = _chunk_text(text)
        # Filter sub-paragraph artifacts: page headers, arXiv stamps, lone captions,
        # etc. These short fragments embed close to unrelated queries and pollute
        # vector search results without contributing useful content.
        chunks = [c for c in raw_chunks if len(c) >= AURA_MIN_CHUNK_CHARS]
        source = (meta or {}).get("source", f"doc_{_now_ms()}")
        if not chunks:
            print(
                f"[LightRAG] insert: 0 usable chunks from '{source}' "
                f"(raw={len(raw_chunks)}, all filtered by AURA_MIN_CHUNK_CHARS={AURA_MIN_CHUNK_CHARS}). "
                f"Longest raw chunk: {max((len(c) for c in raw_chunks), default=0)} chars."
            )
            return

        dropped = len(raw_chunks) - len(chunks)
        total = len(chunks)
        if dropped:
            print(f"[LightRAG] insert: {total} chunk(s) from '{source}' ({dropped} short chunk(s) filtered, min={AURA_MIN_CHUNK_CHARS})")
        else:
            print(f"[LightRAG] insert: {total} chunk(s) from '{source}'")

        # ── 1. Batch-embed ALL chunks in one round-trip ───────────────────────
        # This is the main speed win vs one embed call per chunk.
        print(f"[LightRAG]  batch-embedding {total} chunk(s)...")
        all_embs: List[np.ndarray] = []
        for b_start in range(0, total, AURA_EMBED_BATCH_SIZE):
            batch = chunks[b_start : b_start + AURA_EMBED_BATCH_SIZE]
            batch_embs = await self.client.embed_batch(batch)
            all_embs.extend(batch_embs)
            end = min(b_start + len(batch), total)
            if total > AURA_EMBED_BATCH_SIZE:
                print(f"[LightRAG]  embedded chunks {b_start + 1}–{end}/{total}")
        print(f"[LightRAG]  all {total} chunk embeddings ready")

        # ── 2. Store chunks + embeddings ──────────────────────────────────────
        chunk_start_idx = len(self._rows)
        for idx, (chunk, emb) in enumerate(zip(chunks, all_embs)):
            emb_norm = _normalize(emb)
            _id = f"chunk_{_now_ms()}_{len(self._rows)}"
            self._rows.append({
                "id": _id,
                "text": chunk,
                "meta": {**(meta or {}), "chunk_index": idx, "chunk_count": total},
            })
            if self._emb is None:
                self._emb = emb_norm.reshape(1, -1)
                self._index = faiss.IndexFlatIP(int(self._emb.shape[1]))
                self._index.add(self._emb)
            else:
                self._emb = np.vstack([self._emb, emb_norm.reshape(1, -1)])
                self._index.add(emb_norm.reshape(1, -1))  # type: ignore[union-attr]
            self._bm25_tokens.append(_tokenize(chunk))
            self._inserts_since_bm25 += 1

        if self._inserts_since_bm25 >= BM25_REBUILD_EVERY:
            self._bm25 = BM25Okapi(self._bm25_tokens)
            self._inserts_since_bm25 = 0

        # ── 3. Graph extraction (sequential — single GPU can't parallelize) ──
        if AURA_GRAPH_EXTRACT:
            kw_list = self._graph["chunk_keywords"]
            # Pre-fill keyword list so indices are aligned with self._rows
            while len(kw_list) < len(self._rows):
                kw_list.append([])

            for idx, chunk in enumerate(chunks):
                row_idx = chunk_start_idx + idx
                print(f"[LightRAG]  graph extract chunk {idx + 1}/{total}...")
                extracted = await self._extract_graph(chunk, source=source)

                n_ent = len(extracted.get("entities", []))
                n_rel = len(extracted.get("relations", []))
                print(f"[LightRAG]  chunk {idx + 1}/{total}: {n_ent} entities, {n_rel} relations")

                for ent in extracted.get("entities", []):
                    name = str(ent.get("name", "")).strip()
                    if name:
                        self._merge_entity(
                            name=name,
                            etype=str(ent.get("type", "concept")),
                            description=str(ent.get("description", "")),
                            source=source,
                        )

                for rel in extracted.get("relations", []):
                    src = str(rel.get("src", "")).strip()
                    tgt = str(rel.get("tgt", "")).strip()
                    if src and tgt:
                        self._add_relation(
                            src=src, tgt=tgt,
                            description=str(rel.get("description", "")),
                            keywords=[str(k) for k in rel.get("keywords", [])],
                            strength=int(rel.get("strength", 5)),
                            source=source,
                        )

                kw_list[row_idx] = [
                    str(k) for k in extracted.get("high_level_keywords", [])
                ]

            total_ent = len(self._graph["entities"])
            total_rel = len(self._graph["relations"])
            print(f"[LightRAG]  graph totals: {total_ent} entities, {total_rel} relations")

            self._entity_index_dirty = True
            await self._rebuild_entity_index()

        print(f"[LightRAG] insert done — DB now has {len(self._rows)} chunk(s)")

    # ── search helpers ────────────────────────────────────────────────────────

    def _search_vector(self, q_emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self._index is None or not self._rows:
            return []
        q = _normalize(q_emb).reshape(1, -1)
        scores, idxs = self._index.search(q, max(1, int(top_k)))
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self._rows:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None
        if self._bm25 is None:
            return []
        toks = _tokenize(query)
        scores = self._bm25.get_scores(toks)
        idxs = np.argsort(-scores)[: max(1, int(top_k))]
        return [(int(i), float(scores[i])) for i in idxs]

    def _search_entity_local(self, low_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Low-level retrieval (paper §3.2): match entity names against low-level
        query keywords; return entity descriptions + 1-hop relations.
        """
        if not low_keywords or not self._graph.get("entities"):
            return []

        entities  = self._graph["entities"]
        relations = self._graph.get("relations", [])
        kw_lower  = {kw.lower() for kw in low_keywords if kw}

        matched: Set[str] = set()

        # Exact substring matching
        for kw in kw_lower:
            for name in entities:
                if kw in name.lower() or name.lower() in kw:
                    matched.add(name)

        # Token overlap fallback
        if not matched:
            for kw in kw_lower:
                kw_toks = set(_tokenize(kw))
                if not kw_toks:
                    continue
                for name in entities:
                    if kw_toks & set(_tokenize(name)):
                        matched.add(name)

        results: List[Dict[str, Any]] = []
        for name in list(matched)[: AURA_LOCAL_TOP_K]:
            info = entities.get(name, {})
            ent_rels = [
                r for r in relations
                if r.get("src") == name or r.get("tgt") == name
            ][:5]
            results.append({
                "name": name,
                "type": info.get("type", ""),
                "description": info.get("description", ""),
                "relations": ent_rels,
            })
        return results

    def _search_entity_vector(
        self, q_emb: np.ndarray, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Global retrieval (paper §3.2): vector similarity on entity descriptions.
        """
        if self._entity_index is None or not self._entity_rows:
            return []
        q = _normalize(q_emb).reshape(1, -1)
        scores, idxs = self._entity_index.search(q, max(1, int(top_k)))
        results: List[Dict[str, Any]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            row = self._entity_rows[int(i)]
            results.append({
                "name":        row.get("name", ""),
                "description": row.get("text", ""),
                "score":       float(s),
            })
        return results

    async def _extract_query_keywords(self, query: str) -> Dict[str, List[str]]:
        """
        Use LLM to extract high-level (themes) and low-level (specific entities)
        keywords from the query.  Fast: num_predict=80, num_ctx=512.
        Falls back to simple tokenization on any error.
        """
        prompt = _KEYWORD_EXTRACT_PREFIX + query
        try:
            raw = await self._fast_client.generate(
                prompt=prompt,
                system=_KEYWORD_EXTRACT_SYSTEM,
                timeout_s=15.0,
                num_predict=60,
                num_ctx=256,
                temperature=0.0,
            )
            parsed = _parse_json_from_llm(raw)
            return {
                "high": [str(k) for k in parsed.get("high_level_keywords", [])],
                "low":  [str(k) for k in parsed.get("low_level_keywords", [])],
            }
        except Exception:
            return {"high": [], "low": _tokenize(query)[:8]}

    # ── context assembly ──────────────────────────────────────────────────────

    def _build_context(
        self,
        chunk_hits:    List[Dict[str, Any]],
        entity_hits:   List[Dict[str, Any]],
        relation_hits: List[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []

        if entity_hits or relation_hits:
            graph_lines: List[str] = []
            if entity_hits:
                graph_lines.append("=== Entities ===")
                for e in entity_hits:
                    graph_lines.append(
                        f"• {e['name']} ({e.get('type', '')}): {e.get('description', '')}"
                    )
            if relation_hits:
                graph_lines.append("=== Relationships ===")
                for r in relation_hits:
                    graph_lines.append(
                        f"• {r.get('src', '')} → {r.get('tgt', '')}: "
                        f"{r.get('description', '')}"
                    )
            parts.append("\n".join(graph_lines))

        if chunk_hits:
            parts.append("=== Source Passages ===")
            # Track chars used so far (joining with \n\n between each part).
            # Stop adding passages once the next one would push past MAX_CTX_CHARS
            # so the LLM always sees complete passages — a mid-cut paragraph is
            # worse than one fewer complete paragraph.
            used = len("\n\n".join(parts))
            for i, hit in enumerate(chunk_hits, 1):
                src = (hit.get("meta") or {}).get("source", "")
                label = f"--- Passage {i} (from: {src}) ---" if src else f"--- Passage {i} ---"
                passage = f"{label}\n{hit.get('text', '')}"
                needed = len(passage) + 2  # +2 for the \n\n joiner
                if used + needed > MAX_CTX_CHARS:
                    break
                parts.append(passage)
                used += needed

        return "\n\n".join(parts)

    # ── query ─────────────────────────────────────────────────────────────────

    async def aquery(
        self,
        query: str,
        param: Optional[QueryParam] = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        param = param or QueryParam()

        if not self._rows:
            return {
                "answer": "Database is empty. Build the database first.",
                "sources": [], "hits": [],
            }

        mode      = (param.mode or "hybrid").lower()
        top_k     = max(1, int(param.top_k))
        has_graph = bool(self._graph.get("entities"))

        # ── 1. Embed query + extract keywords concurrently ────────────────
        # Keywords run regardless of graph: they expand the BM25 query so
        # retrieval still works when the raw query term appears in every chunk
        # (near-zero IDF).  Graph-retrieval steps below still guard on has_graph.
        # Lowercase query before embedding: nomic-embed-text is case-sensitive and
        # produces different vectors for "LightRAG" vs "lightrag", causing different
        # chunks to rank first depending on casing — the same conceptual query then
        # hits different sections and gets different answers. Lowercase normalizes
        # this. BM25 already lowercases via _tokenize, so this makes FAISS consistent.
        q_emb, keywords = await asyncio.gather(
            self.client.embed(query.lower()),
            self._extract_query_keywords(query),
        )

        q_norm = _normalize(np.array(q_emb, dtype=np.float32))

        # ── Cache lookup ──────────────────────────────────────────────────
        cache_tier, cached = self._cache.lookup(q_norm)

        if cache_tier == "exact":
            # Return cached answer immediately; fire on_token so streaming TTS works
            if on_token is not None:
                for tok in cached.answer.split(" "):
                    await on_token(tok + " ")
            return {
                "answer":        cached.answer,
                "sources":       cached.sources,
                "hits":          cached.chunk_hits,
                "entities_used": 0,
                "relations_used": 0,
                "_cache":        "exact",
            }

        # For context-reuse hits, skip steps 2–4 (retrieval) entirely
        _reuse_context = cache_tier == "context"

        if _reuse_context:
            # Skip all retrieval — reuse passages from the semantically similar query.
            # Same passages → same prompt prefix → Ollama KV cache hits automatically.
            chunk_hits    = cached.chunk_hits
            sources       = cached.sources
            entity_hits   = []
            relation_hits = []
        else:
            # ── 2. Low-level (local) entity retrieval ─────────────────────────
            local_entity_hits:   List[Dict] = []
            local_relation_hits: List[Dict] = []
            if has_graph and mode in ("local", "hybrid") and keywords["low"]:
                for er in self._search_entity_local(keywords["low"]):
                    local_entity_hits.append(er)
                    local_relation_hits.extend(er.get("relations", []))

            # ── 3. Global (high-level) entity vector retrieval ────────────────
            global_entity_hits: List[Dict] = []
            if has_graph and mode in ("global", "hybrid"):
                global_entity_hits = self._search_entity_vector(
                    q_emb, top_k=AURA_GLOBAL_TOP_K
                )

            # Merge entity hits, deduplicate by name
            seen_entities: Set[str] = set()
            entity_hits: List[Dict] = []
            for e in local_entity_hits + global_entity_hits:
                name = e.get("name", "")
                if name and name not in seen_entities:
                    seen_entities.add(name)
                    entity_hits.append(e)

            # Relation context: from local matches, or top-strength relations
            relation_hits: List[Dict] = list(local_relation_hits)
            if not relation_hits and seen_entities:
                all_rels = self._graph.get("relations", [])
                for rel in sorted(all_rels, key=lambda r: -r.get("strength", 1)):
                    if rel.get("src") in seen_entities or rel.get("tgt") in seen_entities:
                        relation_hits.append(rel)
                    if len(relation_hits) >= 5:
                        break

            # ── 4. Chunk-level hybrid retrieval ──────────────────────────────
            candidates: Dict[int, Dict[str, float]] = {}

            for idx, score in self._search_vector(q_emb, top_k=top_k * 2):
                candidates.setdefault(idx, {})["vec"] = score

            if mode in ("bm25", "hybrid", "global"):
                # Expand BM25 query with extracted keywords: when the raw query term
                # appears in every chunk its IDF → 0, making BM25 useless. Injecting
                # semantically related keywords (from the keyword-extraction LLM call)
                # gives BM25 non-zero signal via those alternative terms.
                kw_terms = (keywords.get("low") or []) + (keywords.get("high") or [])
                bm25_q = query + (" " + " ".join(kw_terms) if kw_terms else "")
                for idx, score in self._search_bm25(bm25_q, top_k=top_k * 2):
                    candidates.setdefault(idx, {})["bm25"] = score

            scored: List[Tuple[float, int]] = []
            for idx, d in candidates.items():
                vec   = d.get("vec", 0.0)
                bm    = d.get("bm25", 0.0)
                # Normalization constant 4.0 (was 8.0): tighter constant spreads BM25
                # scores further apart, making keyword hits more discriminating.
                bm_n  = bm / (abs(bm) + 4.0)
                # 60/40 split (was 75/25): BM25 needs more weight when the query term
                # appears in nearly every chunk (common in domain-specific docs), so
                # term-frequency differences within each chunk can override vector rank.
                total = (0.60 * vec + 0.40 * bm_n) if mode == "hybrid" else (
                    vec if mode == "vector" else bm_n
                )
                scored.append((float(total), int(idx)))

            scored.sort(key=lambda x: x[0], reverse=True)

            chunk_hits: List[Dict[str, Any]] = []
            sources:    List[str] = []
            for score, idx in scored[:top_k]:
                r = self._rows[idx]
                chunk_hits.append({"score": score, "text": r.get("text", ""), "meta": r.get("meta", {})})
                src = (r.get("meta") or {}).get("source")
                if isinstance(src, str) and src and src not in sources:
                    sources.append(src)

        # ── 5. Build context + generate answer ───────────────────────────
        context = self._build_context(chunk_hits, entity_hits, relation_hits)
        system = (
            "You are AURA, a helpful lab assistant robot. "
            "Answer the question using the retrieved passages as your primary source. "
            "If the passages directly answer the question, answer from them. "
            "If the passages only partially answer or mention the topic indirectly, "
            "combine what the passages say with your general knowledge to give a complete answer — "
            "briefly note which parts come from the documents vs. your general knowledge. "
            "If the passages contain no relevant information at all, answer from your general knowledge "
            "and say 'The loaded documents do not cover this topic, but generally:' before your answer. "
            "Never invent specific measurements, values, formulas, or technical specifications "
            "not stated in the passages. "
            "Never expand an acronym unless its full form is explicitly written in the passages. "
            "Do not create numbered lists, bullet points, or headers unless the passages use that structure. "
            "Stop after fully answering the question."
        )
        prompt = (
            f"Retrieved passages from the knowledge base:\n\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        answer = await self.client.generate(
            prompt=prompt, system=system, timeout_s=AURA_OLLAMA_TIMEOUT_S,
            on_token=on_token,
        )

        self._cache.store(q_norm, chunk_hits, sources, answer)

        return {
            "answer":         answer,
            "sources":        sources,
            "hits":           chunk_hits,
            "entities_used":  len(entity_hits),
            "relations_used": len(relation_hits),
            "_cache":         "context" if _reuse_context else "miss",
        }
