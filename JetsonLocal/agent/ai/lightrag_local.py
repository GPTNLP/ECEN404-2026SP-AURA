# lightrag_local.py
#
# LightRAG-inspired hybrid retrieval system for the Jetson Orin Nano.
#
# Architecture (per the LightRAG paper):
#   1. Dual-level retrieval: vector (local/entity-level) + entity-graph (global-level)
#   2. Entity index: extracted via regex during ingestion, stored in entities.json
#   3. At query time: vector hits + graph-expanded hits are merged before LLM generation
#   4. BM25 keyword search provides a third retrieval signal (hybrid fusion)
#
# This avoids heavy graph-DB deps (networkx, neo4j, etc.) by using a plain
# adjacency list in JSON — appropriate for the edge compute constraints of the Nano.
from __future__ import annotations

import os
import re
import json
import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import urllib.request

import faiss
from rank_bm25 import BM25Okapi

# ---------------------------
# Tunables (env override)
# ---------------------------
AURA_OLLAMA_TIMEOUT_S = float(os.getenv("AURA_OLLAMA_TIMEOUT_S", "180"))
AURA_NUM_PREDICT     = int(os.getenv("AURA_NUM_PREDICT", "200"))
AURA_NUM_CTX         = int(os.getenv("AURA_NUM_CTX", "2048"))
AURA_TEMPERATURE     = float(os.getenv("AURA_TEMPERATURE", "0.2"))
AURA_NUM_THREAD      = int(os.getenv("AURA_NUM_THREAD", "0"))
AURA_KEEP_ALIVE      = os.getenv("AURA_KEEP_ALIVE", "10m")

MAX_CTX_CHARS        = int(os.getenv("AURA_MAX_CTX_CHARS", "6000"))
DEFAULT_TOP_K        = int(os.getenv("AURA_TOP_K", "4"))
BM25_REBUILD_EVERY   = int(os.getenv("AURA_BM25_REBUILD_EVERY", "50"))

# Graph retrieval: extra chunks pulled via entity overlap
GRAPH_EXPANSION_K    = int(os.getenv("AURA_GRAPH_EXPANSION_K", "3"))
# Minimum entity length and co-occurrence threshold
MIN_ENTITY_LEN       = int(os.getenv("AURA_MIN_ENTITY_LEN", "3"))


@dataclass
class QueryParam:
    mode: str = "hybrid"   # "vector" | "bm25" | "hybrid"
    top_k: int = DEFAULT_TOP_K


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
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
    os.replace(tmp, path)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v) + 1e-12
    return (v / norm).astype(np.float32)


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


# ---------------------------
# Entity extraction (regex-based, no LLM overhead during ingestion)
# ---------------------------
_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
_ACRONYM_RE     = re.compile(r"\b([A-Z]{2,6})\b")
_NUMBER_RE      = re.compile(r"\b\d{1,4}(?:\.\d+)?(?:\s*%|ms|kb|mb|gb|hz|mhz|ghz)?\b", re.IGNORECASE)

_STOP_WORDS: Set[str] = {
    "The", "This", "That", "These", "Those", "From", "With", "Into", "Over",
    "Also", "Then", "When", "Here", "There", "Some", "More", "Have", "Will",
    "Not", "Are", "Was", "Were", "Has", "Had", "Its", "For", "And", "But",
    "Or", "In", "On", "At", "By", "As", "An", "Be", "Is", "To", "Of",
    "Section", "Chapter", "Figure", "Table", "Page", "Note",
}


def extract_entities(text: str) -> List[str]:
    """
    Extract candidate entities from text using regex patterns.
    Returns deduplicated, filtered entity strings.
    """
    entities: List[str] = []

    for m in _CAPITALIZED_RE.finditer(text):
        e = m.group(1).strip()
        if e not in _STOP_WORDS and len(e) >= MIN_ENTITY_LEN:
            entities.append(e.lower())

    for m in _ACRONYM_RE.finditer(text):
        e = m.group(1).strip()
        if len(e) >= 2:
            entities.append(e.lower())

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


# ---------------------------
# Ollama client
# ---------------------------
class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, llm_model: str):
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.embed_model = embed_model
        self.llm_model = llm_model

    def _post_json(self, path: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw else {}

    async def embed(self, text: str, timeout_s: float = 30.0) -> np.ndarray:
        payload = {"model": self.embed_model, "prompt": text}
        try:
            out = await asyncio.to_thread(self._post_json, "/api/embeddings", payload, timeout_s)
        except Exception as e:
            raise RuntimeError(f"Ollama embed failed — is Ollama running at {self.base_url}? ({e})")

        emb = out.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError("Ollama returned no embedding vector.")
        return np.array(emb, dtype=np.float32)

    async def generate(self, prompt: str, system: str = "", timeout_s: float = 180.0) -> str:
        options: Dict[str, Any] = {
            "temperature": AURA_TEMPERATURE,
            "num_predict": AURA_NUM_PREDICT,
            "num_ctx": AURA_NUM_CTX,
        }
        if AURA_NUM_THREAD > 0:
            options["num_thread"] = AURA_NUM_THREAD

        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": AURA_KEEP_ALIVE,
            "options": options,
        }

        try:
            out = await asyncio.to_thread(self._post_json, "/api/generate", payload, timeout_s)
        except Exception as e:
            raise RuntimeError(f"Ollama generate failed — is Ollama running at {self.base_url}? ({e})")

        txt = out.get("response")
        if not isinstance(txt, str):
            raise RuntimeError("Ollama generate returned no response text.")
        return txt.strip()


# ---------------------------
# LightRAG
# ---------------------------
class LightRAG:
    """
    Persistent hybrid retrieval store implementing the LightRAG paper's
    dual-level (local entity + global graph) + BM25 retrieval strategy.

    Storage layout in working_dir:
      meta.json       — chunk records [{id, text, meta}]
      embeddings.npy  — float32 normalized vectors (N x D)
      faiss.index     — IndexFlatIP for cosine similarity
      entities.json   — entity graph: {entity: [chunk_id, ...]}
    """

    def __init__(
        self,
        working_dir: str,
        llm_model_name: str,
        embed_model_name: str,
        ollama_base_url: str = "http://127.0.0.1:11434",
    ):
        self.working_dir = os.path.abspath(working_dir)
        _safe_mkdir(self.working_dir)

        self.meta_path     = os.path.join(self.working_dir, "meta.json")
        self.emb_path      = os.path.join(self.working_dir, "embeddings.npy")
        self.index_path    = os.path.join(self.working_dir, "faiss.index")
        self.entity_path   = os.path.join(self.working_dir, "entities.json")

        self.client = OllamaClient(
            base_url=ollama_base_url,
            embed_model=embed_model_name,
            llm_model=llm_model_name,
        )

        # Chunk store
        self._rows: List[Dict[str, Any]] = _load_json(self.meta_path, default=[])
        self._emb: Optional[np.ndarray] = None
        self._index: Optional[faiss.Index] = None

        # BM25
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []
        self._inserts_since_bm25 = 0

        # Entity graph: entity_string → set of chunk IDs
        raw_graph = _load_json(self.entity_path, default={})
        self._entity_graph: Dict[str, List[str]] = {
            k: list(v) for k, v in raw_graph.items()
        }
        # Reverse map: chunk_id → row index (rebuilt in _load_store)
        self._id_to_idx: Dict[str, int] = {}

        self._load_store()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_store(self):
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

        self._id_to_idx = {r["id"]: i for i, r in enumerate(self._rows)}

    def flush(self):
        if self._bm25_tokens:
            self._bm25 = BM25Okapi(self._bm25_tokens)

        _save_json(self.meta_path, self._rows)

        if self._emb is not None:
            np.save(self.emb_path, self._emb.astype(np.float32))

        if self._index is not None:
            faiss.write_index(self._index, self.index_path)

        # Serialize entity graph (sets → lists)
        graph_serial = {k: list(v) for k, v in self._entity_graph.items()}
        _save_json(self.entity_path, graph_serial)

    def reset(self):
        self._rows = []
        self._emb = None
        self._index = None
        self._bm25 = None
        self._bm25_tokens = []
        self._inserts_since_bm25 = 0
        self._entity_graph = {}
        self._id_to_idx = {}

        for p in [self.meta_path, self.emb_path, self.index_path, self.entity_path]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def stats(self) -> Dict[str, Any]:
        return {
            "chunk_count": len(self._rows),
            "entity_count": len(self._entity_graph),
            "vdb_path": self.working_dir,
        }

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    async def ainsert(self, text: str, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        emb = await self.client.embed(text)
        emb = _normalize(emb)

        _id = f"chunk_{_now_ms()}_{len(self._rows)}"
        self._rows.append({"id": _id, "text": text, "meta": meta})
        self._id_to_idx[_id] = len(self._rows) - 1

        if self._emb is None:
            self._emb = emb.reshape(1, -1)
            dim = int(self._emb.shape[1])
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._emb)
        else:
            self._emb = np.vstack([self._emb, emb.reshape(1, -1)])
            self._index.add(emb.reshape(1, -1))

        # BM25
        self._bm25_tokens.append(_tokenize(text))
        self._inserts_since_bm25 += 1
        if self._inserts_since_bm25 >= BM25_REBUILD_EVERY:
            self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None
            self._inserts_since_bm25 = 0

        # --- Entity graph update (LightRAG local-level indexing) ---
        entities = extract_entities(text)
        for entity in entities:
            if entity not in self._entity_graph:
                self._entity_graph[entity] = []
            if _id not in self._entity_graph[entity]:
                self._entity_graph[entity].append(_id)

    # Sync wrapper kept for compatibility
    def insert(self, text: str, meta: Optional[Dict[str, Any]] = None):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Use ainsert() inside async context")
            loop.run_until_complete(self.ainsert(text, meta))
        except RuntimeError:
            asyncio.run(self.ainsert(text, meta))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _search_vector(self, q_emb: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self._index is None or self._emb is None or len(self._rows) == 0:
            return []
        q_emb = _normalize(q_emb).reshape(1, -1)
        scores, idxs = self._index.search(q_emb, max(1, int(top_k)))
        out: List[Tuple[int, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i >= 0:
                out.append((int(i), float(s)))
        return out

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

    def _graph_expand(self, query: str) -> List[int]:
        """
        LightRAG global-level retrieval:
        Extract entities from the query, find all chunks that share those
        entities, and return their row indices (de-duplicated).
        """
        if not self._entity_graph:
            return []

        query_entities = extract_entities(query)
        # Also match any entity whose text overlaps with query tokens
        query_tokens = set(_tokenize(query))

        related_ids: Set[str] = set()
        for entity, chunk_ids in self._entity_graph.items():
            entity_tokens = set(_tokenize(entity))
            if entity_tokens & query_tokens:  # token overlap
                related_ids.update(chunk_ids)
            # Also check query entities directly
            for qe in query_entities:
                if qe in entity or entity in qe:
                    related_ids.update(chunk_ids)
                    break

        # Convert chunk IDs to row indices
        idxs = []
        for cid in related_ids:
            idx = self._id_to_idx.get(cid)
            if idx is not None:
                idxs.append(idx)
        return idxs

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def aquery(self, query: str, param: Optional[QueryParam] = None) -> Dict[str, Any]:
        param = param or QueryParam()
        if not self._rows:
            return {
                "answer": "The database is empty. Build the database first.",
                "sources": [],
                "hits": [],
            }

        mode = (param.mode or "hybrid").lower()
        top_k = max(1, int(param.top_k))

        candidates: Dict[int, Dict[str, float]] = {}

        # --- Local (vector) retrieval ---
        if mode in ("vector", "hybrid"):
            q_emb = await self.client.embed(query)
            for idx, score in self._search_vector(q_emb, top_k=top_k * 2):
                candidates.setdefault(idx, {})
                candidates[idx]["vec"] = score

        # --- BM25 keyword retrieval ---
        if mode in ("bm25", "hybrid"):
            for idx, score in self._search_bm25(query, top_k=top_k * 2):
                candidates.setdefault(idx, {})
                candidates[idx]["bm25"] = score

        # --- Global (entity graph) retrieval — LightRAG paper §3.2 ---
        graph_idxs = self._graph_expand(query)
        for idx in graph_idxs[:GRAPH_EXPANSION_K]:
            candidates.setdefault(idx, {})
            # Give graph-expanded chunks a baseline score boost
            candidates[idx]["graph"] = candidates[idx].get("graph", 0.0) + 0.3

        # --- Fusion scoring ---
        scored: List[Tuple[float, int]] = []
        for idx, d in candidates.items():
            vec   = d.get("vec", 0.0)
            bm    = d.get("bm25", 0.0)
            graph = d.get("graph", 0.0)
            bm_norm = bm / (abs(bm) + 8.0)

            if mode == "hybrid":
                total = (0.60 * vec) + (0.20 * bm_norm) + (0.20 * graph)
            elif mode == "vector":
                total = vec + (0.15 * graph)
            else:  # bm25
                total = bm_norm + (0.15 * graph)

            scored.append((float(total), int(idx)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        hits = []
        sources = []
        ctx_parts = []

        for score, idx in top:
            r = self._rows[idx]
            hits.append({"score": score, "text": r.get("text", ""), "meta": r.get("meta", {})})
            ctx_parts.append(r.get("text", ""))
            src = (r.get("meta") or {}).get("source")
            if isinstance(src, str) and src and src not in sources:
                sources.append(src)

        context = "\n\n---\n\n".join(ctx_parts)
        if len(context) > MAX_CTX_CHARS:
            context = context[:MAX_CTX_CHARS] + "\n\n[...context truncated...]"

        system = (
            "You are AURA, an AI assistant. "
            "Answer ONLY using the provided context. "
            "Be concise and direct. "
            "If the context does not contain the answer, say so clearly."
        )
        prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
        answer = await self.client.generate(
            prompt=prompt, system=system, timeout_s=AURA_OLLAMA_TIMEOUT_S
        )

        return {"answer": answer, "sources": sources, "hits": hits}
