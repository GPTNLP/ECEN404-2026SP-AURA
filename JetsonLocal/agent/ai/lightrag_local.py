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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import urllib.request

import faiss
from rank_bm25 import BM25Okapi

# ─── Tunables ────────────────────────────────────────────────────────────────

# Ollama connectivity
AURA_OLLAMA_URL    = os.getenv("AURA_OLLAMA_URL", "http://127.0.0.1:11434")
AURA_OLLAMA_TIMEOUT_S = float(os.getenv("AURA_OLLAMA_TIMEOUT_S", "180"))
AURA_KEEP_ALIVE    = os.getenv("AURA_KEEP_ALIVE", "30m")   # keep model loaded
AURA_NUM_GPU       = int(os.getenv("AURA_NUM_GPU", "99"))   # GPU layers to offload

# Answer generation
AURA_NUM_PREDICT   = int(os.getenv("AURA_NUM_PREDICT", "256"))
AURA_NUM_CTX       = int(os.getenv("AURA_NUM_CTX", "4096"))
AURA_TEMPERATURE   = float(os.getenv("AURA_TEMPERATURE", "0.2"))
AURA_NUM_THREAD    = int(os.getenv("AURA_NUM_THREAD", "0"))  # 0 = auto

# Graph extraction (build-time, expensive but one-off)
AURA_GRAPH_EXTRACT       = os.getenv("AURA_GRAPH_EXTRACT", "true").lower() == "true"
AURA_GRAPH_TIMEOUT_S     = float(os.getenv("AURA_GRAPH_TIMEOUT_S", "90"))
AURA_GRAPH_NUM_PREDICT   = int(os.getenv("AURA_GRAPH_NUM_PREDICT", "512"))
AURA_GRAPH_NUM_CTX       = int(os.getenv("AURA_GRAPH_NUM_CTX", "3072"))

# Retrieval
MAX_CTX_CHARS       = int(os.getenv("AURA_MAX_CTX_CHARS", "8000"))
DEFAULT_TOP_K       = int(os.getenv("AURA_TOP_K", "6"))
BM25_REBUILD_EVERY  = int(os.getenv("AURA_BM25_REBUILD_EVERY", "50"))
AURA_LOCAL_TOP_K    = int(os.getenv("AURA_LOCAL_TOP_K", "5"))   # entity matches
AURA_GLOBAL_TOP_K   = int(os.getenv("AURA_GLOBAL_TOP_K", "5"))  # entity FAISS hits

# Chunking — paper uses 1200 chars
AURA_INSERT_CHUNK_SIZE    = int(os.getenv("AURA_INSERT_CHUNK_SIZE", "1200"))
AURA_INSERT_CHUNK_OVERLAP = int(os.getenv("AURA_INSERT_CHUNK_OVERLAP", "200"))

# Batch embedding
AURA_EMBED_BATCH_SIZE = int(os.getenv("AURA_EMBED_BATCH_SIZE", "8"))


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


def _chunk_text(
    text: str,
    max_chars: int = AURA_INSERT_CHUNK_SIZE,
    overlap: int = AURA_INSERT_CHUNK_OVERLAP,
) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        if j >= n:
            break
        i = max(0, j - overlap)
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
    "Extract structured information from text. "
    "Always output only valid JSON. No explanations, no markdown fences."
)

_GRAPH_EXTRACT_PREFIX = """\
Extract entities and relationships from the following text.

Output JSON with EXACTLY this structure:
{
  "entities": [
    {"name": "EntityName", "type": "concept|person|organization|equipment|process|location", "description": "brief description"}
  ],
  "relations": [
    {"src": "EntityA", "tgt": "EntityB", "description": "how they relate", "keywords": ["keyword1", "keyword2"], "strength": 7}
  ],
  "high_level_keywords": ["theme1", "theme2", "theme3"]
}

Rules:
- entity_name: key term, capitalized
- strength: integer 1-10
- high_level_keywords: 2-5 overarching concepts/themes
- If nothing meaningful found, return {"entities": [], "relations": [], "high_level_keywords": []}

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
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw else {}

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
            payload = {"model": self.embed_model, "input": texts}
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
                    {"model": self.embed_model, "prompt": text},
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
        fast: bool = False,
    ) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "num_gpu": AURA_NUM_GPU,  # offload all layers to Jetson GPU
        }
        if AURA_NUM_THREAD > 0:
            opts["num_thread"] = AURA_NUM_THREAD
        if fast:
            opts["mirostat"] = 0  # disable mirostat for deterministic fast output
        return opts

    async def generate(
        self,
        prompt: str,
        system: str = "",
        timeout_s: float = AURA_OLLAMA_TIMEOUT_S,
        num_predict: int = AURA_NUM_PREDICT,
        num_ctx: int = AURA_NUM_CTX,
        temperature: float = AURA_TEMPERATURE,
        fast: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "keep_alive": AURA_KEEP_ALIVE,
            "options": self._make_options(num_predict, num_ctx, temperature, fast),
        }
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
                fast=True,
            )
            return _parse_json_from_llm(raw)
        except Exception as e:
            print(f"[LightRAG] Graph extraction skipped for chunk (non-fatal): {e}")
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

        print(f"[LightRAG] Embedding {len(texts)} entities (batch size {AURA_EMBED_BATCH_SIZE})...")
        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), AURA_EMBED_BATCH_SIZE):
            batch = texts[i : i + AURA_EMBED_BATCH_SIZE]
            embs = await self.client.embed_batch(batch)
            all_embs.extend(embs)

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

        print(f"[LightRAG] Entity index ready — {len(self._entity_rows)} entities")

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
        chunks = _chunk_text(text)
        if not chunks:
            return

        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            await self._insert_one_chunk(
                chunk, meta={**(meta or {}), "chunk_index": idx, "chunk_count": total}
            )

        # Rebuild entity FAISS after all chunks for this document
        if AURA_GRAPH_EXTRACT and self._entity_index_dirty:
            await self._rebuild_entity_index()

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
            raw = await self.client.generate(
                prompt=prompt,
                system=_KEYWORD_EXTRACT_SYSTEM,
                timeout_s=15.0,
                num_predict=80,
                num_ctx=512,
                temperature=0.0,
                fast=True,
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
            for hit in chunk_hits:
                parts.append(hit.get("text", ""))

        context = "\n\n".join(parts)
        if len(context) > MAX_CTX_CHARS:
            context = context[:MAX_CTX_CHARS] + "\n\n[...truncated...]"
        return context

    # ── query ─────────────────────────────────────────────────────────────────

    async def aquery(
        self, query: str, param: Optional[QueryParam] = None
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
        if has_graph and mode in ("local", "hybrid"):
            q_emb, keywords = await asyncio.gather(
                self.client.embed(query),
                self._extract_query_keywords(query),
            )
        else:
            q_emb    = await self.client.embed(query)
            keywords = {"high": [], "low": []}

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
            for idx, score in self._search_bm25(query, top_k=top_k * 2):
                candidates.setdefault(idx, {})["bm25"] = score

        scored: List[Tuple[float, int]] = []
        for idx, d in candidates.items():
            vec   = d.get("vec", 0.0)
            bm    = d.get("bm25", 0.0)
            bm_n  = bm / (abs(bm) + 8.0)
            total = (0.75 * vec + 0.25 * bm_n) if mode == "hybrid" else (
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
            "You are AURA, a helpful lab assistant. "
            "Answer ONLY using the provided context. "
            "Be concise and accurate. "
            "If the context does not contain the answer, say so clearly."
        )
        prompt  = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
        answer  = await self.client.generate(
            prompt=prompt, system=system, timeout_s=AURA_OLLAMA_TIMEOUT_S
        )

        return {
            "answer":         answer,
            "sources":        sources,
            "hits":           chunk_hits,
            "entities_used":  len(entity_hits),
            "relations_used": len(relation_hits),
        }
