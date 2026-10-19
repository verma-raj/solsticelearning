"""Hybrid search: BM25 (keywords) + vector search (meaning), merged with Reciprocal Rank Fusion.

search() returns one result per document, best document first. It never decides
"found / not found" - that is gate.py's job.
"""
from __future__ import annotations

import json
import threading

import numpy as np
from rank_bm25 import BM25Okapi

import config
from embedder import embed
from indexlock import INDEX_LOCK
from textutils import tokenize


class IndexNotBuiltError(RuntimeError):
    """Raised when the index files do not exist yet."""


class Retriever:
    """Loads the index files once and answers queries."""

    def __init__(self) -> None:
        # Read the files under the lock so a rebuild that is writing them cannot be caught half-done.
        with INDEX_LOCK:
            if not config.CHUNKS_JSON.exists() or not config.EMBEDDINGS_NPY.exists():
                raise IndexNotBuiltError(
                    "The search index is not ready yet. Wait a few seconds and try again, or run: python ingest.py"
                )
            data = json.loads(config.CHUNKS_JSON.read_text(encoding="utf-8"))
            self.matrix: np.ndarray = np.load(config.EMBEDDINGS_NPY)
            self.collection = self._open_chroma()
        self.fingerprint: str | None = data.get("fingerprint")  # which version of the documents this index is for
        self.documents: dict[str, dict] = data["documents"]
        self.chunks: list[dict] = data["chunks"]
        self._index_of = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

        self.bm25 = BM25Okapi(
            [tokenize(c["index_text"]) for c in self.chunks], k1=config.BM25_K1, b=config.BM25_B
        )
        self._max_idf = max(self.bm25.idf.values()) if self.bm25.idf else 1.0

        # Words of each whole document (used by the gate's lexical-overlap check),
        # plus how many documents each word appears in.
        self.doc_terms: dict[str, set[str]] = {doc_id: set(tokenize(info["title"])) for doc_id, info in self.documents.items()}
        for chunk in self.chunks:
            self.doc_terms[chunk["doc_id"]].update(tokenize(chunk["text"]))
        self.doc_freq: dict[str, int] = {}
        for terms in self.doc_terms.values():
            for term in terms:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1


    @staticmethod
    def _open_chroma():
        """Open the ChromaDB collection, or return None so we fall back to NumPy."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            return client.get_collection(config.CHROMA_COLLECTION)
        except Exception:
            return None

    # ---------------------------------------------------------- rankings ----
    def _bm25_ranking(self, tokens: list[str]) -> tuple[list[int], np.ndarray]:
        """Chunk indexes ordered by BM25, plus every chunk's normalised BM25 score (0..1)."""
        unique = list(dict.fromkeys(tokens))  # repeated query words must not count twice
        raw = np.asarray(self.bm25.get_scores(unique), dtype=np.float32)

        # Normalise by the best score a chunk could possibly get for THIS query (every query
        # word present, term frequency -> infinity). 0 = nothing matched, 1 = everything matched,
        # so BM25_MIN means the same thing for every query. Unknown words count as the rarest-term max.
        bound = sum(
            (self.bm25.idf.get(t, self._max_idf)) * (config.BM25_K1 + 1) for t in unique
        )
        normalised = np.clip(raw / bound, 0.0, 1.0) if bound > 0 else np.zeros_like(raw)

        order = [int(i) for i in np.argsort(-raw)[: config.TOP_K] if raw[i] > 0]  # zero score = no match at all
        return order, normalised

    def _vector_ranking(self, query_vec: np.ndarray) -> list[int]:
        """Chunk indexes ordered by vector similarity (ChromaDB, or NumPy if unavailable)."""
        if self.collection is not None:
            try:
                res = self.collection.query(
                    query_embeddings=[query_vec.tolist()], n_results=min(config.TOP_K, len(self.chunks))
                )
                return [self._index_of[i] for i in res["ids"][0]]
            except Exception:
                pass  # fall through to the exact NumPy search
        return [int(i) for i in np.argsort(-(self.matrix @ query_vec))[: config.TOP_K]]

    # ------------------------------------------------------------ search ----
    def search(self, query: str, method: str = "hybrid") -> list[dict]:
        """Return up to MAX_DOCS_RETURNED documents, best first.

        method: "bm25", "vector" or "hybrid" (default).
        """
        if method not in ("bm25", "vector", "hybrid"):
            raise ValueError("method must be 'bm25', 'vector' or 'hybrid'")
        if not query.strip():
            return []

        query_vec = embed([query], kind="query")[0]
        cosines = self.matrix @ query_vec  # unit vectors, so this IS cosine similarity
        bm25_order, bm25_scores = self._bm25_ranking(tokenize(query))

        rankings = []
        if method in ("bm25", "hybrid"):
            rankings.append(bm25_order)
        if method in ("vector", "hybrid"):
            rankings.append(self._vector_ranking(query_vec))

        # Reciprocal Rank Fusion: a chunk scores 1/(k + rank) in every list it appears in.
        # Using ranks (not raw scores) lets BM25 and cosine be combined without rescaling.
        rrf: dict[int, float] = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)

        # Collapse to one result per document, keeping each document's best chunk.
        ordered = sorted(rrf, key=lambda i: (-rrf[i], -float(cosines[i])))
        results: list[dict] = []
        seen: set[str] = set()
        for idx in ordered:
            chunk = self.chunks[idx]
            if chunk["doc_id"] in seen:
                continue
            seen.add(chunk["doc_id"])
            results.append({
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "heading": chunk["heading"],
                "excerpt": chunk["text"],  # verbatim from the document
                "cosine_similarity": round(float(cosines[idx]), 4),
                "bm25_score": round(float(bm25_scores[idx]), 4),
                "rrf_score": round(rrf[idx], 5),
            })
            if len(results) == config.MAX_DOCS_RETURNED:
                break
        return results


_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    """Load the index once and reuse it (loading is the slow part)."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


def reload_retriever() -> Retriever:
    """Load the index files again and switch to them. Searches running meanwhile finish on the old copy."""
    global _retriever
    fresh = Retriever()  # built completely before the switch, so there is never a gap
    _retriever = fresh
    return fresh


def loaded_fingerprint() -> str | None:
    """The document version the app is currently searching, or None if no index is loaded."""
    return _retriever.fingerprint if _retriever is not None else None


def search(query: str, method: str = "hybrid") -> list[dict]:
    """Module-level shortcut used by the API and the evaluation scripts."""
    return get_retriever().search(query, method)
