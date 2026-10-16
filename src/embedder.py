"""Turns text into vectors using the local Ollama server (no cloud, no Hugging Face).

The rest of the app only calls embed(); to swap the embedding method, change this file.
"""
from __future__ import annotations

import numpy as np
import requests

import config


class OllamaError(RuntimeError):
    """Raised with a friendly, actionable message when Ollama cannot embed text."""


# One shared session. trust_env=False makes requests ignore the corporate proxy
# settings, otherwise calls to localhost can be sent to the proxy and fail.
_session = requests.Session()
_session.trust_env = False


def get_session() -> requests.Session:
    """Return the shared no-proxy session (also used by the /health check)."""
    return _session


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Send one batch to POST /api/embed and return the raw vectors."""
    try:
        resp = _session.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": texts},
            timeout=config.EMBED_TIMEOUT_SECONDS,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {config.OLLAMA_URL}. "
            "Start the Ollama app (or run `ollama serve`) and try again."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            "Ollama took too long to answer. It may still be loading the model; wait a moment and retry."
        ) from exc

    if resp.status_code == 404:
        raise OllamaError(
            f"The model '{config.EMBED_MODEL}' is not installed. Run: ollama pull {config.EMBED_MODEL}"
        )
    if not resp.ok:
        raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

    embeddings = resp.json().get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise OllamaError("Ollama returned an unexpected response (wrong number of embeddings).")
    return embeddings


def embed(texts: list[str], kind: str = "document") -> np.ndarray:
    """Embed texts and return an (n, dim) float32 array of L2-normalised vectors.

    kind is "document" for text being indexed or "query" for a user's question.
    """
    if kind not in ("document", "query"):
        raise ValueError("kind must be 'document' or 'query'")
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    prefix = config.DOCUMENT_PREFIX if kind == "document" else config.QUERY_PREFIX
    prefixed = [prefix + t for t in texts]

    vectors: list[list[float]] = []
    for start in range(0, len(prefixed), config.EMBED_BATCH_SIZE):
        vectors.extend(_embed_batch(prefixed[start:start + config.EMBED_BATCH_SIZE]))

    matrix = np.asarray(vectors, dtype=np.float32)
    # Unit-length vectors make "dot product" identical to "cosine similarity".
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def check_ollama() -> dict:
    """Report whether Ollama is reachable and the embedding model is installed."""
    try:
        resp = _session.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return {"reachable": False, "model_installed": False, "detail": str(exc)}
    names = [m.get("name", "") for m in resp.json().get("models", [])]
    installed = any(n.split(":")[0] == config.EMBED_MODEL.split(":")[0] for n in names)
    return {"reachable": True, "model_installed": installed, "detail": ""}
