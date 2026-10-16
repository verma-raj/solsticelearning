"""Reads the Markdown docs, splits them into chunks and builds the search indexes.

Run with:  python ingest.py        (rebuilds everything from scratch)
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi

import config
from embedder import OllamaError, embed
from textutils import tokenize

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


# ----------------------------------------------------------------- parsing ----
def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split a file into its YAML-style header (between --- lines) and the body.

    A tiny hand-written parser: the headers are plain "key: value" lines, and a
    full YAML library would choke on titles that contain a colon.
    """
    lines = text.lstrip("﻿").splitlines()
    meta: dict = {}
    if not lines or lines[0].strip() != "---":
        return meta, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body = "\n".join(lines[i + 1:])
            return meta, body
        key, sep, value = lines[i].partition(":")  # split on the FIRST colon only
        if sep:
            meta[key.strip()] = value.strip().strip("\"'")
    return {}, text  # no closing --- found: treat the whole file as body


def load_documents() -> list[dict]:
    """Read every .md file in the docs folder into {doc_id, title, file, ...}."""
    files = sorted(config.DOCS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {config.DOCS_DIR}")
    documents = []
    for path in files:
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append({
            "doc_id": meta.get("doc_id") or path.stem.split("_")[0],
            "title": meta.get("title") or path.stem,
            "topic": meta.get("topic", ""),
            "last_updated": meta.get("last_updated", ""),
            "file": path.name,
            "body": body,
        })
    return documents


# ---------------------------------------------------------------- chunking ----
def split_sections(body: str) -> list[tuple[str, str]]:
    """Split the body at Markdown headings into (heading, text) pairs, skipping empty ones."""
    sections: list[tuple[str, str]] = []
    heading, buffer = "", []
    for line in body.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            sections.append((heading, "\n".join(buffer).strip()))
            heading, buffer = match.group(2), []
        else:
            buffer.append(line)
    sections.append((heading, "\n".join(buffer).strip()))
    return [(h, t) for h, t in sections if t]


def split_long_text(text: str, max_words: int) -> list[str]:
    """Break text that is longer than max_words into paragraph-sized pieces.

    Paragraphs are packed together until the limit is reached. Splitting only at
    blank lines keeps tables and lists in one piece and the text verbatim.
    """
    if len(text.split()) <= max_words:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in re.split(r"\n\s*\n", text):
        words = len(paragraph.split())
        if current and count + words > max_words:
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words  # a single oversized paragraph stays whole rather than being cut mid-sentence
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_document(doc: dict) -> list[dict]:
    """Turn one document into chunks, each carrying its metadata."""
    chunks = []
    for heading, section_text in split_sections(doc["body"]):
        for piece in split_long_text(section_text, config.CHUNK_MAX_WORDS):
            # The document title is prepended to the TEXT THAT IS SEARCHED so a chunk like
            # "Click Save" still knows which topic it belongs to. The text SHOWN to the
            # user ("text") stays exactly as written in the document.
            label = doc["title"] if heading in ("", doc["title"]) else f"{doc['title']} - {heading}"
            chunks.append({
                "chunk_id": f"{doc['doc_id']}-c{len(chunks)}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "heading": heading,
                "text": piece,
                "index_text": f"{label}\n{piece}",
            })
    return chunks


# ------------------------------------------------------------------ indexes ----
def build_vector_index(chunks: list[dict], vectors: np.ndarray) -> str:
    """Store vectors in ChromaDB (cosine space). Returns a short description of what was built."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(config.CHROMA_COLLECTION)  # start clean on every rebuild
        except Exception:
            pass  # collection did not exist yet
        collection = client.create_collection(config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in chunks],
        )
        return f"ChromaDB ({collection.count()} vectors) in {config.CHROMA_DIR}"
    except Exception as exc:  # ChromaDB missing or broken: retrieval.py will use NumPy instead
        return f"ChromaDB unavailable ({exc}); using the NumPy fallback only"


def build_index() -> dict:
    """Rebuild the whole index and return a summary of what was created."""
    documents = load_documents()
    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Embeddings: always saved as a NumPy file; it is the fallback index and is
    #    also used to read the exact cosine similarity of any chunk.
    vectors = embed([c["index_text"] for c in chunks], kind="document")
    np.save(config.EMBEDDINGS_NPY, vectors)

    # 2) Vector index in ChromaDB.
    vector_info = build_vector_index(chunks, vectors)

    # 3) BM25 keyword index. It builds in milliseconds, so retrieval.py rebuilds it
    #    from chunks.json at start-up; we build it here too to prove the data is valid.
    BM25Okapi([tokenize(c["index_text"]) for c in chunks])

    # 4) Cache documents + chunks as JSON (the file the web app and evaluator load).
    registry = {d["doc_id"]: {k: d[k] for k in ("title", "topic", "last_updated", "file")} for d in documents}
    config.CHUNKS_JSON.write_text(
        json.dumps({"documents": registry, "chunks": chunks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    words = [len(c["text"].split()) for c in chunks]
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "avg_words": round(sum(words) / len(words)),
        "max_words": max(words),
        "dim": int(vectors.shape[1]),
        "vector_info": vector_info,
    }


def main() -> int:
    try:
        info = build_index()
    except OllamaError as exc:
        print(f"ERROR: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Documents parsed : {info['documents']}")
    print(f"Chunks created   : {info['chunks']} (avg {info['avg_words']} words, longest {info['max_words']})")
    print(f"BM25 index       : built over {info['chunks']} chunks")
    print(f"Vector index     : {info['vector_info']}")
    print(f"Embedding size   : {info['dim']} dimensions")
    print(f"Chunk cache      : {config.CHUNKS_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
