"""Tiny text helpers shared by ingest, retrieval and the gate."""
from __future__ import annotations

import re

import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Very light plural stripping ("exports" -> "export") so singular and plural match."""
    # Only plurals are handled: a full stemmer can merge unrelated words, which would blur the gate.
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case the text and split it into stemmed word tokens (used for BM25)."""
    return [stem(w) for w in _WORD_RE.findall(text.lower())]


# Stopwords go through the same stemmer so they match the tokens they should remove.
STOPWORDS = {stem(w) for w in config.STOPWORDS}


def content_terms(text: str) -> list[str]:
    """Tokens that are not stopwords and are at least 3 characters long (or a number)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) >= 3 or t.isdigit())]
