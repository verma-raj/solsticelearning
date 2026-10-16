"""The "found / not found" decision.

Default answer is NOT FOUND. A document is only accepted when it passes BOTH:
  1. a semantic check  - its cosine similarity is high enough, and
  2. a lexical check   - the query shares real words with the document.
Requiring two independent signals to agree is what stops confident-looking
but wrong matches (a wrong document is worse than "No document available").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from retrieval import Retriever, get_retriever
from textutils import content_terms


@dataclass
class GateDecision:
    """The outcome, an explanation, and the numbers behind it."""
    found: bool
    reason: str
    scores: dict = field(default_factory=dict)       # numbers for the top document
    accepted: list[dict] = field(default_factory=list)  # documents that passed, best first


def meaningful_terms(query: str, retriever: Retriever) -> list[str]:
    """Query words that can actually identify a document.

    Stopwords ("how", "do", "the") and words found in most documents ("solstice")
    are dropped: matching on them would "prove" relevance for almost any question.
    """
    n_docs = max(len(retriever.documents), 1)
    terms = dict.fromkeys(content_terms(query))  # unique, order kept
    return [t for t in terms if retriever.doc_freq.get(t, 0) / n_docs <= config.COMMON_TERM_MAX_DOC_RATIO]


def _check(doc: dict, terms: list[str], cos_min: float, bm25_min: float, retriever: Retriever) -> tuple[bool, str, dict]:
    """Apply both checks to one document and explain the result."""
    overlap = [t for t in terms if t in retriever.doc_terms.get(doc["doc_id"], set())]
    scores = {
        "doc_id": doc["doc_id"],
        "cosine": doc["cosine_similarity"],
        "bm25": doc["bm25_score"],
        "overlap_terms": overlap,
    }
    semantic_ok = doc["cosine_similarity"] >= cos_min
    lexical_ok = doc["bm25_score"] >= bm25_min or len(overlap) >= config.MIN_OVERLAP_TERMS

    if not semantic_ok:
        why = f"cosine {doc['cosine_similarity']:.3f} is below the minimum {cos_min:.2f}"
    elif not lexical_ok:
        why = (f"cosine {doc['cosine_similarity']:.3f} is fine, but the query shares only {len(overlap)} meaningful "
               f"word(s) with the document (need {config.MIN_OVERLAP_TERMS}) and BM25 {doc['bm25_score']:.3f} "
               f"is below {bm25_min:.2f}")
    else:
        why = (f"cosine {doc['cosine_similarity']:.3f} >= {cos_min:.2f}; "
               f"BM25 {doc['bm25_score']:.3f}, shared words: {', '.join(overlap) or 'none'}")
    return semantic_ok and lexical_ok, why, scores


def decide(
    query: str,
    results: list[dict],
    cos_min: float | None = None,
    bm25_min: float | None = None,
    retriever: Retriever | None = None,
) -> GateDecision:
    """Decide whether the search results contain a real answer.

    The TOP document alone decides found / not found. Lower-ranked documents are
    only listed (in accepted) if they pass the same checks themselves.
    """
    cos_min = config.COS_MIN if cos_min is None else cos_min
    bm25_min = config.BM25_MIN if bm25_min is None else bm25_min
    retriever = retriever or get_retriever()

    if not results:
        return GateDecision(False, "No candidate documents were retrieved for this query.")

    terms = meaningful_terms(query, retriever)
    top_ok, top_why, top_scores = _check(results[0], terms, cos_min, bm25_min, retriever)
    if not top_ok:
        return GateDecision(False, f"NOT FOUND - top document {results[0]['doc_id']}: {top_why}.", top_scores)

    accepted = [results[0]]
    for doc in results[1:]:
        ok, _, _ = _check(doc, terms, cos_min, bm25_min, retriever)
        if ok:
            accepted.append(doc)
    return GateDecision(True, f"FOUND - top document {results[0]['doc_id']}: {top_why}.", top_scores, accepted)
