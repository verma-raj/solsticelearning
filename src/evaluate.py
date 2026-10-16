"""Scores the document finder against kb/eval_questions.csv.

Run with:  python evaluate.py [--retrieval bm25|vector|hybrid] [--cos-min 0.6] [--bm25-min 0.1]

Groups 1 and 2 must return an expected document, group 3 must return nothing,
group 4 (ambiguous) is only recorded for manual review.
"""
from __future__ import annotations

import argparse
import re
import sys

import pandas as pd

import config
import gate
import retrieval
from embedder import OllamaError

DOC_ID_RE = re.compile(r"[A-Z]{3}-\d{2}")


def parse_expected(text: str) -> list[str]:
    """Pull doc ids out of cells like "PWD-01 (web), PWD-04" or "EXP-02, EXP-03"."""
    return DOC_ID_RE.findall(text or "")


def run_retrieval(method: str) -> list[dict]:
    """Search once per question. The expensive part (embedding) is done here, one time."""
    questions = pd.read_csv(config.EVAL_CSV, dtype=str, keep_default_na=False)
    records = []
    for row in questions.to_dict("records"):
        row["results"] = retrieval.search(row["question"], method)
        records.append(row)
    return records


def apply_gate(records: list[dict], cos_min: float | None = None, bm25_min: float | None = None) -> pd.DataFrame:
    """Apply the gate with the given thresholds and score every question."""
    rows = []
    for rec in records:
        results = rec["results"]
        decision = gate.decide(rec["question"], results, cos_min, bm25_min)
        expected = parse_expected(rec["expected_docs"])
        raw_ids = [r["doc_id"] for r in results]            # what retrieval ranked, before the gate
        shown_ids = [r["doc_id"] for r in decision.accepted[: config.MAX_DOCS_SHOWN]]
        group = rec["group"]

        if group in ("1_clear", "2_paraphrased_vague"):
            correct = decision.found and bool(set(shown_ids) & set(expected))
        elif group == "3_not_covered":
            correct = not decision.found
        else:
            correct = "REVIEW"  # ambiguous: a human decides

        rows.append({
            "id": rec["id"],
            "group": group,
            "question": rec["question"],
            "expected_behavior": rec["expected_behavior"],
            "expected_docs": rec["expected_docs"],
            "found": decision.found,
            "returned_docs": "; ".join(shown_ids),
            "retrieved_top3": "; ".join(raw_ids[:3]),
            "top1_hit": bool(raw_ids[:1] and raw_ids[0] in expected),
            "top3_hit": bool(set(raw_ids[:3]) & set(expected)),
            "top_cosine": results[0]["cosine_similarity"] if results else None,
            "top_bm25": results[0]["bm25_score"] if results else None,
            "correct": correct,
            "reason": decision.reason,
            "key_facts_or_reason": rec["key_facts_or_reason"],
        })
    return pd.DataFrame(rows)


def headline_metrics(df: pd.DataFrame) -> dict:
    """The four numbers that matter for the case study."""
    g3 = df[df["group"] == "3_not_covered"]
    g12 = df[df["group"].isin(["1_clear", "2_paraphrased_vague"])]
    return {
        "false_found_rate_g3": float(g3["found"].mean()) if len(g3) else 0.0,  # wrongly returned a document
        "miss_rate_g12": float((~g12["found"]).mean()) if len(g12) else 0.0,   # wrongly said "No document available"
        "top1_hit_rate_g12": float(g12["top1_hit"].mean()) if len(g12) else 0.0,
        "top3_hit_rate_g12": float(g12["top3_hit"].mean()) if len(g12) else 0.0,
        "n_g3": len(g3),
        "n_g12": len(g12),
    }


def summary_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group: how many were answered, and how many were right."""
    out = []
    for group, part in df.groupby("group"):
        out.append({
            "group": group,
            "questions": len(part),
            "returned_a_document": int(part["found"].sum()),
            "correct": int((part["correct"] == True).sum()) if group != "4_ambiguous" else "manual review",  # noqa: E712
            "top1_hit": int(part["top1_hit"].sum()),
            "top3_hit": int(part["top3_hit"].sum()),
        })
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the document finder.")
    parser.add_argument("--retrieval", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--cos-min", type=float, default=None, help=f"override COS_MIN (default {config.COS_MIN})")
    parser.add_argument("--bm25-min", type=float, default=None, help=f"override BM25_MIN (default {config.BM25_MIN})")
    args = parser.parse_args()

    try:
        records = run_retrieval(args.retrieval)
    except (OllamaError, retrieval.IndexNotBuiltError) as exc:
        print(f"ERROR: {exc}")
        return 1

    df = apply_gate(records, args.cos_min, args.bm25_min)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel opens it cleanly

    cos = config.COS_MIN if args.cos_min is None else args.cos_min
    bm25 = config.BM25_MIN if args.bm25_min is None else args.bm25_min
    m = headline_metrics(df)
    pd.set_option("display.width", 160)
    print(f"Retrieval: {args.retrieval} | COS_MIN={cos} | BM25_MIN={bm25}\n")
    print(summary_by_group(df).to_string(index=False))
    print("\nHeadline metrics")
    print(f"  False-found rate (group 3)   : {m['false_found_rate_g3']:.0%}  (lower is better; wrongly returned a document)")
    print(f"  Miss rate (groups 1-2)       : {m['miss_rate_g12']:.0%}  (lower is better; wrongly said 'No document available')")
    print(f"  Top-1 hit rate (groups 1-2)  : {m['top1_hit_rate_g12']:.0%}  (retrieval ranking, before the gate)")
    print(f"  Top-3 hit rate (groups 1-2)  : {m['top3_hit_rate_g12']:.0%}")

    print("\nGroup 4 (ambiguous) - for manual review")
    for r in df[df["group"] == "4_ambiguous"].itertuples():
        print(f"  {r.id}: {r.question}\n      expected {r.expected_behavior} / {r.expected_docs} -> "
              f"{'returned ' + r.returned_docs if r.found else 'No document available'}")

    wrong = df[df["correct"] == False]  # noqa: E712
    if len(wrong):
        print("\nWrong answers")
        for r in wrong.itertuples():
            print(f"  {r.id} [{r.group}] {r.question}\n      {r.reason}")
    print(f"\nPer-question results saved to {config.RESULTS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
