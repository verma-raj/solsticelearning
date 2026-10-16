"""Tries many COS_MIN values so you can pick (and justify) the threshold.

Run with:  python sweep_threshold.py [--retrieval hybrid] [--bm25-min 0.1]

Raising COS_MIN lowers the false-found rate (group 3) but raises the miss rate
(groups 1-2). The best value sits where false-found is 0 and the miss rate is smallest.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config
import retrieval
from embedder import OllamaError
from evaluate import apply_gate, headline_metrics, run_retrieval


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep COS_MIN and report both error rates.")
    parser.add_argument("--retrieval", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--bm25-min", type=float, default=None)
    parser.add_argument("--start", type=float, default=0.30)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.02)
    args = parser.parse_args()

    try:
        records = run_retrieval(args.retrieval)  # embed each question once, then re-run only the cheap gate
    except (OllamaError, retrieval.IndexNotBuiltError) as exc:
        print(f"ERROR: {exc}")
        return 1

    rows = []
    for cos_min in np.round(np.arange(args.start, args.stop + args.step / 2, args.step), 2):
        m = headline_metrics(apply_gate(records, float(cos_min), args.bm25_min))
        rows.append({
            "cos_min": float(cos_min),
            "false_found_g3": round(m["false_found_rate_g3"], 3),
            "miss_rate_g12": round(m["miss_rate_g12"], 3),
            "total_error": round(m["false_found_rate_g3"] + m["miss_rate_g12"], 3),
        })
    table = pd.DataFrame(rows)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.SWEEP_CSV, index=False)
    print(f"Retrieval: {args.retrieval} | BM25_MIN={config.BM25_MIN if args.bm25_min is None else args.bm25_min}\n")
    print(table.to_string(index=False))

    # Suggestion: wrong documents are worse than silence, so among the values that let NO
    # group-3 question through, keep those with the fewest group 1-2 misses. Inside that
    # plateau the stricter value costs nothing, so take a high one, two steps in from the edge
    # (the very edge is fragile: one new question could tip it over).
    safe = table[table["false_found_g3"] == 0]
    if safe.empty:
        print("\nNo COS_MIN removes every false-found answer; the gate needs another signal.")
    else:
        plateau = safe[safe["miss_rate_g12"] == safe["miss_rate_g12"].min()]["cos_min"].tolist()
        pick = plateau[max(len(plateau) - 3, 0)]
        print(f"\nSafe plateau: COS_MIN {plateau[0]:.2f} to {plateau[-1]:.2f} "
              f"(false-found 0%, miss rate {safe['miss_rate_g12'].min():.0%}). "
              f"Suggested COS_MIN = {pick:.2f}. Currently set: {config.COS_MIN}")
    print(f"Table saved to {config.SWEEP_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
