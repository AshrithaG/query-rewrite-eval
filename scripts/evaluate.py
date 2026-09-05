#!/usr/bin/env python3
"""Score every rewriter twice: ROUGE against the reference, retrieval against
the corpus. Then ask whether the two agree.

    PYTHONPATH=src python scripts/evaluate.py [--limit N] [--rewrites FILE.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qrw.data import build_corpus, load_turns
from qrw.metrics import agreement, per_query_rouge, rouge_all, system_ranking_agreement
from qrw.retrieval import Index, metrics, rank_of
from qrw.rewriters import BASELINES

K = 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="use only the first N turns")
    ap.add_argument("--rewrites", type=Path, action="append", default=[],
                    help="JSON {uid: rewrite} from a model, repeatable")
    ap.add_argument("--out", type=Path, default=Path("results/evaluation.json"))
    args = ap.parse_args()

    turns = load_turns("test", needs_context=True)
    if args.limit:
        turns = turns[: args.limit]
    docs, relevant = build_corpus(turns)
    print(f"{len(turns)} turns, corpus of {len(docs)} documents\n")

    index = Index(docs)
    systems: dict[str, list[str]] = {name: [fn(t) for t in turns] for name, fn in BASELINES.items()}
    for path in args.rewrites:
        loaded = json.loads(path.read_text())
        name = path.stem
        missing = sum(1 for t in turns if t.uid not in loaded)
        if missing:
            print(f"  warning: {name} is missing {missing} of {len(turns)} turns; "
                  f"falling back to the raw question for those")
        systems[name] = [loaded.get(t.uid, t.question) for t in turns]

    refs = [t.gold_rewrite for t in turns]
    report: dict[str, dict] = {}
    rouge_by_system, retr_by_system = {}, {}

    print(f"{'system':<14}{'ROUGE-L':>9}{'R@10':>8}{'MRR@10':>9}{'nDCG@10':>9}{'len':>7}")
    print("-" * 56)
    for name, queries in systems.items():
        hits = index.search(queries, k=K)
        ranks = [rank_of(hits[i], relevant[t.uid]) for i, t in enumerate(turns)]
        retr = metrics(ranks, k=K)
        rg = rouge_all(queries, refs)
        per_q_rouge = per_query_rouge(queries, refs)
        per_q_retr = np.array([1.0 / np.log2(r + 1) if r else 0.0 for r in ranks])

        # per-query ranks are what any power or significance analysis needs;
        # system means throw away exactly the variance those questions are about
        report[name] = {
            "per_query_rank": [int(r) if r else 0 for r in ranks],
            "per_query_rouge": [float(x) for x in per_q_rouge],
            "rouge": rg, "retrieval": retr,
            "mean_query_words": float(np.mean([len(q.split()) for q in queries])),
            "per_query_agreement": agreement(per_q_rouge, per_q_retr),
        }
        # gold is the reference, so its ROUGE is 1.0 by construction; excluding it
        # from the system-level ranking keeps that artifact out of the comparison
        if name != "gold":
            rouge_by_system[name] = rg["rougeL"]
            retr_by_system[name] = retr[f"ndcg@{K}"]

        print(f"{name:<14}{rg['rougeL']:>9.3f}{retr[f'recall@{K}']:>8.3f}"
              f"{retr[f'mrr@{K}']:>9.3f}{retr[f'ndcg@{K}']:>9.3f}"
              f"{report[name]['mean_query_words']:>7.1f}")

    rank_ag = system_ranking_agreement(rouge_by_system, retr_by_system)
    report["_system_ranking"] = rank_ag
    print(f"\nsystem-level ranking agreement (gold excluded, {rank_ag['n_systems']} systems)")
    print(f"  Kendall tau between ROUGE-L and nDCG@{K}: {rank_ag['kendall_tau']:+.3f} (p={rank_ag['p']:.3f})")
    print(f"  best by ROUGE-L: {rank_ag['best_by_rouge']}   best by nDCG: {rank_ag['best_by_retrieval']}")
    print(f"  nDCG given up by trusting ROUGE to pick: {rank_ag['regret']:.3f}")

    report["_uids"] = [t.uid for t in turns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
