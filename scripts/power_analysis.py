#!/usr/bin/env python3
"""How much traffic would it take to ship each of these rewrites?

    PYTHONPATH=src python scripts/power_analysis.py
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from qrw.experiment import (detectable_effect, interleaving_power, ndcg_from_rank,
                            paired_ab, unpaired_ab)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("results/evaluation.json"))
    ap.add_argument("--baseline", default="concat_2", help="the system to beat")
    ap.add_argument("--out", type=Path, default=Path("results/power.json"))
    args = ap.parse_args()

    report = json.loads(args.eval.read_text())
    systems = {k: v for k, v in report.items() if not k.startswith("_")}
    ranks = {k: np.array(v["per_query_rank"]) for k, v in systems.items()}
    ndcg = {k: np.array([ndcg_from_rank(int(r)) for r in v]) for k, v in ranks.items()}
    n_queries = len(next(iter(ranks.values())))
    print(f"{len(systems)} systems, {n_queries:,} queries\n")

    base = args.baseline
    out: dict[str, dict] = {}

    print(f"Traffic needed to detect each system against '{base}' at 80% power, alpha 0.05")
    print(f"{'system':<14}{'dNDCG':>9}{'paired':>12}{'A/B split':>13}{'interleaved':>13}{'vs A/B':>9}")
    print("-" * 70)
    for name in sorted(systems):
        if name == base:
            continue
        p = paired_ab(ndcg[name], ndcg[base])
        u = unpaired_ab(ndcg[name], ndcg[base])
        il = interleaving_power(ranks[name], ranks[base])
        speedup = (u.n_required / il.n_required) if (u.n_required and il.n_required) else float("nan")
        out[name] = {"paired": p.__dict__, "unpaired": u.__dict__, "interleaving": il.__dict__}
        fmt = lambda r: f"{r.n_required:,}" if r.n_required else "n/a"
        print(f"{name:<14}{p.effect:>+9.4f}{fmt(p):>12}{fmt(u):>13}{fmt(il):>13}"
              f"{speedup:>8.1f}x" if speedup == speedup else
              f"{name:<14}{p.effect:>+9.4f}{fmt(p):>12}{fmt(u):>13}{fmt(il):>13}{'n/a':>9}")

    print("\nSmallest nDCG difference each traffic level can resolve (paired, vs "
          f"'{base}')")
    sd = float((ndcg["concat_4"] - ndcg[base]).std(ddof=1))
    print(f"{'queries':>12}{'detectable dNDCG':>20}{'as % of baseline':>20}")
    print("-" * 52)
    baseline_mean = float(ndcg[base].mean())
    for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000):
        d = detectable_effect(sd, n)
        print(f"{n:>12,}{d:>20.5f}{100 * d / baseline_mean:>19.3f}%")
    out["_detectable"] = {str(n): detectable_effect(sd, n)
                          for n in (1_000, 10_000, 100_000, 1_000_000, 10_000_000)}
    out["_baseline_ndcg"] = baseline_mean

    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
