#!/usr/bin/env python3
"""Partition a corpus into shard indexes on disk.

Global document frequencies are computed across the whole collection and stored
with every shard, so a shard scores a document the same way the unsharded index
would. See src/qrw/sharded.py for what happens when they are not.

    PYTHONPATH=src python scripts/build_shards.py --shards 1 2 4 8 --source qrecc
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qrw.data import build_corpus, load_turns
from qrw.invindex import DocFreq, Shard, tokenize
from qrw.sharded import ShardedIndex


def load_source(name: str, limit: int) -> list[tuple[str, str]]:
    if name == "qrecc":
        docs, _ = build_corpus(load_turns("test", needs_context=True))
        return [(f"q{i}", d) for i, d in enumerate(docs)]
    if name == "msmarco":
        from datasets import load_dataset
        ds = load_dataset("BeIR/msmarco", "corpus", split="corpus", streaming=True)
        out = []
        for i, r in enumerate(ds):
            if limit and i >= limit:
                break
            text = (r.get("title", "") + " " + r.get("text", "")).strip()
            out.append((str(r["_id"]), text))
        return out
    raise SystemExit(f"unknown source {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="qrecc", choices=["qrecc", "msmarco"])
    ap.add_argument("--limit", type=int, default=0, help="cap documents, 0 = all")
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out", type=Path, default=Path("data/shards"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = load_source(args.source, args.limit)
    print(f"{len(docs)} documents loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    gdf = DocFreq()
    for _, text in docs:
        gdf.add_document(tokenize(text))
    print(f"global df over {gdf.n_docs} docs, {len(gdf.df)} terms, "
          f"{time.perf_counter()-t0:.1f}s")

    for n in args.shards:
        t0 = time.perf_counter()
        buckets: list[list[tuple[str, str]]] = [[] for _ in range(n)]
        for ext_id, text in docs:
            buckets[ShardedIndex.assign(ext_id, n)].append((ext_id, text))

        d = args.out / f"n{n}"
        d.mkdir(parents=True, exist_ok=True)
        sizes = []
        for i, bucket in enumerate(buckets):
            s = Shard()
            s.build(bucket)
            (d / f"shard-{i}.pkl").write_bytes(
                pickle.dumps({"shard": s, "global_df": gdf}, protocol=5))
            sizes.append(s.memory_bytes() / 1e6)
        spread = max(len(b) for b in buckets) / max(1, min(len(b) for b in buckets))
        print(f"  n={n:>2}: built in {time.perf_counter()-t0:>6.1f}s  "
              f"docs/shard {min(len(b) for b in buckets)}-{max(len(b) for b in buckets)} "
              f"(imbalance {spread:.2f}x)  index {sum(sizes):.0f} MB total")


if __name__ == "__main__":
    main()
