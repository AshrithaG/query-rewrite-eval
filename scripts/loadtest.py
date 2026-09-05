#!/usr/bin/env python3
"""Latency and failure behaviour of the sharded service.

Three questions, in order of how much they matter.

1. What does fan-out do to latency? Splitting the corpus makes each shard's work
   smaller, so the median should improve. But a query is not done until its
   slowest shard is done, so the tail is a maximum over N samples and should get
   worse as N grows. Median and tail are expected to move in opposite directions.

2. What does one slow shard do? In a fan-out design a single straggler sets the
   latency of every query that touches it, which with full fan-out is all of them.

3. What happens when a shard dies? With a deadline the coordinator returns a
   partial answer rather than hanging, and the question becomes how much of the
   result is actually lost.

    PYTHONPATH=src python scripts/loadtest.py --shards 1 2 4 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qrw.data import load_turns  # noqa: E402
from qrw.service import Coordinator, ShardEndpoint  # noqa: E402

BASE_PORT = 9300


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def start_shards(n: int, index_dir: Path) -> list[subprocess.Popen]:
    procs = []
    for i in range(n):
        procs.append(subprocess.Popen(
            [sys.executable, "scripts/serve_shard.py", "--shard", str(i), "--of", str(n),
             "--port", str(BASE_PORT + i), "--index-dir", str(index_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**__import__("os").environ, "PYTHONPATH": "src"}))
    return procs


def wait_ready(n: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            for i in range(n):
                httpx.get(f"http://127.0.0.1:{BASE_PORT+i}/health", timeout=1.0).raise_for_status()
            return True
        except Exception:
            time.sleep(0.5)
    return False


async def run_load(coord: Coordinator, queries: list[str], concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    incomplete = 0

    async def one(q: str):
        nonlocal incomplete
        async with sem:
            r = await coord.search(q, k=10)
            lat.append(r["latency_ms"])
            if not r["complete"]:
                incomplete += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(q) for q in queries))
    wall = time.perf_counter() - t0
    return {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99),
            "mean": st.mean(lat), "qps": len(queries) / wall, "incomplete": incomplete}


async def main_async(args) -> None:
    queries = [t.gold_rewrite for t in load_turns("test")[: args.queries]]
    print(f"{len(queries)} queries\n")
    report: dict[str, dict] = {}

    print(f"{'shards':>7}{'conc':>6}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'qps':>9}")
    print("-" * 49)
    for n in args.shards:
        procs = start_shards(n, args.index_dir)
        try:
            if not wait_ready(n):
                print(f"  n={n}: shards did not come up"); continue
            eps = [ShardEndpoint(f"http://127.0.0.1:{BASE_PORT+i}", i) for i in range(n)]
            coord = Coordinator(eps, deadline_ms=args.deadline)
            await run_load(coord, queries[:100], 8)  # warm
            for c in args.concurrency:
                m = await run_load(coord, queries, c)
                report[f"n{n}_c{c}"] = m
                print(f"{n:>7}{c:>6}{m['p50']:>9.1f}{m['p95']:>9.1f}{m['p99']:>9.1f}{m['qps']:>9.0f}")
            await coord.close()
        finally:
            for p in procs:
                p.terminate()
            for p in procs:
                p.wait()

    # faults, on the largest configuration
    n = max(args.shards)
    if n > 1:
        print(f"\nfault injection at {n} shards, concurrency {args.concurrency[-1]}")
        procs = start_shards(n, args.index_dir)
        try:
            wait_ready(n)
            eps = [ShardEndpoint(f"http://127.0.0.1:{BASE_PORT+i}", i) for i in range(n)]
            coord = Coordinator(eps, deadline_ms=args.deadline)
            await run_load(coord, queries[:100], 8)

            base = await run_load(coord, queries, args.concurrency[-1])
            print(f"  healthy          p50 {base['p50']:6.1f}  p99 {base['p99']:6.1f}  "
                  f"qps {base['qps']:5.0f}  incomplete {base['incomplete']}")

            httpx.post(f"http://127.0.0.1:{BASE_PORT}/inject/slow", params={"ms": 100})
            slow = await run_load(coord, queries, args.concurrency[-1])
            print(f"  1 shard +100ms   p50 {slow['p50']:6.1f}  p99 {slow['p99']:6.1f}  "
                  f"qps {slow['qps']:5.0f}  incomplete {slow['incomplete']}")
            httpx.post(f"http://127.0.0.1:{BASE_PORT}/inject/slow", params={"ms": 0})

            procs[0].terminate(); procs[0].wait()
            dead = await run_load(coord, queries, args.concurrency[-1])
            print(f"  1 shard killed   p50 {dead['p50']:6.1f}  p99 {dead['p99']:6.1f}  "
                  f"qps {dead['qps']:5.0f}  incomplete {dead['incomplete']}/{len(queries)}")
            report["fault"] = {"healthy": base, "slow_100ms": slow, "one_dead": dead}
        finally:
            for p in procs:
                if p.poll() is None:
                    p.terminate(); p.wait()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--deadline", type=float, default=250.0)
    ap.add_argument("--index-dir", type=Path, default=Path("data/shards"))
    ap.add_argument("--out", type=Path, default=Path("results/loadtest.json"))
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
