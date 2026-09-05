"""The retrieval service: shard servers, and a coordinator that fans out to them.

Structure mirrors how a real sharded search tier works. Each shard is its own
process holding its own slice of the index and answering only about that slice.
The coordinator holds no index at all: it fans a query out to every shard,
waits, and merges. That makes query latency the *maximum* over shards rather
than the mean, which is the property the load test is there to expose.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import time
from dataclasses import dataclass

import httpx


@dataclass
class ShardEndpoint:
    url: str
    shard_id: int
    replica: int = 0


class Coordinator:
    """Fans out, merges, and survives shards that are slow or gone.

    `deadline_ms` is the point of the whole design. A search tier cannot wait
    for its slowest shard, because with enough shards one of them is always
    slow. It returns what it has and reports what it lost, so the caller can
    tell a complete answer from a degraded one.
    """

    def __init__(self, endpoints: list[ShardEndpoint], deadline_ms: float = 250.0,
                 timeout_s: float = 5.0):
        self.by_shard: dict[int, list[ShardEndpoint]] = {}
        for e in endpoints:
            self.by_shard.setdefault(e.shard_id, []).append(e)
        self.deadline_ms = deadline_ms
        self._client = httpx.AsyncClient(timeout=timeout_s)
        self._rr = 0

    async def close(self) -> None:
        await self._client.aclose()

    def _pick(self, shard_id: int) -> ShardEndpoint:
        """Round-robin across replicas. Enough to spread load; deliberately not
        least-outstanding-request, so the tail-latency effect stays visible."""
        reps = self.by_shard[shard_id]
        self._rr += 1
        return reps[self._rr % len(reps)]

    async def _one(self, shard_id: int, query: str, k: int) -> list[tuple[str, float]]:
        ep = self._pick(shard_id)
        r = await self._client.get(f"{ep.url}/search", params={"q": query, "k": k})
        r.raise_for_status()
        return [(d["id"], d["score"]) for d in r.json()["results"]]

    async def search(self, query: str, k: int = 10) -> dict:
        started = time.perf_counter()
        tasks = {asyncio.create_task(self._one(s, query, k)): s for s in self.by_shard}
        done, pending = await asyncio.wait(tasks, timeout=self.deadline_ms / 1000.0)

        merged: list[tuple[str, float]] = []
        failed: list[int] = []
        for t in done:
            try:
                merged.extend(t.result())
            except Exception:
                failed.append(tasks[t])
        for t in pending:
            t.cancel()
            failed.append(tasks[t])

        return {
            "results": heapq.nlargest(k, merged, key=lambda x: x[1]),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "shards_total": len(self.by_shard),
            "shards_missing": sorted(failed),
            "complete": not failed,
        }


def make_app(shard, df, shard_id: int, slow_ms: float = 0.0):
    """A shard server. `slow_ms` exists so the load test can inject a straggler
    without killing the process, which is the more common production failure."""
    from fastapi import FastAPI

    app = FastAPI(title=f"shard-{shard_id}")
    state = {"slow_ms": slow_ms, "queries": 0}

    @app.get("/search")
    def search(q: str, k: int = 10):
        t0 = time.perf_counter()
        if state["slow_ms"]:
            time.sleep(state["slow_ms"] / 1000.0)
        hits = shard.search(q, df, k=k)
        state["queries"] += 1
        return {"results": [{"id": i, "score": s} for i, s in hits],
                "shard": shard_id,
                "took_ms": (time.perf_counter() - t0) * 1000.0}

    @app.get("/health")
    def health():
        return {"shard": shard_id, "docs": len(shard.ext_ids),
                "vocab": len(shard.vocab), "queries": state["queries"],
                "slow_ms": state["slow_ms"]}

    @app.post("/inject/slow")
    def inject(ms: float):
        """Fault injection, on the shard itself rather than in the coordinator,
        so the coordinator's behaviour is observed and not simulated."""
        state["slow_ms"] = ms
        return {"shard": shard_id, "slow_ms": ms}

    return app
