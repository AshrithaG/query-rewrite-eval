"""Sharded retrieval: fan out to every shard, merge the local top-k.

The subtle part is IDF. BM25 weights a term by how rare it is in the collection,
but a shard only holds a slice, so a shard computing IDF from its own documents
weights terms by how rare they are *there*. The same document then scores
differently depending on which shard it was assigned to, and the merged list is
wrong. Nothing about it looks wrong: no error is raised, the results are
plausible, they are just not the results the unsharded index would return.

Both are implemented here so the difference can be measured instead of asserted.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from qrw.invindex import DocFreq, Shard, tokenize


@dataclass
class ShardStats:
    n_docs: int
    vocab: int
    memory_mb: float


class ShardedIndex:
    def __init__(self, n_shards: int, k1: float = 0.9, b: float = 0.4):
        self.n_shards = n_shards
        self.shards = [Shard(k1=k1, b=b) for _ in range(n_shards)]
        self.global_df = DocFreq()
        self.local_df: list[DocFreq] = [DocFreq() for _ in range(n_shards)]

    @staticmethod
    def assign(doc_id: str, n_shards: int) -> int:
        """Hash partitioning. Deterministic, no coordinator state, and it spreads
        a topic across shards so no single shard owns a subject area."""
        return hash(doc_id) % n_shards

    def build(self, docs: list[tuple[str, str]]) -> list[ShardStats]:
        buckets: list[list[tuple[str, str]]] = [[] for _ in range(self.n_shards)]
        for ext_id, text in docs:
            buckets[self.assign(ext_id, self.n_shards)].append((ext_id, text))

        for i, bucket in enumerate(buckets):
            for _, text in bucket:
                toks = tokenize(text)
                self.global_df.add_document(toks)
                self.local_df[i].add_document(toks)
            self.shards[i].build(bucket)

        return [ShardStats(len(b), len(s.vocab), s.memory_bytes() / 1e6)
                for b, s in zip(buckets, self.shards)]

    def search(self, query: str, k: int = 10, global_idf: bool = True) -> list[tuple[str, float]]:
        merged: list[tuple[str, float]] = []
        for i, shard in enumerate(self.shards):
            df = self.global_df if global_idf else self.local_df[i]
            merged.extend(shard.search(query, df, k=k))
        return heapq.nlargest(k, merged, key=lambda x: x[1])
