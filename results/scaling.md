# When is sharding worth it? Measured, on a 12-core M4 Pro.

Single index, MS MARCO passages, own BM25 implementation.

| documents | build | index size | search p50 |
|---|---|---|---|
| 50,000 | 1.2 s | 18 MB | 0.17 ms |
| 100,000 | 2.5 s | 32 MB | 0.52 ms |
| 200,000 | 5.4 s | 60 MB | 0.72 ms |
| 400,000 | 10.8 s | 114 MB | 1.48 ms |
| 12,353 (QReCC answers) | 0.1 s | 3 MB | 0.07 ms |

Search cost grows about linearly with collection size. A loopback HTTP round
trip costs roughly 1 ms, so a shard doing less than about 1 ms of work spends
more on being called than on searching.

At the QReCC scale used for the retrieval evaluation, splitting the collection
eight ways saves 0.04 ms of search and adds about 1 ms of coordination: a 25x
pessimization. Fan-out to eight local processes made p50 at concurrency 32 go
from 21.6 ms to 776 ms, because eight uvicorn workers and a coordinator were
competing for the same twelve cores to avoid 0.07 ms of work.

The crossover is near 400k documents per shard. Extrapolating the trend to the
full 8.8M-passage collection gives roughly 33 ms for a single index, where
splitting eight ways should cost about 4 ms of search plus 1 ms of
coordination.

The useful form of this is a rule rather than a number: shard when per-shard
work exceeds your coordination cost, and measure both before assuming.
