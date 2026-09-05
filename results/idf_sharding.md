# Sharded BM25 silently changes the answer if IDF is computed per shard

BM25 weights a term by its rarity in the collection. A shard holds a slice, so
a shard computing IDF from its own documents weights terms by their rarity
*there*. The same document then scores differently depending on which shard it
landed on. Nothing errors; the results look fine; they are not the results the
unsharded index returns.

Agreement with a single unsharded index over the same collection, top-10
overlap, 800 queries, 7,850 documents:

| shards | global IDF | per-shard IDF | top-1 preserved (per-shard IDF) |
|---|---|---|---|
| 2 | 97.1% | 90.6% | 93.6% |
| 4 | 97.1% | 85.9% | 88.8% |
| 8 | 96.9% | 83.3% | 86.9% |
| 16 | 96.8% | 79.6% | 83.9% |

With global document frequencies distributed to every shard, sharding is
effectively lossless and stays flat as shard count grows; the residual 3% is
tie-breaking, and matches the disagreement between this implementation and
bm25s under identical parameters.

With per-shard IDF, a fifth of the top-10 is wrong at 16 shards and one query
in six gets a different top result. The cost of correctness is one pass over
the collection to collect document frequencies before any shard is built.
