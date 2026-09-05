"""BM25 retrieval and the metrics that go with it.

BM25 rather than a dense retriever, deliberately. A dense model would fold its
own semantics into the result, so a rewrite could score well because the encoder
happens to like it. BM25 is a transparent lexical function: if a rewrite
retrieves better it is because it put better terms in the query, which is what
query rewriting is supposed to do.
"""

from __future__ import annotations

import numpy as np

try:
    import bm25s
    import Stemmer
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install bm25s PyStemmer") from e


class Index:
    def __init__(self, docs: list[str]):
        self._stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self._stemmer, show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens, show_progress=False)
        self.n_docs = len(docs)

    def search(self, queries: list[str], k: int = 10) -> np.ndarray:
        """Doc ids for each query, best first. Shape (len(queries), k)."""
        k = min(k, self.n_docs)
        tokens = bm25s.tokenize(queries, stopwords="en", stemmer=self._stemmer, show_progress=False)
        ids, _ = self._bm25.retrieve(tokens, k=k, show_progress=False)
        return np.asarray(ids)


def rank_of(hits: np.ndarray, relevant: int) -> int | None:
    """1-based rank of the relevant doc, or None if it is not in the list."""
    where = np.flatnonzero(hits == relevant)
    return int(where[0]) + 1 if where.size else None


def metrics(ranks: list[int | None], k: int = 10) -> dict[str, float]:
    """Recall@k, MRR@k and nDCG@k for a single relevant document per query.

    With exactly one relevant document, nDCG@k reduces to 1/log2(rank+1), and
    MRR to 1/rank. Both are reported because they weight the head of the ranking
    differently, and a rewrite that moves a document from rank 8 to rank 2 should
    be visible somewhere.
    """
    n = len(ranks)
    if n == 0:
        return {f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"ndcg@{k}": 0.0}
    hit = [r is not None and r <= k for r in ranks]
    mrr = [1.0 / r if (r is not None and r <= k) else 0.0 for r in ranks]
    ndcg = [1.0 / np.log2(r + 1) if (r is not None and r <= k) else 0.0 for r in ranks]
    return {
        f"recall@{k}": float(np.mean(hit)),
        f"mrr@{k}": float(np.mean(mrr)),
        f"ndcg@{k}": float(np.mean(ndcg)),
    }
