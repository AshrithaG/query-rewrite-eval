"""A BM25 inverted index, built rather than imported.

Two reasons it is hand-written. The scoring has to accept externally supplied
document frequencies, because a sharded index that computes IDF from its own
slice scores documents differently depending on which shard they landed in, and
libraries do not expose that. And the memory layout matters at 8.8M passages:
postings are held as flat numpy arrays with a per-term offset table, not as a
dict of lists, which is the difference between a few hundred MB and running out
of RAM.

Layout, after build():
    vocab     term -> term_id
    offsets   term_id -> [start, end) into doc_ids / tfs
    doc_ids   concatenated postings, sorted by term then doc
    tfs       term frequency, parallel to doc_ids
    doc_len   tokens per document
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# Closed-class words carry no discrimination and dominate postings volume.
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it its no not of on or such that
the their then there these they this to was will with what which who when where
how do does did done have has had been being your you i we he she him her his
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) < 32]


@dataclass
class DocFreq:
    """Global document frequencies, gathered before any shard is scored.

    BM25's IDF term needs the frequency of a word in the whole collection. A
    shard only sees its own slice, so if each shard computes IDF locally the
    same document scores differently depending on which shard it landed in, and
    merged top-k lists are wrong in a way that is invisible unless you look.
    """
    df: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0

    def add_document(self, tokens: list[str]) -> None:
        self.n_docs += 1
        for t in set(tokens):
            self.df[t] = self.df.get(t, 0) + 1

    def merge(self, other: "DocFreq") -> "DocFreq":
        out = DocFreq(dict(self.df), self.n_docs + other.n_docs)
        for t, c in other.df.items():
            out.df[t] = out.df.get(t, 0) + c
        return out

    def idf(self, term: str) -> float:
        # Robertson/Sparck Jones IDF with the +0.5 smoothing, as in Lucene.
        n = self.df.get(term, 0)
        return float(np.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5)))


class Shard:
    """One partition of the collection."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b          # MS MARCO-tuned defaults, not Lucene's 1.2/0.75
        self.vocab: dict[str, int] = {}
        self.offsets: np.ndarray | None = None
        self.doc_ids: np.ndarray | None = None
        self.tfs: np.ndarray | None = None
        self.doc_len: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.ext_ids: list[str] = []      # local id -> collection id

    def build(self, docs: list[tuple[str, str]]) -> None:
        """docs is [(external_id, text)]. Built once, then read-only."""
        terms: list[int] = []
        locals_: list[int] = []
        counts: list[int] = []
        lengths = np.zeros(len(docs), dtype=np.int32)
        self.ext_ids = [d[0] for d in docs]

        for local_id, (_, text) in enumerate(docs):
            toks = tokenize(text)
            lengths[local_id] = len(toks)
            seen: dict[int, int] = {}
            for t in toks:
                tid = self.vocab.get(t)
                if tid is None:
                    tid = self.vocab[t] = len(self.vocab)
                seen[tid] = seen.get(tid, 0) + 1
            for tid, tf in seen.items():
                terms.append(tid); locals_.append(local_id); counts.append(tf)

        t_arr = np.asarray(terms, dtype=np.int32)
        d_arr = np.asarray(locals_, dtype=np.int32)
        f_arr = np.asarray(counts, dtype=np.int32)
        # Sort by term so each term's postings are one contiguous slice; that is
        # what makes lookup a slice rather than a gather.
        order = np.argsort(t_arr, kind="stable")
        t_arr, self.doc_ids, self.tfs = t_arr[order], d_arr[order], f_arr[order]

        self.offsets = np.zeros((len(self.vocab), 2), dtype=np.int64)
        if t_arr.size:
            bounds = np.searchsorted(t_arr, np.arange(len(self.vocab) + 1))
            self.offsets[:, 0] = bounds[:-1]
            self.offsets[:, 1] = bounds[1:]
        self.doc_len = lengths
        self.avgdl = float(lengths.mean()) if len(lengths) else 0.0

    def search(self, query: str, df: DocFreq, k: int = 10) -> list[tuple[str, float]]:
        """Top-k for one query, scored with the supplied global IDF."""
        if self.doc_ids is None or not len(self.ext_ids):
            return []
        scores = np.zeros(len(self.ext_ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))

        for term in set(tokenize(query)):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            s, e = self.offsets[tid]
            if s == e:
                continue
            docs, tf = self.doc_ids[s:e], self.tfs[s:e].astype(np.float32)
            # BM25 saturation: term frequency helps with diminishing returns
            np.add.at(scores, docs, df.idf(term) * (tf * (self.k1 + 1)) / (tf + norm[docs]))

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ext_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def memory_bytes(self) -> int:
        arrays = [self.offsets, self.doc_ids, self.tfs, self.doc_len]
        return sum(a.nbytes for a in arrays if a is not None) + sum(
            len(t) + 60 for t in self.vocab)
