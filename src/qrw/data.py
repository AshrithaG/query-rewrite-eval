"""QReCC loading, and the retrieval task built on top of it.

QReCC gives conversational turns: the conversation so far, an ambiguous
question, a human self-contained rewrite, and an answer. Its `Passages` field
holds URLs rather than text, and the official collection is 54M passages, so
this builds a self-contained retrieval task instead: the corpus is every unique
answer in the split, a query's relevant document is its own answer, and every
other answer is a distractor.

That is a smaller and easier task than the official QReCC benchmark, and the
numbers here should not be compared against published QReCC retrieval results.
What it supports is the comparison this project cares about: the same queries
scored by ROUGE against the gold rewrite and by retrieval against a fixed
corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from huggingface_hub import hf_hub_download

REPO = "svakulenk0/qrecc"


@dataclass(frozen=True)
class Turn:
    conv_id: int
    turn_no: int
    context: tuple[str, ...]   # alternating question/answer, oldest first
    question: str              # as asked, often not self-contained
    gold_rewrite: str          # human self-contained version
    answer: str                # the retrieval target
    source: str                # quac | nq | trec

    @property
    def uid(self) -> str:
        return f"{self.source}-{self.conv_id}-{self.turn_no}"


def _load_raw(split: str) -> list[dict]:
    fname = "qrecc-training.json" if split == "train" else "qrecc-test.json"
    return json.load(open(hf_hub_download(REPO, fname, repo_type="dataset")))


@lru_cache(maxsize=4)
def load_turns(split: str, needs_context: bool = True) -> tuple[Turn, ...]:
    """Turns usable for this study.

    `needs_context=True` keeps only turns where the conversation has history,
    which are the turns where rewriting can do anything at all. A first turn is
    already self-contained, so including them would dilute every measurement
    with cases that have no headroom.
    """
    out = []
    for r in _load_raw(split):
        q = (r.get("Question") or "").strip()
        rw = (r.get("Rewrite") or "").strip()
        a = (r.get("Answer") or "").strip()
        ctx = tuple(c.strip() for c in (r.get("Context") or []) if c and c.strip())
        if not (q and rw and a):
            continue
        if needs_context and not ctx:
            continue
        out.append(Turn(
            conv_id=int(r["Conversation_no"]), turn_no=int(r["Turn_no"]),
            context=ctx, question=q, gold_rewrite=rw, answer=a,
            source=r.get("Conversation_source", "?"),
        ))
    return tuple(out)


def build_corpus(turns: tuple[Turn, ...]) -> tuple[list[str], dict[str, int]]:
    """The document collection, plus each turn's relevant document id.

    Answers are deduplicated, so two turns that share an answer share a
    relevant document. That is correct: if the same text answers both, both
    should retrieve it.
    """
    doc_id: dict[str, int] = {}
    docs: list[str] = []
    for t in turns:
        if t.answer not in doc_id:
            doc_id[t.answer] = len(docs)
            docs.append(t.answer)
    relevant = {t.uid: doc_id[t.answer] for t in turns}
    return docs, relevant
