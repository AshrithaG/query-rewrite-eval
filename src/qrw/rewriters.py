"""The rewriters under comparison.

Every one of these turns (context, question) into a query string. Keeping them
behind one interface means the evaluation never knows which it is scoring.
"""

from __future__ import annotations

from collections.abc import Callable

from qrw.data import Turn

Rewriter = Callable[[Turn], str]


def raw(t: Turn) -> str:
    """No rewriting. The floor: what a system does if it ignores the conversation."""
    return t.question


def gold(t: Turn) -> str:
    """The human reference. The ceiling any learned rewriter is chasing."""
    return t.gold_rewrite


def concat_last(n_turns: int = 2) -> Rewriter:
    """Prepend the last n context utterances. Crude, free, and a strong baseline
    that papers often skip past on the way to a model."""
    def f(t: Turn) -> str:
        return " ".join(list(t.context[-n_turns:]) + [t.question])
    return f


def concat_all(t: Turn) -> str:
    """The whole conversation. Maximum recall of useful terms, and maximum noise."""
    return " ".join(list(t.context) + [t.question])


def first_turn_plus(t: Turn) -> str:
    """The opening question carries the topic; later turns carry the specifics.
    A cheap approximation of coreference resolution."""
    head = t.context[0] if t.context else ""
    return f"{head} {t.question}".strip()


BASELINES: dict[str, Rewriter] = {
    "raw": raw,
    "concat_1": concat_last(1),
    "concat_2": concat_last(2),
    "concat_4": concat_last(4),
    "concat_all": concat_all,
    "first_turn": first_turn_plus,
    "gold": gold,
}
