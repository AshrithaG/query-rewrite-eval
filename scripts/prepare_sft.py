#!/usr/bin/env python3
"""Build the supervised fine-tuning set from QReCC train.

One decision worth stating: the target is the human rewrite, which is exactly
what a query-rewriting paper trains on and exactly what ROUGE then measures
against. That is deliberate. The point of this study is to take the standard
recipe seriously and then ask whether its metric means anything.

    PYTHONPATH=src python scripts/prepare_sft.py --out data/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qrw.data import load_turns

PROMPT = (
    "Rewrite the final question so it can be understood on its own, without the "
    "conversation. Keep it short and keep the user's intent. Reply with the "
    "rewritten question only.\n\n"
    "Conversation:\n{context}\n\nFinal question: {question}\n\nRewritten question:"
)


def format_context(context: tuple[str, ...], max_turns: int = 6) -> str:
    """QReCC context alternates question, answer, question, answer, starting
    with a question at index 0. Roles must be derived from the absolute index,
    not the index within a truncated window, or truncating an odd number of
    turns silently swaps every label."""
    start = max(0, len(context) - max_turns)
    lines = []
    for i, utt in enumerate(context[start:], start=start):
        speaker = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{speaker}: {utt}")
    return "\n".join(lines) if lines else "(none)"


def build_prompt(turn) -> str:
    return PROMPT.format(context=format_context(turn.context), question=turn.question)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/sft.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    turns = load_turns("train", needs_context=True)
    if args.limit:
        turns = turns[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for t in turns:
            f.write(json.dumps({
                "uid": t.uid,
                "prompt": build_prompt(t),
                "completion": " " + t.gold_rewrite,
            }) + "\n")
    print(f"{len(turns)} training examples -> {args.out}")


if __name__ == "__main__":
    main()
