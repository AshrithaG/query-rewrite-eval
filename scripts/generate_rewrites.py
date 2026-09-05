#!/usr/bin/env python3
"""Generate rewrites for the eval turns, for the base model and every checkpoint.

vLLM with LoRA adapters, so all checkpoints are served from one loaded copy of
the base weights rather than reloading a model per system.

    python scripts/generate_rewrites.py --adapters runs/sft/step-* --out results/rewrites
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prepare_sft import build_prompt  # noqa: E402
from qrw.data import load_turns  # noqa: E402


def clean(text: str) -> str:
    """Models like to add commentary. Keep the first non-empty line and strip
    any label it decided to prefix."""
    line = next((l.strip() for l in text.strip().splitlines() if l.strip()), "")
    for prefix in ("Rewritten question:", "Rewrite:", "Question:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix):].strip()
    return line.strip('"').strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapters", nargs="*", default=[], help="LoRA checkpoint dirs")
    ap.add_argument("--out", type=Path, default=Path("results/rewrites"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    turns = load_turns("test", needs_context=True)
    if args.limit:
        turns = turns[: args.limit]
    prompts = [build_prompt(t) for t in turns]
    print(f"{len(prompts)} prompts")

    llm = LLM(model=args.model, enable_lora=bool(args.adapters),
              max_lora_rank=args.max_lora_rank, max_model_len=1536,
              gpu_memory_utilization=0.85, enforce_eager=True)
    # greedy, so a checkpoint's output is a property of the checkpoint and not
    # of the sampling seed
    params = SamplingParams(temperature=0.0, max_tokens=64)

    args.out.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, LoRARequest | None]] = [("base", None)]
    for i, a in enumerate(args.adapters, start=1):
        jobs.append((f"sft_{Path(a).name}", LoRARequest(Path(a).name, i, a)))

    for name, lora in jobs:
        outs = llm.generate(prompts, params, lora_request=lora)
        mapping = {t.uid: clean(o.outputs[0].text) for t, o in zip(turns, outs)}
        empty = sum(1 for v in mapping.values() if not v)
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(mapping, indent=2))
        print(f"  {name}: wrote {len(mapping)} rewrites ({empty} empty) -> {path}", flush=True)


if __name__ == "__main__":
    main()
