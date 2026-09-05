#!/usr/bin/env python3
"""LoRA fine-tune a small model to rewrite conversational queries.

Checkpoints are saved at several points during training on purpose. Each one is
a system in the evaluation, which is what gives the study enough systems to say
something statistically, and it reproduces the decision a practitioner actually
faces: several checkpoints, pick one. If ROUGE and retrieval disagree about
which to pick, that is the finding.

    python scripts/train_sft.py --data data/sft.jsonl --out runs/qwen1.7b
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class SFTData(Dataset):
    """Prompt plus completion, with the prompt masked out of the loss.

    Training on the prompt tokens as well would spend capacity learning to
    reproduce the instruction, which is not the task.
    """

    def __init__(self, path: Path, tok, max_len: int = 768):
        self.rows = [json.loads(l) for l in path.open()]
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        p = self.tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = self.tok(r["completion"] + self.tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + c)[: self.max_len]
        labels = ([-100] * len(p) + c)[: self.max_len]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        k = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * k)
        out["labels"].append(b["labels"] + [-100] * k)
        out["attention_mask"].append([1] * len(b["input_ids"]) + [0] * k)
    return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", type=Path, default=Path("data/sft.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("runs/sft"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--checkpoints", type=int, default=6, help="how many to save during training")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    ds = SFTData(args.data, tok)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps = int(len(dl) * args.epochs / args.accum)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

    save_every = max(1, steps // args.checkpoints)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ds)} examples, {steps} optimizer steps, checkpoint every {save_every}")

    model.train()
    step = 0
    running = 0.0
    done = False
    for epoch in range(math.ceil(args.epochs)):
        if done:
            break
        for i, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += loss.item()
            if (i + 1) % args.accum:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1

            if step % 25 == 0:
                print(f"  step {step}/{steps}  loss {running / 25:.4f}  lr {sched.get_last_lr()[0]:.2e}",
                      flush=True)
                running = 0.0
            if step % save_every == 0 or step == steps:
                d = args.out / f"step-{step:05d}"
                model.save_pretrained(d); tok.save_pretrained(d)
                print(f"  saved {d}", flush=True)
            if step >= steps:
                done = True
                break
    print("done")


if __name__ == "__main__":
    main()
