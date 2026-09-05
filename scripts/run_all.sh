#!/usr/bin/env bash
# The whole study, on one GPU. Roughly 2 to 4 hours end to end.
set -euo pipefail

PYTHONPATH=src python scripts/prepare_sft.py --out data/sft.jsonl
PYTHONPATH=src python scripts/train_sft.py --data data/sft.jsonl --out runs/sft
PYTHONPATH=src python scripts/generate_rewrites.py --adapters runs/sft/step-* --out results/rewrites
PYTHONPATH=src python scripts/evaluate.py \
  $(for f in results/rewrites/*.json; do echo --rewrites "$f"; done) \
  --out results/evaluation.json
