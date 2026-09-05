#!/usr/bin/env python3
"""Run one shard as its own process.

    PYTHONPATH=src python scripts/serve_shard.py --shard 0 --of 4 --port 9000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn

from qrw.service import make_app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("data/shards"))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--slow-ms", type=float, default=0.0)
    args = ap.parse_args()

    blob = pickle.loads((args.index_dir / f"n{args.of}" / f"shard-{args.shard}.pkl").read_bytes())
    app = make_app(blob["shard"], blob["global_df"], args.shard, slow_ms=args.slow_ms)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
