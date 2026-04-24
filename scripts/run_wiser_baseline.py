#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from tts_cir.baselines.wiser_like import BaselineQuery, run_wiser_like_experiment
from tts_cir.config import ExperimentConfig, SamplingConfig, ScoringConfig
from tts_cir.retrieval import Candidate


def _load_queries(path: Path) -> List[BaselineQuery]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                BaselineQuery(
                    query_id=obj["query_id"],
                    reference_embedding=[float(x) for x in obj["reference_embedding"]],
                    modification_text=obj["modification_text"],
                    target_id=obj["target_id"],
                )
            )
    return rows


def _load_candidates(path: Path) -> List[Candidate]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(Candidate(image_id=obj["image_id"], embedding=[float(x) for x in obj["embedding"]]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WISER-like serial refinement baseline.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/wiser_baseline.csv"))
    parser.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sampling-mode", type=str, choices=["textual", "latent"], default="textual")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        sampling=SamplingConfig(mode=args.sampling_mode),
        scoring=ScoringConfig(),
    )

    results = run_wiser_like_experiment(
        queries=_load_queries(args.queries),
        candidates=_load_candidates(args.candidates),
        cfg=cfg,
        rounds_list=args.rounds,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rounds", "r1", "r5", "r10", "forward_passes", "wall_clock_s"])
        for r in results:
            writer.writerow([r.rounds, f"{r.r1:.4f}", f"{r.r5:.4f}", f"{r.r10:.4f}", r.forward_passes, f"{r.wall_clock_s:.4f}"])

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
