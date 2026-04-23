#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from tts_cir.config import ExperimentConfig, SamplingConfig, ScoringConfig
from tts_cir.retrieval import Candidate
from tts_cir.runner import Query, run_experiment


def _load_queries(path: Path) -> List[Query]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                Query(
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
    parser = argparse.ArgumentParser(description="Run TTS-CIR MVP scaling experiment.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results_mvp.csv"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--sampling-mode", type=str, choices=["textual", "latent"], default="textual")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        k_values=args.k_values,
        sampling=SamplingConfig(mode=args.sampling_mode),
        scoring=ScoringConfig(),
    )

    results = run_experiment(_load_queries(args.queries), _load_candidates(args.candidates), cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "r1", "r5", "r10", "oracle_r100", "forward_passes", "wall_clock_s"])
        for r in results:
            writer.writerow([r.k, f"{r.r1:.4f}", f"{r.r5:.4f}", f"{r.r10:.4f}", f"{r.oracle_r100:.4f}", r.forward_passes, f"{r.wall_clock_s:.4f}"])

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
