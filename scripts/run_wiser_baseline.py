#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from tts_cir.baselines.wiser_like import BaselineQuery, run_wiser_like_experiment
from tts_cir.config import ExperimentConfig, SamplingConfig, ScoringConfig
from tts_cir.metrics import bootstrap_mean_ci
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
    parser.add_argument("--top-n", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    queries = _load_queries(args.queries)
    candidates = _load_candidates(args.candidates)
    all_seed_results = []
    for seed in args.seeds:
        cfg = ExperimentConfig(
            top_n=args.top_n,
            seeds=[seed],
            bootstrap_samples=args.bootstrap_samples,
            sampling=SamplingConfig(mode=args.sampling_mode, seed=seed),
            scoring=ScoringConfig(),
        )
        all_seed_results.append(run_wiser_like_experiment(queries=queries, candidates=candidates, cfg=cfg, rounds_list=args.rounds))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rounds", "r1", "r1_ci_low", "r1_ci_high", "r5", "r10", "forward_passes", "wall_clock_s"])
        for idx, rounds in enumerate(args.rounds):
            r1_vals = [seed_res[idx].r1 for seed_res in all_seed_results]
            r5_vals = [seed_res[idx].r5 for seed_res in all_seed_results]
            r10_vals = [seed_res[idx].r10 for seed_res in all_seed_results]
            fp_vals = [seed_res[idx].forward_passes for seed_res in all_seed_results]
            wall_vals = [seed_res[idx].wall_clock_s for seed_res in all_seed_results]
            r1_mean, r1_low, r1_high = bootstrap_mean_ci(r1_vals, n_bootstrap=args.bootstrap_samples, seed=args.seeds[0])
            r5_mean, _, _ = bootstrap_mean_ci(r5_vals, n_bootstrap=args.bootstrap_samples, seed=args.seeds[0] + 1)
            r10_mean, _, _ = bootstrap_mean_ci(r10_vals, n_bootstrap=args.bootstrap_samples, seed=args.seeds[0] + 2)
            fp_mean, _, _ = bootstrap_mean_ci([float(x) for x in fp_vals], n_bootstrap=args.bootstrap_samples, seed=args.seeds[0] + 3)
            wall_mean, _, _ = bootstrap_mean_ci(wall_vals, n_bootstrap=args.bootstrap_samples, seed=args.seeds[0] + 4)
            writer.writerow([rounds, f"{r1_mean:.4f}", f"{r1_low:.4f}", f"{r1_high:.4f}", f"{r5_mean:.4f}", f"{r10_mean:.4f}", int(round(fp_mean)), f"{wall_mean:.4f}"])

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
