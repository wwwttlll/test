from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from dataclasses import dataclass
from typing import Dict, List

from .budget import BudgetTracker
from .config import ExperimentConfig
from .metrics import bootstrap_mean_ci, oracle_recall_at_k, recall_at_k
from .retrieval import Candidate, CandidateIndex
from .sampling import sample_hypotheses
from .scoring import aggregate_max_over_hypotheses, structured_score
from .vector_ops import Vector, add


@dataclass(frozen=True)
class Query:
    query_id: str
    reference_embedding: Vector
    modification_text: str
    target_id: str


@dataclass
class KResult:
    k: int
    r1: float
    r5: float
    r10: float
    oracle_r100: float
    r1_ci_low: float = 0.0
    r1_ci_high: float = 0.0
    r5_ci_low: float = 0.0
    r5_ci_high: float = 0.0
    r10_ci_low: float = 0.0
    r10_ci_high: float = 0.0
    forward_passes: int = 0
    wall_clock_s: float = 0.0


def _rank_for_query(query: Query, index: CandidateIndex, cfg: ExperimentConfig, k: int, budget: BudgetTracker) -> List[str]:
    hypotheses = sample_hypotheses(
        mod_text=query.modification_text,
        ref_embedding=query.reference_embedding,
        k=k,
        cfg=cfg.sampling,
    )

    per_candidate_scores: Dict[str, Dict[str, object]] = defaultdict(dict)

    for h_idx, hyp in enumerate(hypotheses):
        query_vec = add(query.reference_embedding, hyp.direction, 1.0)
        retrieved = index.top_n(query_vec, cfg.top_n)
        budget.add_forward(len(retrieved) + 1)
        for candidate, _ in retrieved:
            per_candidate_scores[candidate.image_id][f"h_{h_idx}"] = structured_score(
                ref_embedding=query.reference_embedding,
                candidate=candidate,
                hypothesis=hyp,
                cfg=cfg.scoring,
            )

    ranked = sorted(
        per_candidate_scores.items(),
        key=lambda kv: aggregate_max_over_hypotheses(kv[1]),
        reverse=True,
    )
    return [image_id for image_id, _ in ranked]


def run_experiment(queries: List[Query], candidates: List[Candidate], cfg: ExperimentConfig) -> List[KResult]:
    if len(cfg.seeds) <= 1:
        return _run_single_seed(queries, candidates, cfg)
    by_seed: List[List[KResult]] = []
    for seed in cfg.seeds:
        seeded_cfg = replace(cfg, sampling=replace(cfg.sampling, seed=seed), seeds=[seed])
        by_seed.append(_run_single_seed(queries, candidates, seeded_cfg))

    merged: List[KResult] = []
    for idx, k in enumerate(cfg.k_values):
        r1_vals = [seed_res[idx].r1 for seed_res in by_seed]
        r5_vals = [seed_res[idx].r5 for seed_res in by_seed]
        r10_vals = [seed_res[idx].r10 for seed_res in by_seed]
        oracle_vals = [seed_res[idx].oracle_r100 for seed_res in by_seed]
        fp_vals = [seed_res[idx].forward_passes for seed_res in by_seed]
        wall_vals = [seed_res[idx].wall_clock_s for seed_res in by_seed]

        r1_mean, r1_low, r1_high = bootstrap_mean_ci(r1_vals, n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed)
        r5_mean, r5_low, r5_high = bootstrap_mean_ci(r5_vals, n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed + 1)
        r10_mean, r10_low, r10_high = bootstrap_mean_ci(r10_vals, n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed + 2)
        oracle_mean, _, _ = bootstrap_mean_ci(oracle_vals, n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed + 3)
        forward_mean, _, _ = bootstrap_mean_ci([float(v) for v in fp_vals], n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed + 4)
        wall_mean, _, _ = bootstrap_mean_ci(wall_vals, n_bootstrap=cfg.bootstrap_samples, seed=cfg.sampling.seed + 5)

        merged.append(
            KResult(
                k=k,
                r1=r1_mean,
                r5=r5_mean,
                r10=r10_mean,
                oracle_r100=oracle_mean,
                r1_ci_low=r1_low,
                r1_ci_high=r1_high,
                r5_ci_low=r5_low,
                r5_ci_high=r5_high,
                r10_ci_low=r10_low,
                r10_ci_high=r10_high,
                forward_passes=int(round(forward_mean)),
                wall_clock_s=wall_mean,
            )
        )
    return merged


def _run_single_seed(queries: List[Query], candidates: List[Candidate], cfg: ExperimentConfig) -> List[KResult]:
    index = CandidateIndex(candidates)
    all_results: List[KResult] = []

    for k in cfg.k_values:
        budget = BudgetTracker()
        budget.start()
        r1 = r5 = r10 = oracle100 = 0.0

        for query in queries:
            ranked_by_h = []
            hypotheses = sample_hypotheses(query.modification_text, query.reference_embedding, k, cfg.sampling)
            for hyp in hypotheses:
                query_vec = add(query.reference_embedding, hyp.direction, 1.0)
                retrieved = index.top_n(query_vec, 100)
                budget.add_forward(len(retrieved) + 1)
                ranked_by_h.append([c.image_id for c, _ in retrieved])

            final_rank = _rank_for_query(query, index, cfg, k, budget)
            r1 += recall_at_k(final_rank, query.target_id, 1)
            r5 += recall_at_k(final_rank, query.target_id, 5)
            r10 += recall_at_k(final_rank, query.target_id, 10)
            oracle100 += oracle_recall_at_k(ranked_by_h, query.target_id, 100)

        budget.stop()
        qn = max(len(queries), 1)
        all_results.append(
            KResult(
                k=k,
                r1=r1 / qn,
                r5=r5 / qn,
                r10=r10 / qn,
                oracle_r100=oracle100 / qn,
                forward_passes=budget.forward_passes,
                wall_clock_s=budget.wall_clock_s,
            )
        )

    return all_results
