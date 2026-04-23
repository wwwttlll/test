from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .budget import BudgetTracker
from .config import ExperimentConfig
from .metrics import oracle_recall_at_k, recall_at_k
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
    forward_passes: int
    wall_clock_s: float


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
