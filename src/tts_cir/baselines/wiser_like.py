from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..budget import BudgetTracker
from ..config import ExperimentConfig
from ..metrics import recall_at_k
from ..retrieval import Candidate, CandidateIndex
from ..sampling import sample_hypotheses
from ..vector_ops import Vector, add


@dataclass
class RefinementResult:
    rounds: int
    r1: float
    r5: float
    r10: float
    forward_passes: int
    wall_clock_s: float


@dataclass(frozen=True)
class BaselineQuery:
    query_id: str
    reference_embedding: Vector
    modification_text: str
    target_id: str


def _serial_refine_rank(query: BaselineQuery, index: CandidateIndex, cfg: ExperimentConfig, rounds: int, budget: BudgetTracker) -> List[str]:
    """
    WISER-like lightweight serial baseline:
    - one hypothesis per round
    - retrieve top-N
    - use top-1 candidate as pseudo-feedback to update query vector
    """
    current = query.reference_embedding
    ranking: List[str] = []

    for step in range(rounds):
        hypothesis = sample_hypotheses(query.modification_text, current, 1, cfg.sampling)[0]
        query_vec = add(current, hypothesis.direction, 1.0)
        retrieved = index.top_n(query_vec, cfg.top_n)
        budget.add_forward(len(retrieved) + 1)

        if not retrieved:
            continue

        ranking = [cand.image_id for cand, _ in retrieved]
        best_candidate = retrieved[0][0]

        # Refinement with conservative momentum to avoid drift.
        current = add(current, best_candidate.embedding, 0.2)

    return ranking


def run_wiser_like_experiment(queries: List[BaselineQuery], candidates: List[Candidate], cfg: ExperimentConfig, rounds_list: List[int]) -> List[RefinementResult]:
    index = CandidateIndex(candidates)
    outputs: List[RefinementResult] = []

    for rounds in rounds_list:
        budget = BudgetTracker()
        budget.start()
        r1 = r5 = r10 = 0.0

        for query in queries:
            ranked = _serial_refine_rank(query, index, cfg, rounds, budget)
            r1 += recall_at_k(ranked, query.target_id, 1)
            r5 += recall_at_k(ranked, query.target_id, 5)
            r10 += recall_at_k(ranked, query.target_id, 10)

        budget.stop()
        qn = max(len(queries), 1)
        outputs.append(
            RefinementResult(
                rounds=rounds,
                r1=r1 / qn,
                r5=r5 / qn,
                r10=r10 / qn,
                forward_passes=budget.forward_passes,
                wall_clock_s=budget.wall_clock_s,
            )
        )

    return outputs
