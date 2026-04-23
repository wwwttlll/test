from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import ScoringConfig
from .retrieval import Candidate, cosine_similarity
from .sampling import EditHypothesis
from .vector_ops import Vector, add


@dataclass(frozen=True)
class ScoreBreakdown:
    s_obj: float
    s_attr: float
    s_style: float
    s_neg: float
    total: float


def structured_score(
    ref_embedding: Vector,
    candidate: Candidate,
    hypothesis: EditHypothesis,
    cfg: ScoringConfig,
) -> ScoreBreakdown:
    base = candidate.embedding
    s_obj = cosine_similarity(ref_embedding, base) * hypothesis.weights["obj"]
    s_attr = cosine_similarity(add(ref_embedding, hypothesis.direction, 0.5), base) * hypothesis.weights["attr"]
    s_style = cosine_similarity(add(ref_embedding, hypothesis.direction, 1.0), base) * hypothesis.weights["style"]

    neg_anchor = add(ref_embedding, hypothesis.direction, -1.0)
    s_neg = max(0.0, cosine_similarity(neg_anchor, base)) * hypothesis.weights["neg"]

    total = cfg.w_obj * s_obj + cfg.w_attr * s_attr + cfg.w_style * s_style - cfg.lambda_neg * s_neg
    return ScoreBreakdown(s_obj=s_obj, s_attr=s_attr, s_style=s_style, s_neg=s_neg, total=total)


def aggregate_max_over_hypotheses(score_by_hyp: Dict[str, ScoreBreakdown]) -> float:
    if not score_by_hyp:
        return float("-inf")
    return max(v.total for v in score_by_hyp.values())
