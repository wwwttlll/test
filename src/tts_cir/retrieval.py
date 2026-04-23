from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .vector_ops import Vector, dot, norm


def cosine_similarity(a: Vector, b: Vector) -> float:
    denom = (norm(a) * norm(b)) + 1e-8
    return float(dot(a, b) / denom)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    embedding: Vector


class CandidateIndex:
    def __init__(self, candidates: List[Candidate]) -> None:
        self.candidates = candidates

    def top_n(self, query_vec: Vector, n: int) -> List[Tuple[Candidate, float]]:
        scored = [(cand, cosine_similarity(query_vec, cand.embedding)) for cand in self.candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
