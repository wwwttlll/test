from __future__ import annotations

import math
import random
from typing import List, Sequence

Vector = List[float]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def add(a: Sequence[float], b: Sequence[float], alpha: float = 1.0) -> Vector:
    return [x + alpha * y for x, y in zip(a, b)]


def unit(v: Sequence[float]) -> Vector:
    n = norm(v)
    if n == 0:
        return [0.0 for _ in v]
    return [x / n for x in v]


def gaussian_vector(dim: int, scale: float, seed: int) -> Vector:
    rng = random.Random(seed)
    return [rng.gauss(0.0, scale) for _ in range(dim)]
