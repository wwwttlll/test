import random
from typing import Iterable, List, Sequence, Tuple


def recall_at_k(ranked_ids: Sequence[str], target_id: str, k: int) -> float:
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def oracle_recall_at_k(hypothesis_ranked_ids: Iterable[Sequence[str]], target_id: str, k: int) -> float:
    for ranked in hypothesis_ranked_ids:
        if target_id in ranked[:k]:
            return 1.0
    return 0.0


def bootstrap_mean_ci(values: Sequence[float], n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(values) / n
    low_idx = int((alpha / 2) * (len(means) - 1))
    high_idx = int((1 - alpha / 2) * (len(means) - 1))
    return mean, means[low_idx], means[high_idx]
