from typing import Iterable, Sequence


def recall_at_k(ranked_ids: Sequence[str], target_id: str, k: int) -> float:
    return 1.0 if target_id in ranked_ids[:k] else 0.0


def oracle_recall_at_k(hypothesis_ranked_ids: Iterable[Sequence[str]], target_id: str, k: int) -> float:
    for ranked in hypothesis_ranked_ids:
        if target_id in ranked[:k]:
            return 1.0
    return 0.0
