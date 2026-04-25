"""Pairwise verifier for SPVI score fusion."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from .llm_client import pairwise_verdict


@dataclass
class VerifierConfig:
    beta: float = 0.5  # Weight for preference score
    top_m: int = 20    # Only verify top-M candidates
    use_cache: bool = True


_verdict_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}


def clear_verifier_cache() -> None:
    """Clear the verifier cache."""
    _verdict_cache.clear()


def get_preference_score(
    ref_caption: str,
    mod_text: str,
    candidate_ids: List[str],
    cand_embeddings: Dict[str, Any],
    cfg: VerifierConfig,
) -> Dict[str, float]:
    """Compute pairwise preference scores for candidates."""
    n = len(candidate_ids)
    if n == 0:
        return {}
    
    # Build pairwise results
    win_count: Dict[str, int] = defaultdict(int)
    conflict_count = 0
    total_pairs = 0
    
    # Only verify up to top_m candidates
    check_ids = candidate_ids[:min(cfg.top_m, n)]
    
    for i in range(len(check_ids)):
        for j in range(i + 1, len(check_ids)):
            cand_a = check_ids[i]
            cand_b = check_ids[j]
            cache_key = (ref_caption, mod_text, cand_a, cand_b)
            
            if cfg.use_cache and cache_key in _verdict_cache:
                verdict = _verdict_cache[cache_key]
            else:
                desc_a = _get_cand_desc(cand_embeddings.get(cand_a, {}))
                desc_b = _get_cand_desc(cand_embeddings.get(cand_b, {}))
                verdict = pairwise_verdict(ref_caption, mod_text, cand_a, cand_b, desc_a, desc_b)
                if cfg.use_cache:
                    _verdict_cache[cache_key] = verdict
                    # Add reverse cache too
                    _verdict_cache[(ref_caption, mod_text, cand_b, cand_a)] = {
                        "winner": "B" if verdict["winner"] == "A" else "A" if verdict["winner"] == "B" else "TIE",
                        "reason": verdict["reason"],
                        "confidence": verdict["confidence"],
                    }
            
            total_pairs += 1
            if verdict["winner"] == "A":
                win_count[cand_a] += 1
            elif verdict["winner"] == "B":
                win_count[cand_b] += 1
            else:
                win_count[cand_a] += 1
                win_count[cand_b] += 1
            
            # Check for conflicts (A>B vs B>A)
            rev_cache = (ref_caption, mod_text, cand_b, cand_a)
            if rev_cache in _verdict_cache:
                rev_verdict = _verdict_cache[rev_cache]
                if verdict["winner"] != "TIE" and rev_verdict["winner"] != "TIE":
                    if (verdict["winner"] == "A" and rev_verdict["winner"] == "A") or                        (verdict["winner"] == "B" and rev_verdict["winner"] == "B"):
                        conflict_count += 1
    
    # Convert to Borda-like scores (normalized)
    max_possible_wins = len(check_ids) - 1
    pref_scores: Dict[str, float] = {}
    
    # Initialize all candidates
    for cid in candidate_ids:
        pref_scores[cid] = 0.0
    
    # Add scores for checked candidates
    for cid in check_ids:
        if max_possible_wins > 0:
            pref_scores[cid] = win_count[cid] / max_possible_wins
    
    return pref_scores


def _get_cand_desc(cand_emb: Dict[str, Any]) -> str:
    """Get candidate description from embedding."""
    # This would ideally come from image features, but we use ID for now
    return cand_emb.get("description", "")  # type: ignore


def fuse_scores(
    retrieval_scores: Dict[str, float],
    preference_scores: Dict[str, float],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Fuse retrieval and preference scores."""
    fused: Dict[str, float] = {}
    
    # Get max retrieval score for normalization
    max_ret = max(retrieval_scores.values()) if retrieval_scores else 1.0
    max_pref = max(preference_scores.values()) if preference_scores else 1.0
    
    for cid in set(list(retrieval_scores.keys()) + list(preference_scores.keys())):
        ret_score = retrieval_scores.get(cid, 0.0) / max_ret
        pref_score = preference_scores.get(cid, 0.0) / max_pref
        fused[cid] = ret_score + beta * pref_score
    
    return fused
