from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import SamplingConfig
from .vector_ops import Vector, gaussian_vector, unit


@dataclass(frozen=True)
class EditHypothesis:
    text: str
    weights: Dict[str, float]
    direction: Vector


def _extract_semantic_weights(text: str) -> Dict[str, float]:
    low = text.lower()
    return {
        "obj": 1.0,
        "attr": 1.0 if any(t in low for t in ["color", "material", "sleeve", "long", "short"]) else 0.8,
        "style": 1.0 if any(t in low for t in ["elegant", "vintage", "minimal", "casual"]) else 0.6,
        "neg": 1.0 if any(t in low for t in ["without", "remove", "not", "no "]) else 0.5,
    }


def sample_hypotheses(mod_text: str, ref_embedding: Vector, k: int, cfg: SamplingConfig) -> List[EditHypothesis]:
    hypotheses: List[EditHypothesis] = []

    for idx in range(k):
        if cfg.mode == "textual":
            suffix = f" [hyp_{idx}]"
            hyp_text = mod_text + suffix
            noise_scale = cfg.noise_scale * cfg.temperature
        elif cfg.mode == "latent":
            hyp_text = mod_text
            noise_scale = cfg.noise_scale
        else:
            raise ValueError(f"Unsupported sampling mode: {cfg.mode}")

        noise = gaussian_vector(len(ref_embedding), noise_scale, cfg.seed + idx)
        direction = unit(noise)
        hypotheses.append(
            EditHypothesis(
                text=hyp_text,
                weights=_extract_semantic_weights(hyp_text),
                direction=direction,
            )
        )
    return hypotheses
