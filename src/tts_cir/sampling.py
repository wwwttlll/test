from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Dict, List, Sequence

from .config import SamplingConfig
from .vector_ops import Vector, gaussian_vector, unit

_TEMPLATE_BANK: List[str] = [
    "{mod}",
    "keep identity, then {mod}",
    "same object but {mod}",
    "apply a subtle edit: {mod}",
    "preserve shape and composition, {mod}",
    "edit only requested attributes: {mod}",
    "minimal change request: {mod}",
]

_LEXICAL_BANK: Dict[str, List[str]] = {
    "elegant": ["refined", "polished", "sophisticated"],
    "casual": ["relaxed", "informal", "everyday"],
    "vintage": ["retro", "classic", "old-school"],
    "minimal": ["clean", "simple", "uncluttered"],
    "remove": ["without", "exclude", "drop"],
}


@dataclass(frozen=True)
class EditHypothesis:
    text: str
    weights: Dict[str, float]
    direction: Vector


class CLIPTextEncoder:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: str = "cpu") -> None:
        try:
            import torch
            from transformers import AutoTokenizer, CLIPTextModel
        except Exception as exc:  # pragma: no cover - optional deps
            raise ImportError("Install transformers+torch to use latent text direction.") from exc

        self._torch = torch
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = CLIPTextModel.from_pretrained(model_id).to(device)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> List[Vector]:
        batch = self._tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self._device) for k, v in batch.items()}
        with self._torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
            embeddings = hidden[:, 0, :]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().tolist()


@lru_cache(maxsize=2)
def _get_text_encoder(model_id: str, device: str) -> CLIPTextEncoder:
    return CLIPTextEncoder(model_id=model_id, device=device)


def _extract_semantic_weights(text: str) -> Dict[str, float]:
    low = text.lower()
    return {
        "obj": 1.0,
        "attr": 1.0 if any(t in low for t in ["color", "material", "sleeve", "long", "short"]) else 0.8,
        "style": 1.0 if any(t in low for t in ["elegant", "refined", "vintage", "minimal", "casual"]) else 0.6,
        "neg": 1.0 if any(t in low for t in ["without", "remove", "not", "no "]) else 0.5,
    }


def _template_rewrite(mod_text: str, idx: int, cfg: SamplingConfig) -> str:
    rng = random.Random(cfg.seed + idx)
    text = _TEMPLATE_BANK[idx % len(_TEMPLATE_BANK)].format(mod=mod_text)
    for src, choices in _LEXICAL_BANK.items():
        if src in text.lower() and rng.random() < max(cfg.temperature, 0.2):
            replacement = choices[idx % len(choices)]
            text = text.replace(src, replacement).replace(src.capitalize(), replacement.capitalize())
    return text


def _match_dim(vec: Vector, dim: int) -> Vector:
    if len(vec) == dim:
        return vec
    if len(vec) > dim:
        return vec[:dim]
    return vec + [0.0] * (dim - len(vec))


def text_to_direction(modification_text: str, ref_embedding: Vector, model_id: str = "openai/clip-vit-large-patch14", device: str = "cpu") -> Vector:
    encoder = _get_text_encoder(model_id, device)
    text_embedding = encoder.encode([modification_text])[0]
    text_embedding = _match_dim(text_embedding, len(ref_embedding))
    raw = [t - r for t, r in zip(text_embedding, ref_embedding)]
    return unit(raw)


def sample_hypotheses(mod_text: str, ref_embedding: Vector, k: int, cfg: SamplingConfig) -> List[EditHypothesis]:
    hypotheses: List[EditHypothesis] = []
    for idx in range(k):
        if cfg.mode == "textual":
            hyp_text = _template_rewrite(mod_text, idx, cfg)
            noise = gaussian_vector(len(ref_embedding), cfg.noise_scale * cfg.temperature, cfg.seed + idx)
            direction = unit(noise)
        elif cfg.mode == "latent":
            hyp_text = _template_rewrite(mod_text, idx, cfg)
            direction = text_to_direction(hyp_text, ref_embedding)
        else:
            raise ValueError(f"Unsupported sampling mode: {cfg.mode}")

        hypotheses.append(
            EditHypothesis(
                text=hyp_text,
                weights=_extract_semantic_weights(hyp_text),
                direction=direction,
            )
        )
    return hypotheses
