from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .vector_ops import Vector, dot, norm


def cosine_similarity(a: Vector, b: Vector) -> float:
    denom = (norm(a) * norm(b)) + 1e-8
    return float(dot(a, b) / denom)


@dataclass(frozen=True)
class Candidate:
    image_id: str
    embedding: Vector
    image_path: str | None = None
    caption: str | None = None


@dataclass(frozen=True)
class DatasetQuery:
    query_id: str
    reference_image: str
    target_image: str
    modification_text: str
    category: str | None = None


class CandidateIndex:
    def __init__(self, candidates: List[Candidate]) -> None:
        self.candidates = candidates

    def top_n(self, query_vec: Vector, n: int) -> List[Tuple[Candidate, float]]:
        scored = [(cand, cosine_similarity(query_vec, cand.embedding)) for cand in self.candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]


def load_jsonl_records(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_cirr_jsonl(records: Iterable[Dict[str, object]]) -> List[DatasetQuery]:
    parsed: List[DatasetQuery] = []
    for idx, row in enumerate(records):
        reference = str(row.get("reference_image") or row.get("reference") or "")
        target = str(row.get("target_image") or row.get("target") or "")
        mod_text = str(row.get("rel_caption") or row.get("caption") or "")
        query_id = str(row.get("query_id") or f"cirr_{idx}")
        if reference and target and mod_text:
            parsed.append(
                DatasetQuery(
                    query_id=query_id,
                    reference_image=reference,
                    target_image=target,
                    modification_text=mod_text,
                    category=None,
                )
            )
    return parsed


def parse_fashioniq_jsonl(records: Iterable[Dict[str, object]]) -> List[DatasetQuery]:
    parsed: List[DatasetQuery] = []
    for idx, row in enumerate(records):
        reference = str(row.get("image1") or row.get("candidate") or row.get("reference_image") or "")
        target = str(row.get("target") or row.get("target_image") or row.get("image2") or "")
        mod_text = str(row.get("modification") or row.get("caption") or "")
        category = str(row.get("category")) if row.get("category") is not None else None
        query_id = str(row.get("query_id") or f"fiq_{idx}")
        if reference and target and mod_text:
            parsed.append(
                DatasetQuery(
                    query_id=query_id,
                    reference_image=reference,
                    target_image=target,
                    modification_text=mod_text,
                    category=category,
                )
            )
    return parsed


def build_candidates_from_paths(image_paths: List[str], retriever, image_root: Path | None = None, batch_size: int = 16) -> List[Candidate]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install pillow to build embeddings from image paths.") from exc

    root = image_root or Path(".")
    uniq_paths = list(dict.fromkeys(image_paths))
    candidates: List[Candidate] = []
    for start in range(0, len(uniq_paths), batch_size):
        batch_paths = uniq_paths[start : start + batch_size]
        imgs = [Image.open(root / p).convert("RGB") for p in batch_paths]
        embs = retriever.encode_images(imgs)
        for p, emb in zip(batch_paths, embs.tolist()):
            candidates.append(Candidate(image_id=p, embedding=emb, image_path=p))
        for img in imgs:
            img.close()
    return candidates
