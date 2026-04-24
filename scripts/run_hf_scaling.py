#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tts_cir.config import ExperimentConfig, SamplingConfig, ScoringConfig
from tts_cir.hf_pipeline import HFCIRRetriever, load_hf_dataset_split
from tts_cir.retrieval import Candidate
from tts_cir.runner import Query, run_experiment


def _pick_first(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _resolve_image(value: Any, image_root: Path | None = None):
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise ImportError("pillow is required to load image paths.") from exc
    if hasattr(value, "convert"):  # PIL image
        return value.convert("RGB")
    if isinstance(value, dict) and "path" in value:
        value = value["path"]
    p = Path(str(value))
    if image_root is not None and not p.is_absolute():
        p = image_root / p
    return Image.open(p).convert("RGB")


def _build_queries_and_candidates(
    rows: List[Dict[str, Any]],
    retriever: HFCIRRetriever,
    image_root: Path | None,
    reference_keys: Sequence[str],
    target_keys: Sequence[str],
    modification_keys: Sequence[str],
) -> tuple[List[Query], List[Candidate]]:
    query_rows: List[Dict[str, Any]] = []
    image_store: Dict[str, Any] = {}

    for idx, row in enumerate(rows):
        ref = _pick_first(row, reference_keys)
        tgt = _pick_first(row, target_keys)
        mod = _pick_first(row, modification_keys)
        if ref is None or tgt is None or mod is None:
            continue
        ref_id = str(ref.get("path")) if isinstance(ref, dict) and "path" in ref else str(ref)
        tgt_id = str(tgt.get("path")) if isinstance(tgt, dict) and "path" in tgt else str(tgt)
        query_rows.append(
            {
                "query_id": str(row.get("query_id", f"q_{idx}")),
                "reference_id": ref_id,
                "target_id": tgt_id,
                "modification_text": str(mod),
            }
        )
        image_store[ref_id] = ref
        image_store[tgt_id] = tgt

    image_ids = list(image_store.keys())
    pil_images = [_resolve_image(image_store[i], image_root=image_root) for i in image_ids]
    image_embs = retriever.encode_images(pil_images).tolist()
    for img in pil_images:
        img.close()

    emb_map = {image_id: emb for image_id, emb in zip(image_ids, image_embs)}
    candidates = [Candidate(image_id=image_id, embedding=emb) for image_id, emb in emb_map.items()]
    queries = [
        Query(
            query_id=q["query_id"],
            reference_embedding=emb_map[q["reference_id"]],
            modification_text=q["modification_text"],
            target_id=q["target_id"],
        )
        for q in query_rows
        if q["reference_id"] in emb_map and q["target_id"] in emb_map
    ]
    return queries, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full HF dataset K-scaling experiment with CLIP embeddings.")
    parser.add_argument("--dataset-repo", type=str, default="chuonghm/Refined-FashionIQ")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--reference-keys", nargs="+", default=["image1", "reference_image", "candidate"])
    parser.add_argument("--target-keys", nargs="+", default=["target", "target_image", "image2"])
    parser.add_argument("--modification-keys", nargs="+", default=["modification", "caption", "rel_caption"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--sampling-mode", type=str, choices=["textual", "latent"], default="textual")
    parser.add_argument("--top-n", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--out-csv", type=Path, default=Path("results/hf_scaling.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("results/hf_scaling.json"))
    args = parser.parse_args()

    ds = load_hf_dataset_split(args.dataset_repo, split=args.split, subset=args.subset)
    rows = [dict(ds[i]) for i in range(min(len(ds), args.max_samples))]
    retriever = HFCIRRetriever(model_name_or_path=args.model_id, device=args.device)
    queries, candidates = _build_queries_and_candidates(
        rows=rows,
        retriever=retriever,
        image_root=args.image_root,
        reference_keys=args.reference_keys,
        target_keys=args.target_keys,
        modification_keys=args.modification_keys,
    )

    cfg = ExperimentConfig(
        top_n=args.top_n,
        k_values=args.k_values,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        sampling=SamplingConfig(mode=args.sampling_mode),
        scoring=ScoringConfig(),
    )
    results = run_experiment(queries, candidates, cfg)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "k",
                "r1",
                "r1_ci_low",
                "r1_ci_high",
                "r5",
                "r10",
                "oracle_r100",
                "forward_passes",
                "wall_clock_s",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.k,
                    f"{r.r1:.4f}",
                    f"{r.r1_ci_low:.4f}",
                    f"{r.r1_ci_high:.4f}",
                    f"{r.r5:.4f}",
                    f"{r.r10:.4f}",
                    f"{r.oracle_r100:.4f}",
                    r.forward_passes,
                    f"{r.wall_clock_s:.4f}",
                ]
            )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as handle:
        json.dump([r.__dict__ for r in results], handle, indent=2)

    print(f"Queries: {len(queries)}  Candidates: {len(candidates)}")
    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
