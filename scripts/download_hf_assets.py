#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _download_model(entry: Dict[str, Any], token: str | None) -> str:
    from huggingface_hub import snapshot_download

    repo_id = entry["repo_id"]
    local_dir = entry["local_dir"]
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    return f"model {repo_id} -> {path}"


def _download_dataset(entry: Dict[str, Any], token: str | None) -> str:
    from datasets import load_dataset

    repo_id = entry["repo_id"]
    split = entry.get("split", "train")
    subset = entry.get("subset")
    local_dir = Path(entry["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(repo_id, subset, split=split, token=token)
    output_dir = local_dir / split.replace("/", "_")
    ds.save_to_disk(str(output_dir))
    return f"dataset {repo_id}[{split}] -> {output_dir}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models and datasets from Hugging Face.")
    parser.add_argument("--config", type=Path, default=Path("configs/hf_assets.json"))
    parser.add_argument("--token", type=str, default=None, help="HF token, or set HF_TOKEN env var.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    token = args.token

    logs: List[str] = []
    for model in cfg.get("models", []):
        logs.append(_download_model(model, token))

    for dataset in cfg.get("datasets", []):
        logs.append(_download_dataset(dataset, token))

    for line in logs:
        print(f"[OK] {line}")


if __name__ == "__main__":
    main()
