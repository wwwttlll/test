from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class RetrievedItem:
    image_id: str
    score: float


class HFCIRRetriever:
    """CIR retrieval pipeline built on mature third-party frameworks.

    Stack:
    - transformers (CLIPModel / CLIPProcessor)
    - torch (inference)
    - faiss (ANN retrieval)
    """

    def __init__(self, model_name_or_path: str = "openai/clip-vit-large-patch14", device: str = "cpu") -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Please install extras: pip install -e .[hf] --no-build-isolation"
            ) from exc

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self._index = None
        self._ids: List[str] = []

    def encode_texts(self, texts: Sequence[str]):
        batch = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.torch.no_grad():
            emb = self.model.get_text_features(**batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def encode_images(self, images: Sequence["Image.Image"]):
        batch = self.processor(images=list(images), return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.torch.no_grad():
            emb = self.model.get_image_features(**batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def build_faiss_index(self, image_ids: Sequence[str], image_embs) -> None:
        try:
            import faiss
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Please install faiss-cpu: pip install -e .[hf] --no-build-isolation"
            ) from exc

        dim = int(image_embs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(image_embs)
        self._index = index
        self._ids = list(image_ids)

    def search(self, query_embs, top_k: int = 10) -> List[List[RetrievedItem]]:
        if self._index is None:
            raise RuntimeError("Call build_faiss_index first.")
        scores, idx = self._index.search(query_embs, top_k)

        output: List[List[RetrievedItem]] = []
        for row_scores, row_idx in zip(scores, idx):
            row: List[RetrievedItem] = []
            for s, i in zip(row_scores.tolist(), row_idx.tolist()):
                row.append(RetrievedItem(image_id=self._ids[i], score=float(s)))
            output.append(row)
        return output


def load_hf_dataset_split(repo_id: str, split: str = "train", subset: str | None = None):
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Please install datasets: pip install -e .[hf] --no-build-isolation"
        ) from exc

    return load_dataset(repo_id, subset, split=split)
