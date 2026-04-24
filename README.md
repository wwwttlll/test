# test

## TTS-CIR experiment scaffold (framework-first)

This repository provides a runnable CIR experiment scaffold and **Hugging Face download/inference tooling**.

### Project structure

- `src/tts_cir/`
  - `hf_pipeline.py`: framework-first pipeline based on `transformers + torch + faiss + datasets`
  - `runner.py`: lightweight K-scaling experiment loop
  - `sampling.py`, `scoring.py`, `metrics.py`, `budget.py`: MVP logic
- `scripts/download_hf_assets.py`: download models and datasets from Hugging Face
- `configs/hf_assets.json`: model/dataset download config
- `scripts/run_mvp.py`: local MVP runner from JSONL files
- `tests/`: unit tests

---

## 1) Install

### Minimal (for local MVP code + tests)

```bash
python -m pip install -e . --no-build-isolation
```

### Framework stack (recommended)

```bash
python -m pip install -e .[hf,dev] --no-build-isolation
```

> `hf` extra installs mature third-party frameworks (`huggingface_hub`, `datasets`, `transformers`, `torch`, `faiss-cpu`).

---

## 2) Download models & datasets from Hugging Face

Edit `configs/hf_assets.json` first if needed.

```bash
python scripts/download_hf_assets.py --config configs/hf_assets.json
```

Private/gated repo:

```bash
python scripts/download_hf_assets.py --config configs/hf_assets.json --token <HF_TOKEN>
```

Default config downloads:

- model: `openai/clip-vit-large-patch14`
- dataset: `chuonghm/Refined-FashionIQ`

---

## 3) Run MVP experiment (JSONL inputs)

Query JSONL line format:

```json
{"query_id":"q1","reference_embedding":[0.1,0.2],"modification_text":"make it elegant","target_id":"img_9"}
```

Candidate JSONL line format:

```json
{"image_id":"img_9","embedding":[0.2,0.3]}
```

Run:

```bash
python scripts/run_mvp.py \
  --queries data/queries.jsonl \
  --candidates data/candidates.jsonl \
  --out results/mvp.csv \
  --k-values 1 2 4 8 16 \
  --sampling-mode textual
```

Output columns:

- `k`, `r1`, `r5`, `r10`, `oracle_r100`, `forward_passes`, `wall_clock_s`


---

## 4) Run WISER baseline (arXiv:2602.23029)

We add a **WISER-like serial refinement baseline runner** for fair-budget comparisons.

```bash
python scripts/run_wiser_baseline.py \
  --queries data/queries.jsonl \
  --candidates data/candidates.jsonl \
  --out results/wiser_baseline.csv \
  --rounds 1 2 3 \
  --sampling-mode textual
```

Baseline metadata/config:

- `configs/baselines/wiser_2602_23029.yaml`

---

## 5) Current experiment status

See: `docs/experiment_status.md`

This file tracks:

- experiments already executed,
- pending must-run blocks,
- currently used model(s) and dataset(s),
- baseline lock decision (WISER as primary baseline).
