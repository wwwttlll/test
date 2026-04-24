# Experiment Status (as of 2026-04-24)

## Completed experiments

1. **MVP K-scaling dry-run on toy JSONL data**
   - Script: `scripts/run_mvp.py`
   - Outputs: `k, r1/r5/r10 (+bootstrap CI), oracle_r100, forward_passes, wall_clock_s`
   - Purpose: verify pipeline correctness, not SOTA claims.

2. **Unit tests for core components**
   - `tests/test_scoring_and_metrics.py`
   - `tests/test_hf_pipeline_import.py`

3. **HF asset download workflow validated**
   - Script: `scripts/download_hf_assets.py`
   - Config: `configs/hf_assets.json`

4. **Implemented experiment infra upgrades**
   - Textual-hypothesis template bank (LLM-free)
   - Latent mode now uses CLIP text direction (`text_embedding - ref_embedding`)
   - Multi-seed + bootstrap CI reporting
   - New full-run script: `scripts/run_hf_scaling.py`

## Pending experiments (must-run)

1. **Fair-budget main result (Block 1)**
   - CIRR + FashionIQ with Equal-FWD and Equal-Time
   - Compare TTS-CIR vs WISER baseline (`arXiv:2602.23029`)

2. **Novelty isolation (Block 2)**
   - full vs w/o structured score vs K=1

3. **Simplicity/Pareto check (Block 3)**
   - TTS-CIR final vs overbuilt serial variants

4. **Failure diagnosis by query type (Block 5)**
   - attribute / object / style buckets

## Models used now

- **Implemented default backbone**: `openai/clip-vit-large-patch14`
- **Framework stack**: `transformers`, `datasets`, `torch`, `faiss-cpu`

## Datasets used now

- **Currently run in repo**: toy local JSONL samples under `data/`
- **Configured for HF download**: `chuonghm/Refined-FashionIQ`

## Baseline policy

- Primary baseline is now fixed to **WISER** (`https://arxiv.org/pdf/2602.23029`).
- Baseline run entry is provided at `scripts/run_wiser_baseline.py`.
- Baseline experiment config is tracked in `configs/baselines/wiser_2602_23029.yaml`.
