# test

## TTS-CIR experiment scaffold

This repository now includes runnable MVP experiment code for the TTS-CIR idea:

- `src/tts_cir/`: core package
  - `sampling.py`: test-time hypothesis sampling (`textual` / `latent`)
  - `scoring.py`: structured score + negative penalty
  - `runner.py`: K-scaling experiment loop and budget logging
- `scripts/run_mvp.py`: CLI entry for running experiments
- `tests/`: basic unit tests for metrics/scoring
- `tts_cir_action_plan.md`: experiment strategy and milestones

## Quick start

```bash
python -m pip install -e .
```

Prepare JSONL files:

- queries JSONL line format:

```json
{"query_id":"q1","reference_embedding":[0.1,0.2],"modification_text":"make it elegant","target_id":"img_9"}
```

- candidates JSONL line format:

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
