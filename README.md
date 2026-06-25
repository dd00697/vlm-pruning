# Localisation-Aware Visual Token Pruning for VLMs

Research code for visual-token pruning experiments on vision-language models.
The repo supports full-token inference, random pruning, uniform/grid pruning,
prior-method adapters for VisionZip and FastV, and an isolated official Nuwa
evaluation path.

Generated data, metrics, predictions, figures, and W&B files are intentionally
kept out of git.

## Setup

Create a Python environment with CUDA-enabled PyTorch, then install the runtime
dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure optional cache and W&B settings as needed:

```bash
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT=vlm-pruning
```

Initialize third-party references before using prior-method or Nuwa runners:

```bash
git submodule update --init --recursive
```

## Official Baseline Sweep

The official baseline path evaluates LLaVA-1.5 with full visual tokens and with
random or uniform visual-token retention. The sweep script prepares missing
manifests and runs each baseline sequentially:

```bash
python scripts/run_official_baseline_sweep.py \
  --precision bf16 \
  --wandb-project vlm-pruning \
  --skip-existing \
  --resume
```

Common filters:

```bash
python scripts/run_official_baseline_sweep.py \
  --datasets pope,aokvqa,scienceqa_img,textvqa,refcoco \
  --methods full,uniform,random \
  --tokens 432,288,58 \
  --random-seeds 0 \
  --precision bf16 \
  --wandb-project vlm-pruning \
  --skip-existing \
  --resume
```

Use `--wandb-mode disabled` or `--no-wandb` for local-only runs.

## Single Official Run

Prepare a manifest:

```bash
python scripts/prepare_official_baseline_data.py \
  --dataset pope \
  --split val \
  --output data/manifests/official_baselines/pope_val.jsonl \
  --download
```

Run one baseline:

```bash
python scripts/run_official_baseline.py \
  --dataset pope \
  --task vqa \
  --manifest data/manifests/official_baselines/pope_val.jsonl \
  --method uniform \
  --target-visual-tokens 288 \
  --precision bf16 \
  --wandb-project vlm-pruning
```

For RefCOCO, use `--task grounding` and a RefCOCO manifest.

## Prior-Method Runs

VisionZip and FastV adapters run against the same manifest format:

```bash
python scripts/run_prior_methods.py \
  --task grounding \
  --dataset refcoco \
  --manifest data/manifests/refcoco_val.jsonl \
  --method visionzip \
  --keep-ratio 0.50 \
  --limit 500
```

```bash
python scripts/run_prior_methods.py \
  --task vqa \
  --dataset pope \
  --manifest data/manifests/pope_val.jsonl \
  --method fastv \
  --keep-ratio 0.50 \
  --limit 500
```

The saved 500-example comparison sweep helper is:

```powershell
.\scripts\run_prior_sweep.ps1
```

## Official Nuwa Runs

Official Nuwa is evaluated through its original LLaVA assumptions in an isolated
runner:

```powershell
.\.venv\Scripts\python.exe scripts\run_official_nuwa.py `
  --task vqa `
  --dataset pope `
  --manifest data\manifests\pope_val.jsonl `
  --method official_nuwa `
  --matain-token 64 `
  --distance 280 `
  --limit 1 `
  --seed 0 `
  --no-wandb
```

Sweep helpers:

```powershell
.\scripts\run_official_nuwa_sweep.ps1 -Limit 500 -Seed 0
.\scripts\run_official_nuwa_matched_baselines.ps1 -Limit 500 -Seed 0
```

## Summaries

Aggregate official baseline tables and reports:

```bash
python scripts/summarize_official_baselines.py
```

Combine official Nuwa rows with matched HF-stack baselines:

```bash
python scripts/summarize_official_nuwa_results.py --allow-missing
```

Primary outputs are written under:

- `experiments/runs_official_baselines/`
- `experiments/figures_official_baselines/`
- `experiments/tables/`
- `experiments/reports/`

## Repository Layout

- `scripts/prepare_official_baseline_data.py` prepares official JSONL manifests.
- `scripts/run_official_baseline.py` runs one full, random, or uniform baseline.
- `scripts/run_official_baseline_sweep.py` orchestrates the official baseline matrix.
- `scripts/summarize_official_baselines.py` aggregates official baseline CSVs.
- `scripts/run_baseline.py` contains shared LLaVA pruning and scoring utilities.
- `scripts/run_prior_methods.py` runs VisionZip and FastV comparison adapters.
- `scripts/run_official_nuwa.py` runs official Nuwa in an isolated path.
- `third_party/` contains pinned external comparison code as submodules.
