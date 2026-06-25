# Localisation-Aware Visual Token Pruning for VLMs

This repository supports dissertation experiments on visual-token pruning for vision-language models, with emphasis on how pruning affects grounding, localisation, and VQA performance. The main runnable baselines are full-token inference, random pruning, uniform/grid pruning, prior-method adapters for VisionZip and FastV, and an isolated official Nuwa evaluation path.

The repository intentionally keeps experiment outputs separate from code. Saved metrics, predictions, tables, and figures are used for dissertation analysis, but generated artifacts are not treated as source files.

## Repository Layout

- `scripts/prepare_data.py`: builds JSONL manifests for supported datasets.
- `scripts/run_baseline.py`: runs full-token, random, and uniform/grid pruning experiments.
- `scripts/run_prior_methods.py`: runs VisionZip and FastV adapter baselines against the same manifest format.
- `scripts/run_official_nuwa.py`: runs the official Nuwa path through original LLaVA internals.
- `scripts/summarize_official_nuwa_results.py`: merges official Nuwa and matched-baseline result tables.
- `data/`, `experiments/`, and `outputs/`: local working directories for datasets and generated results.

## Setup

Create a Python environment and install the project dependencies required by the scripts:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch transformers accelerate datasets pillow pyyaml wandb
```

For CUDA runs, install a PyTorch build that matches the target machine. Quantized local runs can use `--quantization 4bit` or `--quantization 8bit` when the required backend is installed. Full-capacity A100/A4000 runs should use the unquantized settings planned for the final dissertation baselines.

Initialize third-party references when using prior-method or official Nuwa runners:

```bash
git submodule update --init --recursive
```

Pinned references:

- VisionZip: `JIA-Lab-research/VisionZip`
- FastV: `pkunlp-icler/FastV`
- Nuwa: `Man-PaperRejected/Nuwa`
- LLaVA: original LLaVA source used by the official Nuwa path

## Data Manifests

The runners consume JSONL manifests. Grounding rows include an image path, query, and box annotation:

```json
{"sample_id": "refcoco_0001", "image_path": "data/raw/refcoco/val/images/0001.jpg", "query": "the red cup", "bbox": [34, 42, 180, 210]}
```

VQA rows include an image path, question, and answer:

```json
{"sample_id": "pope_0001", "image_path": "data/raw/pope/val/images/0001.jpg", "question": "Is there a dog in the image?", "answer": "yes"}
```

Example manifest preparation:

```bash
python scripts/prepare_data.py --dataset refcoco --split val --download --output data/manifests/refcoco_val.jsonl
python scripts/prepare_data.py --dataset pope --split val --download --output data/manifests/pope_val.jsonl
```

## Baseline Runs

Run a random-pruning grounding baseline:

```bash
python scripts/run_baseline.py \
  --task grounding \
  --dataset refcoco \
  --manifest data/manifests/refcoco_val.jsonl \
  --method random \
  --keep-ratio 0.50 \
  --limit 500 \
  --pruning-implementation true_token_removal
```

Run the matching POPE VQA baseline:

```bash
python scripts/run_baseline.py \
  --task vqa \
  --dataset pope \
  --manifest data/manifests/pope_val.jsonl \
  --method random \
  --keep-ratio 0.50 \
  --limit 500 \
  --pruning-implementation true_token_removal
```

Use `--method full` for the full-token baseline and `--method uniform` for the grid baseline. The key output files are `metrics.json`, per-sample predictions, retained-token diagnostics, qualitative figures, and aggregate CSV/Markdown tables under `experiments/`.

## Prior-Method Runs

VisionZip and FastV are kept as comparison adapters for the existing LLaVA-HF experiment pipeline:

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

The sweep helper for the saved 500-example comparison runs is:

```powershell
.\scripts\run_prior_sweep.ps1
```

## Official Nuwa Runs

Official Nuwa is evaluated through its original LLaVA assumptions in an isolated runner:

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

Run the official Nuwa sweep with:

```powershell
.\scripts\run_official_nuwa_sweep.ps1 -Limit 500 -Seed 0
```

Matched random/uniform baselines at official Nuwa token counts are handled by:

```powershell
.\scripts\run_official_nuwa_matched_baselines.ps1 -Limit 500 -Seed 0
```

## Result Tables

Baseline and prior-method runners update CSV and Markdown tables under `experiments/tables/`. To combine official Nuwa rows with matched HF-stack baselines, run:

```bash
python scripts/summarize_official_nuwa_results.py --allow-missing
```

## Reporting Notes

Do not report placeholder values as results. Use `EXPERIMENT_STATUS.md` and the saved `metrics.json`/CSV tables to verify which experiments have actually completed. When comparing methods, keep model stack, token budget, dataset split, prompt format, and quantization setting explicit.
