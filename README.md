# UG-Prune: fixed mid-layer visual-token pruning for LLaVA

This repository contains the implementation, experiment runners and raw
artefacts for an MSc dissertation on training-free visual-token pruning in
decoder-only vision-language models.

The central question is simple: can a model keep grounding quality while
processing fewer image tokens? Full LLaVA-1.5-7B passes 576 visual tokens
through every language-model layer. UG-Prune scores those tokens against the
query, keeps an exact budget at a fixed decoder layer, and continues inference
with the shorter sequence. The retained tokens have three roles:

- target tokens cover high-scoring image regions;
- context tokens retain the radius-one neighbourhood around targets; and
- anchor tokens spread coverage across the image.

The final method is fixed rather than adaptive. It prunes after layer 20, keeps
48 tokens, uses query-max scoring, allocates more of the non-target budget to
anchors, preserves the original position gaps, and uses a layer-aware two-bank
KV cache. There is no learned selector, policy controller or task-specific
retuning after validation.

## Main results

The final configuration was selected on all 8,811 RefCOCO validation referring
expressions and frozen before testA, testB, POPE and TextVQA were examined.
These are the current results derived from the local raw artefacts:

| Evaluation | Samples | Full LLaVA | Frozen UG-Prune, 48 tokens |
|---|---:|---:|---:|
| RefCOCO testA Acc@0.5 | 5,657 | 50.57% | 50.33% |
| RefCOCO testB Acc@0.5 | 5,095 | 38.72% | 38.65% |
| POPE overall accuracy | 9,000 | 86.33% | 86.49% |
| TextVQA soft accuracy | 5,000 | 48.02% | 36.46% |

The paired RefCOCO accuracy intervals include zero on both held-out splits, so
the small observed differences are treated as compatible with comparable
quality, not as proof of equivalence or improvement. POPE shows a similarly
small observed difference. TextVQA is the clear counter-example: aggressive
object-oriented pruning loses substantial text-reading accuracy.

On one NVIDIA A40, five serial 1,000-example timing repetitions gave:

| Configuration | Mean end-to-end latency | Change from full | Mean prompt KV cache |
|---|---:|---:|---:|
| Full LLaVA, 576 tokens | 0.9053 s | reference | 322.8 MB |
| Development reference, K=12/B=48 | 0.8432 s | 6.9% lower | 157.8 MB |
| Frozen method, K=20/B=48 | 0.8828 s | 2.49% lower | 223.8 MB |

For the frozen method, prompt KV storage is 30.67% lower than full inference.
The timing reduction is modest and is not claimed to be hardware-independent.
The selector itself costs about 0.0149 s, which consumes much of the measured
prefill saving.

The locally executed FastV and VisionZip adapters score 7.27% and 3.54%
Acc@0.5 on testA, below the three-seed random-pruning mean of 13.95%. No
positive control validated these adapters against the papers' own reported
results. Their comparison is therefore unvalidated and should not be read as
evidence about the published methods themselves.

Authoritative result sources are:

- `config.json`, `metrics.json` and aligned result JSONL files under
  `experiments/final_evaluation/`;
- `experiments/final_evaluation/protocol/protocol.json` for the registered
  design and environment;
- `experiments/final_evaluation/frozen_method.json` for the selected method;
- `experiments/final_evaluation/FINAL_COMPLETION_REPORT.json` for campaign
  completeness; and
- `../Dissertation_Draft/analysis/derived/final_evaluation/` for regenerated
  tables and intervals.

W&B offline records are retained as secondary provenance. The local artefacts,
not the W&B dashboard, are the numerical source of truth.

`data/`, `experiments/` and W&B outputs are intentionally not Git-tracked. A
GitHub clone therefore contains the compact implementation and protocol
drivers, not the multi-gigabyte raw artefact bundle. The `verify` stage expects
that bundle at the registered paths; a full rerun additionally requires the
datasets prepared by `scripts/prepare_official_baseline_data.py` and the model
checkpoint described below.

## Reproduce the results

The reproduction interface is dry-run by default. Its commands are declared in
`reproduction.yaml`, so there is one readable registry rather than a collection
of shell launchers. This prints the complete workflow without starting an
experiment:

```bash
python scripts/reproduce.py
```

To execute or resume the complete final workflow:

```bash
python scripts/reproduce.py --execute
```

That one command sequences validation selection, provenance equivalence,
freezing, held-out evaluation, transfer evaluation, controlled timing and
evidence rebuilding. It preserves all registered safety gates. Quality runs
use at most the soak-validated parallelism of three. Timing is always serial
and refuses to start while another CUDA process is active. Complete artefacts
are validated and skipped, so the workflow is resumable.

The retained Chapter 5 development ablations are deliberately separate from
the frozen held-out protocol:

```bash
python scripts/reproduce.py ablations          # plan only
python scripts/reproduce.py ablations --execute
```

The final retrospective selector-mechanism ladder and artefact-repair campaign
has its own strict controller. It compares raw query top-k, target NMS only,
and structured target/context/anchor selection at K=12 and K=20 without
changing the already-frozen method. Run its gates in order:

```bash
python scripts/audit_dissertation_artifacts.py
python scripts/run_final_selector_campaign.py --phase register
python scripts/run_final_selector_campaign.py --phase serial --execute
python scripts/run_final_selector_campaign.py --phase soak --execute
python scripts/run_final_selector_campaign.py --phase full --execute --max-parallel 3
python scripts/analyse_final_selector_campaign.py
python scripts/final_selector_completion_audit.py
```

The full phase refuses a dirty source tree and caps concurrency at the result
of the three-process A40 soak. Each job appends only to
`per_sample_results.partial.jsonl`; the canonical result JSONL is published
atomically in manifest order after strict row, ID, token, configuration, and
metric validation. The repair jobs preserve the historical validation row
order through the equivalent unique-ID manifest, because the original manifest
contains duplicate sample IDs that cannot satisfy safe resume semantics.
Set `VLM_CAMPAIGN_PYTHON` to the inference-environment interpreter when it is
not available at the local default `/workspace/.followup_venv/bin/python`.

Each stage can also be invoked directly:

```bash
python scripts/reproduce.py selection
python scripts/reproduce.py postfreeze
python scripts/reproduce.py timing
```

### Verify existing artefacts without inference

The fastest reproduction path does not rerun the model. It compiles and tests
the implementation, validates the existing raw artefacts, and rebuilds the
final CSV tables, figures and claim-to-artefact ledger:

```bash
python scripts/reproduce.py verify --execute
```

Add `--full-audit` to include the already-built dissertation PDF and its
recorded visual-QA status. Neither form runs model inference.

### Environment

The registered environment used Python 3.11.10, PyTorch 2.4.1+cu124, CUDA
12.4, Transformers 5.12.1, Datasets 4.8.5 and W&B 0.28.1. The timing evidence
used one NVIDIA A40 with batch size one and BF16 inference.

The direct package versions recovered from the final W&B artefacts are pinned
in `requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main checkpoint is `llava-hf/llava-1.5-7b-hf`. Set `HF_HOME` if it is not
cached under `/workspace/cache/huggingface`.

Before execution, `scripts/reproduce.py` checks:

- every registered source and manifest SHA-256;
- the recorded Python and core package versions;
- the pinned FastV and VisionZip revisions;
- checkpoint availability; and
- visible GPU hardware for timing.

It fails on drift by default. Explicit `--allow-source-drift`,
`--allow-environment-drift` and `--allow-hardware-drift` flags are available
for diagnostic reruns, but results produced with them are not an exact
reproduction of the registered experiment epoch.

Two limits remain. The registered experiment epoch had a dirty Git working
tree, so the per-file hashes are stronger provenance than its commit alone.
Also, the run metadata records the LLaVA model identifier but not an immutable
Hub revision. The preflight reports both facts rather than claiming bitwise
reproducibility that the artefacts cannot establish.

## Core implementation

- `vlm_pruning/selectors/ug_prune.py`: exact-budget target, context and anchor
  selection.
- `vlm_pruning/features/`: query-token resolution and relevance-map handling.
- `vlm_pruning/llava_hooks/mid_layer_pruning.py`: shortened decoder path,
  position handling and the two-bank cache.
- `scripts/run_ug_prune_single_pass.py`: fixed mid-layer UG-Prune runner.
- `scripts/run_official_baseline.py`: full, random, uniform, FastV and
  VisionZip evaluation.
- `scripts/run_final_evaluation.py`: registered selection campaign and GPU
  soak.
- `scripts/run_final_postfreeze.py`: held-out and task-transfer campaigns.
- `scripts/run_final_timing_campaign.py`: repeated serial efficiency campaign.
- `scripts/run_strong_dissertation_ablations.py`: retained development
  ablations.
- `scripts/run_final_selector_campaign.py`: the retrospective mechanism,
  incomplete-run repair, clean-provenance, smoke and GPU-soak controller.
- `scripts/audit_dissertation_artifacts.py` and
  `scripts/final_selector_completion_audit.py`: pre-launch and independent
  final artefact audits.
- `scripts/analyse_final_selector_campaign.py`: paired bootstrap and spatial
  mechanism analysis plus the supersession ledger.
- `scripts/reproduce.py` and `reproduction.yaml`: the single dry-run-first
  reproduction interface and its declarative command registry.
- `tests/test_phase2_ug_prune.py` and `tests/test_phase2b_single_pass.py`:
  selector, position and cache tests.

The binding scientific protocol is in `GOAL.md`. It distinguishes validation
selection from held-out reporting, fixes the prompts and budgets, and defines
the completion checks used by the orchestrators.
