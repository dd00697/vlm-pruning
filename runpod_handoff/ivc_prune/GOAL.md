# Goal: frozen IVC-Prune replacement campaign and evidence-bound dissertation revision

## 1. Objective

Execute one source-pinned, post-hoc IVC-Prune comparison on the existing dissertation's frozen
RefCOCO validation data, verify the implementation and every produced result, replace FEATHER with
IVC-Prune in the active dissertation evaluation, rebuild and inspect the full dissertation, and
finish with a fail-closed completion report.

This is not a new model-selection campaign. The dissertation's selected method remains
`Z_M448_0`; the completed held-out testA/testB decision remains unchanged. IVC-Prune supplies one
additional external reference only.

## 2. Frozen facts and interpretation

### 2.1 Methods

- `IVC_STACK_FULL`: pruning disabled in the pinned IVC Transformers fork.
- `IVC_PRUNE`: the authors' LLaVA IVC path with `ivcp_k=16`,
  `image_token_start_index=5`, and `image_token_length=576`.

The upstream code computes its selection signal before decoder layer index 16 and physically
prunes before executing layer index 16. State this unambiguously using zero-based code indices;
do not casually call it “layer 15” or “layer 16” without defining the convention. The method keeps
an adaptive union of semantic seed, foreground, and position-aware visual tokens. Record the actual
retained count rather than assigning it a nominal budget.

### 2.2 Pinned source and model

- Official repository: `https://github.com/FireRedTeam/IVC-Prune.git`
- Commit: `03afd9cee41f60bb30b51a4105839f2b5d467814`
- Bundled Transformers version: `4.51.3`
- Model: `llava-hf/llava-1.5-7b-hf`
- Hugging Face revision: `a272c74b2481d8aff3aa6fc2c4bf891fe57334fb`
- Precision: native FP16
- Attention implementation: FlashAttention 2
- Batch size: 1
- Decoding: greedy, one beam, at most 48 generated tokens

The registered patch
`third_party_patches/ivc_prune_remove_debug_print.patch` removes only an unconditional debug print
from the timed forward path. Verify that this is the checkout's only diff. Do not patch any model
calculation, token-selection rule, cache rule, or position handling.

### 2.3 Frozen inputs

- Quality: `data/manifests/final_evaluation/refcoco_val_unique.jsonl`
  - 8,811 ordered rows
  - SHA-256 `f26d7b954189e86f55b50f9506bb670d3a0cf847d3152779af6c08dbf094e3d7`
- Timing: `experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl`
  - 1,000 ordered rows
  - SHA-256 `98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd`
- Prompt: use the exact `REGISTERED_PROMPT` in `vlm_pruning/ivc_campaign.py`.

No testA, testB, other dataset, subgroup analysis, qualitative example search, hyperparameter sweep,
or prompt change is authorized.

## 3. Deliverables

1. A locked source checkout and `SOURCE_LOCK.json`.
2. A frozen `PROTOCOL.json` created before accepted GPU runs.
3. Passing CPU/unit/static tests.
4. Accepted smoke runs for both methods.
5. Complete 8,811-row quality runs for both methods.
6. Five paired natural-timing repetitions per method on all 1,000 requests with 20 warmups.
7. Fixed-32 timing only if the deterministic trigger requires it.
8. Independently rescored quality and hierarchical paired timing analysis.
9. Audited tables/figures generated from analysis artifacts, never hand-entered numbers.
10. A revised dissertation in which FEATHER is no longer an evaluated baseline and IVC-Prune is
    reported with correct stack boundaries.
11. A rebuilt, textually and visually inspected final PDF.
12. `FINAL_COMPLETION_REPORT.json` with no unresolved issues.

## 4. Execution sequence

### Phase A — preserve state and inspect implementation

1. Read this file and root `AGENTS.md` completely.
2. Record `git status`, repository commit, Python/CUDA/driver information, mounted storage, and GPU.
3. Do not discard unrelated user changes. Stop if intended files overlap unknown modifications.
4. Snapshot and hash the current dissertation repository, current final PDF, and all active
   FEATHER-generated tables/figures before any report edit. Store snapshots only under the new
   campaign's `snapshots/` directory. Never modify historical FEATHER experiment evidence.
5. Read completely:
   - `vlm_pruning/ivc_campaign.py`
   - every `scripts/*ivc_prune*.py` file
   - `third_party_patches/ivc_prune_remove_debug_print.patch`
   - relevant existing RefCOCO parser, timing, statistics, artifact, dissertation-build, and audit
     code.
6. Review the pinned upstream IVC files that implement LLaVA loading, `IVCP_forward_flashattention`,
   `compute_attention_weights`, keep-index construction, position IDs, and KV pruning. Confirm the
   wrapper invokes those exact official paths and does not reimplement the selector.

### Phase B — source, environment, and protocol freeze

Use a dedicated Python 3.10 environment. Follow the authors' compatible base unless the installed
RunPod CUDA image requires a documented wheel-index adjustment:

```bash
python scripts/prepare_ivc_prune_source.py \
  --checkout third_party/IVC-Prune \
  --lock-output experiments/ivc_prune_refcoco_val_20260825/SOURCE_LOCK.json

python3.10 -m venv experiments/ivc_prune_refcoco_val_20260825/env
source experiments/ivc_prune_refcoco_val_20260825/env/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e third_party/IVC-Prune/transformers
python -m pip install flash-attn==2.5.8 --no-build-isolation
python -m pip install accelerate pillow numpy matplotlib pytest sentencepiece protobuf
```

If these exact versions cannot load on the available GPU, stop and document the incompatibility;
do not silently upgrade the IVC fork or rewrite its model code. A wheel-index change that leaves
PyTorch 2.3.0 and all method code unchanged is permitted only when fully recorded.

Download the exact model revision and record the returned immutable snapshot path:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="llava-hf/llava-1.5-7b-hf",
    revision="a272c74b2481d8aff3aa6fc2c4bf891fe57334fb",
))
PY
```

Confirm that `transformers.__file__` is beneath `third_party/IVC-Prune/transformers/src`, the
reported version is 4.51.3, the checkout has the pinned commit, and its only diff is the registered
debug-print removal. Then run all CPU tests and freeze the protocol:

```bash
python -m pytest -q
python -m py_compile vlm_pruning/ivc_campaign.py scripts/*ivc_prune*.py
python scripts/init_ivc_prune_campaign.py --model-snapshot <EXACT_SNAPSHOT_PATH>
```

Commit the wrapper, tests, patch, and frozen protocol locally before accepted GPU runs so each run
can bind to clean committed executable source. Do not push.

### Phase C — correctness smoke tests

Ensure no competing GPU process exists. Execute both two-row smoke cells serially:

```bash
python scripts/run_ivc_prune_campaign.py \
  --phase smoke \
  --python experiments/ivc_prune_refcoco_val_20260825/env/bin/python \
  --source-root third_party/IVC-Prune \
  --model-snapshot <EXACT_SNAPSHOT_PATH> \
  --smoke-limit 2
```

Before continuing, manually and programmatically verify:

- the processor produces exactly 576 contiguous visual tokens beginning at sequence index 5;
- `IVC_STACK_FULL` has `[576] * 32` physical visual tokens;
- each IVC row has 16 dense layers followed by 16 layers at one adaptive retained count;
- retained visual indices are sorted, unique, within `[0,575]`, and agree with the physical trace;
- the model config is exactly the registered author configuration;
- outputs parse through the same dissertation box parser;
- only the debug-print patch differs from upstream;
- no result came from a standard Transformers install, CPU offload, quantization, sampling, or a
  different model revision.

Finally, inspect the upstream diff around the disabled path: with `ivcp_config=None`, execution must
take the fork's ordinary language-model branch and the physical trace must remain dense. The
same-stack Full row is the inferential control; do not claim that it is bitwise identical to the
canonical dissertation stack unless a separate parity fixture establishes that fact.

### Phase D — complete quality evaluation

Run the two full validation cells serially:

```bash
python scripts/run_ivc_prune_campaign.py \
  --phase quality \
  --python experiments/ivc_prune_refcoco_val_20260825/env/bin/python \
  --source-root third_party/IVC-Prune \
  --model-snapshot <EXACT_SNAPSHOT_PATH>

python scripts/analyse_ivc_prune_campaign.py --phase quality
```

The analysis must independently align all sample IDs, reparse raw outputs, recompute IoU and
Acc@0.5, validate every layer trace, and produce paired 10,000-resample intervals for IVC minus
IVC-stack Full. No copied `metrics.json` value is accepted without agreement with rescore.

### Phase E — paired timing

Use a quiet GPU, serial jobs, the frozen 1,000-row order, 20 warmups per cell, and five alternating
repetitions:

```bash
python scripts/run_ivc_prune_campaign.py \
  --phase timing-natural \
  --python experiments/ivc_prune_refcoco_val_20260825/env/bin/python \
  --source-root third_party/IVC-Prune \
  --model-snapshot <EXACT_SNAPSHOT_PATH>

python scripts/analyse_ivc_prune_campaign.py --phase timing-natural
```

Inspect `analysis/FIXED32_TRIGGER.json`. Only if `required` is true, run:

```bash
python scripts/run_ivc_prune_campaign.py \
  --phase timing-fixed32 \
  --python experiments/ivc_prune_refcoco_val_20260825/env/bin/python \
  --source-root third_party/IVC-Prune \
  --model-snapshot <EXACT_SNAPSHOT_PATH>
python scripts/analyse_ivc_prune_campaign.py --phase timing-fixed32
```

Use hierarchical paired bootstrap intervals across both repetitions and requests. Do not claim a
speed improvement merely because the point estimate is lower; report uncertainty and the sign of
all repetition means. Include IVC scoring and selector materialization in end-to-end latency.

### Phase F — result audit and staged assets

Build staging assets under the campaign first:

```bash
python scripts/build_ivc_prune_dissertation_assets.py \
  --output-dir experiments/ivc_prune_refcoco_val_20260825/dissertation_assets
```

Check every table cell and plotted point against `quality_analysis.json` and
`timing_natural_analysis.json`. Confirm that adaptive means/minima/maxima are labelled as measured,
not fixed. Confirm IVC quality and speed deltas use `IVC_STACK_FULL` as their paired denominator.

### Phase G — dissertation update

Only after Phases A–F pass, update the active dissertation source in `../Lava-Prune` (or the actual
checked-out dissertation path discovered on RunPod).

1. Rebuild the recursive active-TeX dependency graph before editing.
2. Remove FEATHER as an evaluated method from:
   - abstract result claims;
   - introduction scope/contribution claims;
   - experimental-design implementation and compatibility tables;
   - final validation quality/work tables and figures;
   - timing/resource and uncertainty tables;
   - results interpretation, discussion, conclusions, limitations, and any claimed external-method
     count;
   - generated FLOP/work rows if they are presented as active evaluated evidence.
3. Preserve FEATHER's paper in the literature review only where it genuinely supports background.
   Do not imply the FEATHER experiment was deleted; the evidence remains immutable but is retired
   from active dissertation claims because it was a local adapter rather than an upstream LLaVA
   implementation.
4. Add IVC-Prune accurately:
   - call it the pinned official IVC-Prune implementation, never an adapter;
   - explain its semantic-seed/foreground/position-aware union briefly and clearly;
   - define the zero-based layer boundary;
   - state that its retained token count is adaptive;
   - show `IVC_STACK_FULL` and `IVC_PRUNE` together for within-stack interpretation;
   - identify it as post-hoc and explicitly state it did not affect method selection or held-out
     evaluation.
5. Update the main quality–work figure by removing FEATHER and adding IVC with a visually distinct
   separate-stack marker. Do not draw IVC into the canonical paired/Pareto frontier unless the text
   makes the stack difference unmistakable.
6. Add or replace tables with the smallest set needed:
   - one validation quality/work comparison;
   - one external-stack comparison or clearly separated IVC pair;
   - one timing/resource table with IVC compared only to IVC-stack Full;
   - one uncertainty row for paired IVC minus IVC-stack Full if it aids interpretation.
   Avoid duplicative standalone tables.
7. Regenerate every figure/table from audited JSON. Never type experimental values manually.
8. Update the IVC citation from the official paper/repository and verify all bibliographic metadata.
9. Search for stale FEATHER experimental labels, old method counts, incompatible checkpoint claims,
   and cross-stack speed rankings. Correct all dependent prose, captions, and table references.
10. Use “NVIDIA GPU” in dissertation prose unless a hardware model is scientifically necessary to
    interpret a measured result; preserve exact hardware only in immutable provenance artifacts.

Every numeric or experimental claim changed in the dissertation must be entered into the active
claim-to-evidence ledger with source path, SHA-256, field/key, transformation, and displayed value.
The active claim audit must finish with zero unsupported or mismatched claims.

### Phase H — build and QA

1. Run the complete `vlm-pruning` tests and dissertation-specific tests.
2. Run all experiment and claim audits.
3. Build the dissertation from a clean auxiliary-file state using its registered build command.
4. Run the registered word-count process and confirm compliance.
5. Run `pdftotext` checks for stale FEATHER result labels, required IVC terminology, unresolved
   references/citations, placeholders, malformed Unicode, and accidental debug text.
6. Inspect every page of the rebuilt PDF visually. Check clipping, table width, captions, legends,
   font size, floats, blank pages, review bars, references, and figure readability at normal zoom.
7. Rebuild once more and confirm the inspected PDF is byte-identical to the final published local
   PDF. Do not push or publish it.
8. Run:

```bash
python scripts/audit_ivc_prune_campaign.py \
  --dissertation-root ../Lava-Prune \
  --pdf ../Lava-Prune/Dissertation_Final.pdf
```

If the dissertation's actual main PDF name differs, use the discovered registered output path.

## 5. Stop conditions

Stop and report an issue rather than improvising if any of these occurs:

- the pinned source/model/manifest cannot be obtained or hashed;
- the IVC source contains any unregistered change;
- image tokens do not begin at index 5 under the registered prompt;
- the full control is not dense or IVC's physical trace does not match its keep indices;
- the disabled IVC path is not dense or does not take the fork's ordinary language-model branch;
- any accepted run has missing/duplicate/out-of-order rows or competing GPU processes;
- a timing job overlaps another GPU job;
- independent rescoring differs from stored values;
- a report number lacks immutable evidence;
- the dissertation cannot build cleanly or any PDF page remains uninspected.

Do not work around these failures by changing the method, prompt, checkpoint, layer, data, parser,
or acceptance criteria.

## 6. Required completion report

Write `experiments/ivc_prune_refcoco_val_20260825/FINAL_COMPLETION_REPORT.json` containing at least:

- `status`: exactly `IVC_CAMPAIGN_COMPLETE_HUMAN_REVIEW_REQUIRED`;
- `issues`: exactly `[]`;
- repository and dissertation commits;
- source/model/manifest/protocol hashes;
- run and ledger inventory with hashes;
- independently rescored quality summaries and paired intervals;
- natural timing summaries, repetition means, hierarchical paired intervals, and fixed32 decision;
- adaptive retained-token mean/min/max and token-layer work;
- changed/retired dissertation asset inventory;
- active claim-audit summary with zero unsupported claims;
- test/build/word-count/PDF-text/visual-page-inspection results;
- final PDF path and SHA-256;
- explicit statements that FEATHER evidence was preserved, IVC did not affect candidate selection,
  and no held-out IVC inference occurred.

When and only when all requirements are satisfied, stop for Daniel's human review. Never push.
