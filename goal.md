# Codex Goal: Progressive Visual-Token Pruning and Oracle Feasibility Campaign

## Objective

Extend the existing LLaVA-1.5-7B visual-token pruning repository and execute a
validation-only research campaign that determines how aggressively visual
tokens can be removed before any learned selector is considered.

This goal replaces and cancels the unfinished broad K=12/B=48 transfer and
official-baseline campaign archived in `legacy_goal.md`. Do not read
`legacy_goal.md`, resume that campaign, start its controller, repair its
MMBench/MMMU runs, or execute any non-RefCOCO job unless the user explicitly
reactivates that legacy goal in a future request. Preserve all historical files
unchanged and clearly exclude them from the new campaign registry.

The immediate campaign contains exactly 33 required discovery jobs on one
frozen 1,000-example RefCOCO-validation subset. Up to two exploratory timing
jobs are conditional on a promising discovery result. The four originally
planned 8,811-example confirmations are postponed and are not authorized by
this goal:

| Campaign | Maximum new jobs | Purpose |
|---|---:|---|
| K20 extreme-tail sweep | 14 | Find the smallest useful late visual-token budget |
| Progressive pruning | 9 | Test conservative early removal followed by aggressive late removal |
| Oracle diagnostics | 10 | Test whether future-selected original patches can work from the start |
| Full RefCOCO-val confirmations | 0 now | Postponed; recommend only 1-2 later if the extension is selected for the dissertation |
| Exploratory efficiency | 0-2 conditional | Measure a promising tail-only and/or progressive candidate only if its registered gate passes |
| **Immediate maximum** | **35** | 33 quality cells plus at most 2 conditional timing studies |

Work is complete only when the implementation and scientific gates pass, all
33 discovery jobs have complete physical prediction artifacts, any triggered
timing study is complete, metrics have been independently recomputed, the
registry is internally consistent, and one cumulative Markdown research report
ends with an explicit exploratory A/B/C recommendation. Launching jobs is not
completion.

Do not train, implement, or tune a learned selector/router in this goal.

## Authoritative locations

- Repository: `/workspace/vlm-pruning`
- Checkpoint: `llava-hf/llava-1.5-7b-hf`
- Full RefCOCO validation manifest:
  `data/manifests/final_evaluation/refcoco_val_unique.jsonl`
- Expected full-validation rows: 8,811
- New campaign root:
  `experiments/progressive_oracle_refcoco_val_20260814/`
- Frozen discovery manifest to create:
  `experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl`
- Machine-readable registry:
  `experiments/progressive_oracle_refcoco_val_20260814/EXPERIMENT_REGISTRY.json`
- The single cumulative research report:
  `experiments/progressive_oracle_refcoco_val_20260814/RESEARCH_REPORT.md`
- Final machine-readable completion audit:
  `experiments/progressive_oracle_refcoco_val_20260814/FINAL_COMPLETION_REPORT.json`

`RESEARCH_REPORT.md` is the only experiment-report Markdown file to create for
this campaign. Update it after each campaign decision and finish it with the
final conclusion. JSON, JSONL, CSV, and image assets may be created as needed,
but do not create fragmented per-run Markdown reports. Do not edit the
dissertation or `/workspace/dissertation_final.pdf` in this goal.

## Scope and safety

This goal authorizes:

- implementation and tests for generalized multi-stage physical pruning,
  segment-aware caching, oracle index generation/restart, controllers,
  validators, analysis, timing, and reporting;
- creation of fresh local campaign artifacts under the new campaign root;
- local commits needed to establish clean code epochs; and
- execution of the accepted RefCOCO-validation jobs in this specification.

It does not authorize:

- testA, testB, POPE, TextVQA, MME, GQA, MMBench, MMMU, VQAv2, ScienceQA,
  MMVet, RefCOCO+, or any other transfer/held-out evaluation;
- resuming the cancelled K12/B48 expanded-evaluation controller;
- reading or executing `legacy_goal.md` unless the user explicitly reactivates it;
- overwriting, deleting, splicing, or silently repairing historical raw runs;
- model or router training;
- pushing commits, opening a PR, uploading private artifacts, or changing the
  dissertation;
- running any 8,811-example confirmation in this goal;
- exceeding 33 accepted quality jobs or two conditional timing jobs without
  explicit human approval.

Before doing anything else, verify and record that no previous GPU worker is
still active. If one is active, terminate only the identified old campaign
worker/controller and record the command, PID, and UTC timestamp in
`cancellation_record.json`. Never use a broad process kill.

## Fixed scientific protocol

Unless a cell explicitly says otherwise, hold all of the following fixed:

- model: LLaVA-1.5-7B at the exact locally pinned checkpoint revision;
- precision: BF16;
- batch size: 1;
- original image representation: 576 visual patch tokens on a 24x24 grid;
- decoding: deterministic greedy;
- maximum grounding output length: 48 tokens;
- prompt: the repository's registered RefCOCO grounding prompt;
- evaluator: the repository's registered RefCOCO normalized-box parser,
  Acc@0.5, and mIoU implementation;
- position handling: original gap-preserving position IDs;
- query map: query-patch cosine similarity using the current boundary hidden
  states and the same query-token definition used by the validated historical
  single-stage implementation;
- selector: deterministic query top-k unless a cell explicitly says `random`,
  `uniform`, or `none`;
- tie-breaking: descending relevance, then lower original raster index;
- selected visual tokens gathered in original raster order;
- all text tokens retained at every stage; and
- source revision, dependency versions, hardware, manifest SHA-256, config
  SHA-256, and seed recorded for every run.

`Kx -> B` means that the first `x` decoder layers see the incoming visual-token
set and physical pruning happens after decoder layer `x`, using exactly the
same boundary convention as the existing validated `prune_after_layer=x`
implementation. A pre-LLM/oracle selection occurs at `K=0`, before decoder
layer 1.

The average visual-token count for an ordered candidate schedule is the
arithmetic mean of the visual-token counts seen by each of the 32 decoder
layers. Compute it from the realized trace, not only from the requested
schedule. For oracle cells this candidate-path figure explicitly excludes the
privileged dense teacher pass, whose token-layer work is recorded separately.
Examples:

- `K8->256, K12->48`:
  `(8*576 + 4*256 + 20*48) / 32 = 206`;
- pre-LLM 64: `32*64 / 32 = 64`;
- pre-LLM `B_early64` then `K20->B_tail`:
  `(20*B_early64 + 12*B_tail) / 32 <= 64`.

## Phase 0: cancellation, inventory, and frozen discovery subset

### Cancel and isolate the previous campaign

1. Confirm no old controller or CUDA worker remains.
2. Do not run or repair any job under
   `experiments/k12_b48_expanded_evaluation_20260813/`.
3. Treat existing artifacts as read-only historical evidence.
4. Create the new campaign root and ensure every new output path is beneath it.

### Freeze the discovery subset before inspecting new outcomes

Create exactly one deterministic 1,000-row subset of the 8,811-row RefCOCO-val
manifest before launching any new quality experiment.

Use this registered sampling rule:

1. validate the source manifest has exactly 8,811 rows, unique `sample_id`
   values, valid image paths, and a recorded full SHA-256;
2. for each row compute
   `SHA256("progressive-oracle-refcoco-val-20260814-v1\0" + sample_id)`;
3. sort ascending by the full digest, breaking any impossible digest tie by
   UTF-8 `sample_id`;
4. select the first 1,000 IDs;
5. write the selected rows in their original full-manifest order; and
6. record the rule string, source checksum, selected-ID checksum, output
   checksum, row count, and generation code revision in
   `manifests/discovery_manifest_lock.json`.

Use this exact frozen manifest for every discovery cell in Campaigns 1-3. Do
not regenerate it after results are visible.

The repository's old timing manifest is RefCOCO testA and is forbidden here.
Use the new validation discovery manifest for any triggered exploratory timing
study. If no existing full-token timing reference uses this exact validation
manifest, code revision, prompt, precision, and timing implementation, include
an internal full-reference pass in each timing study; do not touch testA/testB
and do not count the internal reference as a new quality cell.

### Historical K20/B48 reference

Locate the complete historical K20/B48 query-top-k/reference predictions that
match the fixed protocol. Validate their physical 8,811-row JSONL, config,
commit, prompt, BF16 precision, exact B=48 trace, unique IDs, and manifest
order. Rescore the frozen 1,000 IDs directly from raw predictions and register
that subset score as `REF_K20_B48`.

Do not trust only `metrics.json` and do not rerun the full historical B=48 cell.
If no valid matched query-top-k K20/B48 artifact exists, stop before discovery,
record the missing-reference blocker, and request explicit human authorization
for a revised job count; do not silently add a 34th quality cell. Historical
random B=48 may be summarized only if its raw predictions pass the same checks.

## Phase 1: implementation and preflight gates

Generalize the current one-boundary path to accept an ordered schedule of
pruning events:

```text
(boundary layer, requested budget, selector)
```

The implementation must also support pre-LLM selection for oracle and control
cells, and the explicit `B=0` late-tail case.

### Multi-stage pruning behavior

At every pruning boundary:

1. retain every text token;
2. identify the currently surviving visual states and their original raster
   indices;
3. for `query_topk`, recompute query-patch cosine scores from the current
   boundary hidden states—never reuse an earlier stage's score map;
4. for `random`, select deterministically from the current surviving visual
   indices using the recorded cell seed and a stage-specific deterministic
   seed derivation;
5. physically remove discarded visual states;
6. keep selected visual states in original raster order;
7. retain their original gap-preserving position IDs;
8. update all sequence/index mappings used by later boundaries; and
9. emit a per-stage trace containing requested/realized budget, incoming and
   outgoing original indices, score-map checksum where applicable, selector,
   role label, sequence lengths, text-token checksum, and position-ID checksum.

For `B=0`, remove all remaining visual states after K20 while retaining the
entire text sequence and generation capability. Record an empty retained-index
list and a valid checksum. Do not replace B=0 with a dummy image token.

Budgets must be monotonically non-increasing and no stage may select an index
that was removed previously. Reject malformed schedules before loading the
model.

### Segment-aware prompt cache

Generalize the layer-aware cache so decoder-layer segments may have different
prompt lengths. For each layer during prefill and generation, record and assert:

```text
actual cache length = that layer's realized prompt length + generated tokens so far
```

Do not pad a shorter segment with discarded visual states merely to satisfy a
uniform cache API. Preserve the existing behavior for single-stage schedules.

### Oracle implementation

For each oracle example:

1. run a dense teacher prefill through the specified K12 or K20 teacher layer;
2. compute the normal query-conditioned relevance ranking over the 576 original
   patches;
3. serialize only original integer patch indices, their ranking checksum, and
   non-value metadata needed for provenance;
4. destroy/detach all teacher hidden states, activations, caches, and tensors;
5. start a fresh inference from the normal projector output and freshly
   initialized model/cache state;
6. gather only the chosen original projected patch embeddings, in raster
   order, with original position IDs; and
7. run the restarted candidate without access to any teacher hidden-state
   values.

The teacher is privileged only for choosing indices. Future hidden vectors,
attention maps as values, caches, logits, pooled features, or transformed
embeddings must never enter restarted inference. For O7/O8, any K20 tail
query-top-k score is recomputed from the restarted candidate's current K20
hidden states; it is not copied from the dense teacher.

Every oracle result must report both:

- `candidate_average_visual_tokens`: average visual tokens across the 32 layers
  of the freshly restarted counterfactual candidate path only; and
- `oracle_teacher_visual_token_layer_work`: the realized dense visual-token
  layer work used to obtain privileged indices, recorded separately with its
  teacher-layer trace (expected `576*12` for a K12 teacher or `576*20` for a K20
  teacher under the registered boundary convention).

Oracle candidate average-token figures are counterfactual upper bounds: they
describe the architecture that could run if a future lightweight predictor
supplied the same indices without executing the teacher. The actual diagnostic
experiment is not efficient because it first pays for the dense teacher pass.
Never combine, hide, or amortize teacher work into a deployable efficiency
claim, and never present oracle cells as deployable timing results.

### Mandatory tests before any discovery launch

Add unit, integration, and real-model smoke tests proving all of the following:

1. **Unpruned parity:** the generalized empty schedule reproduces the existing
   full path exactly on a fixed smoke manifest (token trace, generated text,
   boxes, Acc@0.5, and mIoU).
2. **Single-stage parity:** generalized single-stage K12/B48 and K20/B48
   query-top-k reproduce the existing implementation's retained indices,
   generated text, boxes, and metrics exactly on a fixed smoke manifest.
3. Every stage realizes exactly its requested budget, including B=0.
4. Retained indices are integers, unique, in range, and a subset of the
   preceding stage's retained set.
5. Every original text token survives every pruning event exactly once and in
   order.
6. Original gap-preserving position IDs remain unchanged for all retained
   tokens.
7. Query score maps are recomputed at every query-top-k stage; stage checksums
   and test hooks prove a previous map was not reused.
8. Tie-breaking and seeded random selection are deterministic.
9. Each decoder layer/segment cache length equals its own prompt length plus
   generated-token count throughout decoding.
10. B=0 performs generation with no post-K20 visual states and without a dummy
    replacement token.
11. Oracle teacher output contains indices only; restarted inference receives
    fresh original projector embeddings and no tensor/value derived from
    teacher hidden states.
12. Oracle reruns with the same stored index file reproduce restarted outputs,
    and corrupt/out-of-range/duplicate index files fail closed.
13. Average-token accounting computed from realized per-layer traces matches
    closed-form values for every registered schedule; oracle tests separately
    verify `candidate_average_visual_tokens` and dense
    `oracle_teacher_visual_token_layer_work` without conflating them.
14. Resume/finalization cannot mix rows from different manifests, schedules,
    configs, source revisions, or code epochs.

Run the complete CPU test suite. Then run serial real-GPU smoke tests for full,
single-stage K12/B48, single-stage K20/B48, B=0, one two-stage schedule, one
three-stage schedule, and one oracle restart. Independently validate each smoke
artifact. No discovery run may start until all gates pass from a clean committed
source revision.

### GPU scheduling

- Timing studies are always isolated and serial with no other CUDA process.
- Oracle jobs default to one at a time until measured peak memory proves two
  can coexist with at least 10% GPU-memory headroom.
- Quality jobs may run concurrently only after a representative soak confirms
  semantic parity with serial execution and at least 10% headroom using the
  sum of measured peaks. Never exceed the safe measured width; on the current
  hardware, do not assume three jobs fit.
- The controller must be resumable, skip only fully validated completed cells,
  and stop launching new jobs after any semantic/integrity failure until that
  failure is resolved.

## Campaign 1: K20 extreme-tail sweep (14 jobs)

Use the frozen 1,000-row RefCOCO-val discovery manifest.

### Query-top-k cells (5)

| ID | Schedule | Final selector |
|---|---|---|
| T1 | `K20->32` | query top-k |
| T2 | `K20->16` | query top-k |
| T3 | `K20->8` | query top-k |
| T4 | `K20->4` | query top-k |
| T5 | `K20->0` | none after score/trace boundary |

### Random cells (9)

Use three independent, explicitly recorded seeds `0`, `1`, and `2` at each
budget:

| IDs | Schedule | Selector | Seeds |
|---|---|---|---|
| R16-S0, R16-S1, R16-S2 | `K20->16` | random | 0, 1, 2 |
| R8-S0, R8-S1, R8-S2 | `K20->8` | random | 0, 1, 2 |
| R4-S0, R4-S1, R4-S2 | `K20->4` | random | 0, 1, 2 |

Do not rerun B=48 unless the historical reference is invalid or required by
the single-stage parity gate.

### Automatic Campaign 1 decisions

For each cell compute Acc@0.5, mIoU, parsing rate, and paired per-example
differences against the raw-prediction-derived `REF_K20_B48` subset.

Define `B_tail` as the smallest of `0, 4, 8, 16, 32` for which query top-k is:

- no more than 1.0 percentage point below `REF_K20_B48` in Acc@0.5; and
- no more than 0.005 below `REF_K20_B48` in mIoU.

These are development tolerances, not equivalence or non-inferiority claims.
If no smaller tested budget passes both tolerances, set `B_tail=48`, state that
the sweep found no acceptable smaller tail, and use the validated reference as
the fallback for downstream schedules.

`B_tail` is a **provisional discovery estimate**, not a definitive breaking
point. On 1,000 examples, a 1.0 percentage-point Acc@0.5 difference is only ten
examples. Report the estimate and its bootstrap interval, explicitly state that
it requires 8,811-row confirmation before a final dissertation claim, and do
not use the threshold alone as evidence of statistical equivalence.

At B=16, 8, and 4 compare query top-k with the mean of the three random seeds.
Use paired bootstrap intervals over discovery examples; compute the random
per-example comparator by averaging each example's three seed outcomes before
resampling examples.

B_tail determines the downstream late **budget only**. It does not choose the
progressive late selector because Campaign 1 has random cells only at B=16, 8,
and 4, whereas B_tail may be 0, 32, or the fallback 48. Progressive P7/P8 use
query top-k at K20 by default. P9 tests the competing random selector at the
same B_tail after the strongest progressive prefix, except for the registered
B_tail=0 edge case defined below. Campaign 1 random cells characterize whether
token identity matters after a dense K20 prefix; they do not select the
progressive late-stage selector.

Record separately whether evidence suggests token identity becomes important
again at extreme budgets. Do not infer random safety at small B from historical
B=48 behavior.

Throughout Campaigns 1-3, treat differences around 0.5-1.0 percentage point as
inconclusive unless unusually strong paired evidence supports them. Emphasize
clear multi-point effects, consistent mIoU movement, seed stability, and the
shape of the oracle capacity curve. Every conclusion remains exploratory until
full-validation confirmation is authorized later.

Update `EXPERIMENT_REGISTRY.json` and `RESEARCH_REPORT.md` before launching
Campaign 2.

## Campaign 2: progressive pruning (9 jobs)

Do not use target/context/anchor selection. Use query top-k at every early and
K12 boundary, recomputing relevance from current hidden states.

### Base schedules (6)

| ID | Schedule | Expected average visual tokens |
|---|---|---:|
| P1 | `K8->256, K12->48` | 206 |
| P2 | `K8->192, K12->48` | 198 |
| P3 | `K4->384, K12->48` | 198 |
| P4 | `K4->256, K12->48` | 166 |
| P5 | `K2->384, K12->48` | 186 |
| P6 | `K2->288, K12->48` | 156 |

Assert the realized per-layer trace gives exactly the expected average above.
P3 is the deliberately conservative K4 test: it asks layer 4 to discard only
the most obviously irrelevant one-third of patches before K12 makes the final
48-token choice. P4 tests the more aggressive K4 alternative.

### Select two base schedules

Rank P1-P6 with this registered deterministic Pareto rule:

1. find the highest Acc@0.5;
2. any candidate within 0.5 percentage points of that accuracy enters the
   near-best set;
3. within that set prefer lower realized average visual tokens;
4. use higher mIoU as the next tie-break;
5. use lexicographically lower ID only as the final deterministic tie-break;
6. remove the winner and repeat the same rule on the remaining candidates to
   obtain the second selected schedule.

Store the inputs and full decision trace in `decisions/progressive_base.json`.

### Three-stage refinements (2)

For the two selected bases, retain their first conservative reduction, replace
their K12 final reduction with 96, and add query-top-k `K20->B_tail`:

- P7: refinement of the first selected base;
- P8: refinement of the second selected base.

If the selected base begins at K2/K4/K8 with budget `E`, its refinement is
`Kearly->E, K12->96, K20->B_tail`.

### Late-selector isolation (1)

Select the stronger of P7/P8 using the same accuracy/efficiency/mIoU rule. If
`B_tail > 0`, run P9 with the identical early and K12 schedule but random seed 0
at `K20->B_tail`. This directly isolates query top-k versus random after the
same already-pruned prefix and at the exact same late budget.

If `B_tail = 0`, token identity is undefined because no visual token survives.
Keep the nine-job Campaign 2 count by defining P9 as the identical strongest
progressive prefix with query-top-k `K20->4`. Interpret P9 in this edge case as
a zero-visual-tail versus minimal-nonzero-tail ablation, not as a selector
comparison.

This comparison is mandatory because random pruning after an already-pruned
prefix cannot be inferred from dense-prefix K20 results.

Update the registry and the report with all nine cells, realized average-token
counts, paired intervals for P7/P8/P9 contrasts, and the progressive Pareto
frontier before launching Campaign 3.

## Campaign 3: oracle early-selection diagnostic (10 jobs)

All oracle results are privileged diagnostics, not deployable methods.

### Oracle capacity curve (6)

| ID | Dense teacher ranking | Restarted visual tokens from pre-LLM |
|---|---:|---:|
| O1 | K12 | 64 |
| O2 | K12 | 96 |
| O3 | K12 | 128 |
| O4 | K20 | 64 |
| O5 | K20 | 96 |
| O6 | K20 | 128 |

The restarted path always gathers the selected **original projector patch
embeddings**, not future hidden states. O1 and O4 have realized average visual
tokens exactly 64.

### Guaranteed <=64-average-token oracle architectures (2)

After Campaign 1 determines B_tail, compute the largest allowable oracle early
budget:

```text
B_early64 = floor((64*32 - 12*B_tail) / 20)
```

Register the exact integer and calculation before running O7/O8. Expected
values are:

| B_tail | B_early64 | Candidate average visual tokens |
|---:|---:|---:|
| 48 | 73 | 63.625 |
| 32 | 83 | 63.875 |
| 16 | 92 | 63.500 |
| 8 | 97 | 63.625 |
| 4 | 100 | 64.000 |
| 0 | 102 | 63.750 |

Then run:

| ID | Schedule |
|---|---|
| O7 | K12 teacher chooses `B_early64` original patches; restart pre-LLM; query-top-k `K20->B_tail` |
| O8 | K20 teacher chooses `B_early64` original patches; restart pre-LLM; query-top-k `K20->B_tail` |

If B_tail=0, the K20 event simply removes all remaining visual states and has
no token-identity selector. Otherwise recompute query top-k from the restarted
candidate's current K20 hidden states. Assert the realized candidate trace
satisfies `(20*B_early64 + 12*B_tail)/32 <= 64` exactly. O7/O8 therefore always
measure the oracle quality ceiling of a <=64-average-token counterfactual
candidate, independent of which tail budget Campaign 1 discovers.

### Pre-LLM controls (2)

| ID | Pre-LLM budget | Selector |
|---|---:|---|
| O9 | 64 | random seed 0 |
| O10 | 64 | uniform |

Report the oracle capacity curve and interpret it diagnostically:

- strong oracle 64/96 supports the sufficiency of patch identity and the case
  for learning a future-relevance predictor;
- weak pure-oracle 64 but strong O7/O8 supports a `B_early64`-token early router
  plus late collapse; and
- weak oracle even at 128 indicates early dense information transfer/mixing is
  important and selector prediction alone is insufficient.

Update the registry and report before the deferred-full-validation recommendation
and conditional timing decisions.

## Deferred full-validation confirmation (no jobs in this goal)

Do not launch F1-F4 or any other 8,811-example run. All 33 quality experiments
in this goal use the same frozen 1,000-row discovery manifest.

After Campaigns 1-3, identify—but do not run—the following future candidates:

- the strongest deployable candidate overall; and
- its single most scientifically important comparator, if a comparator is
  needed to support a selector or schedule claim.

Record their exact frozen configurations under
`deferred_full_validation_recommendations` in the registry and in the report.
Recommend running only the strongest candidate, and at most that one comparator,
on all 8,811 RefCOCO-val examples **later**, if the user decides this extension
is worth adding to the dissertation. A future full run requires explicit human
authorization and a new/revised goal.

Do not present any discovery result, `B_tail`, Pareto winner, or oracle/router
decision as fully validated on RefCOCO val. Do not use testA/testB.

## Conditional exploratory efficiency (0-2 jobs)

Timing remains optional. Evaluate the gates below only after all 33 discovery
cells are accepted:

- **E1 gate:** run the strongest deployable K20-tail-only candidate only if
  discovery yields `B_tail < 48` under the registered development tolerances.
- **E2 gate:** run the strongest deployable progressive candidate only if it
  has realized average visual tokens below the existing K12/B48 average of 246,
  loses no more than 2.0 percentage points Acc@0.5 and 0.010 mIoU relative to
  `REF_K20_B48`, and has no unresolved artifact or parsing concern.

If a gate fails, mark that timing cell `skipped_by_design` and explain why. If a
gate passes, run that one isolated timing study on the fixed 1,000-row
RefCOCO-val discovery manifest:

1. E1: strongest qualifying deployable K20-tail-only candidate;
2. E2: strongest qualifying deployable progressive candidate.

Timing jobs must be serial, run with no competing CUDA process, use the existing
timing instrumentation, fixed generated-token policy, warm-up, synchronization,
and environment capture. Each study must compare its candidate to a matched
full-token reference. Reuse an existing reference only if manifest checksum,
prompt, checkpoint revision, BF16 precision, hardware, generated-token policy,
and timing implementation all match. Otherwise include a full-reference pass
inside E1/E2 without adding a held-out or quality experiment.

Record at least:

- vision encoder latency;
- projector latency;
- LLM prefill latency;
- selector/pruning latency at each stage and total;
- decode latency;
- end-to-end latency;
- generated-token count and time per generated token;
- prompt KV storage by layer/segment;
- final KV storage by layer/segment;
- peak allocated and reserved GPU memory; and
- realized per-layer and average visual-token counts.

Label any E1/E2 output **exploratory timing**. It does not replace the
dissertation's five-repetition final timing protocol, and no dissertation
timing claim is to be changed in this goal.

## Artifact and metric validity

Every run writes to a fresh directory and remains provisional until an
independent validator passes it. A `COMPLETED.json` sentinel may be created only
after final validation and must contain hashes of predictions, config, metrics,
integrity report, manifest, and source revision.

Reject or quarantine a run if any of these fails:

1. exact expected row count of 1,000 for every accepted discovery or timing
   artifact in this goal;
2. unique IDs and exact manifest order;
3. exact manifest checksum and registered selection-decision checksum;
4. checkpoint, prompt, BF16, decoding, and source-revision contract;
5. exact requested and realized stage budgets, including zero;
6. valid unique original retained indices and stage-to-stage subset relation;
7. all text tokens retained in order at every stage;
8. unchanged original gap-preserving position IDs;
9. a freshly recomputed query score map at every query-top-k boundary;
10. correct layer/segment prompt and generated cache lengths;
11. local recomputation of parsed boxes, IoU, Acc@0.5, mIoU, and parse rate from
    raw output;
12. complete physical prediction JSONL with no reconstructed rows;
13. source revision/config hashes matching all rows; and
14. for oracle cells, an index-only teacher artifact, proof that no future
    hidden-state value enters restarted inference, a candidate-only average
    token figure, and separately recorded dense teacher token-layer work.

Interrupted partials may be resumed only when manifest, order, config, source
revision, and row hashes prove exact compatibility. Otherwise move them
recoverably beneath `rejected_runs/` with a machine-readable rejection reason;
never splice incompatible epochs or delete them.

Independently recompute all accepted metrics from raw predictions. Compare the
recomputed values to runner aggregates at tight numeric tolerance and fail
closed on mismatch.

## Statistical analysis

Use paired sample-level analysis whenever two runs share the frozen manifest.
For important contrasts, report point estimates and deterministic 95% paired
bootstrap confidence intervals with at least 10,000 resamples and a recorded
analysis seed. Important contrasts include:

- every T1-T5 cell versus REF_K20_B48;
- query top-k versus the three-seed random mean at B=16, 8, and 4;
- P7 versus P8 and the selected refinement versus P9, labelled as a selector
  contrast when B_tail>0 or a zero-versus-four tail contrast when B_tail=0;
- oracle 64/96/128 capacity comparisons and oracle versus O9/O10 controls;
- strongest deployable progressive candidate versus the matched K20-tail
  candidate.

Acc@0.5 intervals are reported in percentage points; mIoU intervals remain on
the 0-1 scale. Do not call an interval containing zero a proven improvement,
equivalence, or non-inferiority. Campaign thresholds are decision rules only.

## Machine-readable experiment registry

`EXPERIMENT_REGISTRY.json` is authoritative and is atomically updated after
each terminal job. It must include:

- all 33 required discovery IDs plus E1/E2 and their status (`pending`,
  `running`, `accepted`, `rejected`, `skipped_by_design`);
- exact schedule, selector, seed, manifest, hashes, source commit, timestamps,
  command, exit code, and artifact paths;
- requested and realized per-layer token traces and average tokens; oracle rows
  must contain `candidate_average_visual_tokens` and
  `oracle_teacher_visual_token_layer_work` as separate fields;
- metrics and parse rate;
- validation results and completion hashes;
- rejection/failure reason and retry lineage;
- oracle/deployable label;
- discovery/full/timing scope;
- Campaign 1 B_tail and dense-prefix query-versus-random characterization;
- P1-P9 selection records;
- the frozen strongest deployable candidate and optional comparator recommended
  for a possible future 8,811-row confirmation; and
- total accepted quality and timing counts, which must never exceed 33 and 2
  respectively.

Retries of a failed job retain the same scientific cell ID with an attempt
suffix and do not create a hidden new scientific comparison. Keep every attempt
in the lineage and count accepted scientific cells, not infrastructure retries,
against the 33-quality-cell and two-timing-cell caps.

## Single cumulative research report

Maintain only
`experiments/progressive_oracle_refcoco_val_20260814/RESEARCH_REPORT.md`.
Initialize it before execution with protocol and blank status tables, then
update it after each campaign without erasing failures or prior decisions.

The final report must concisely contain:

1. research questions and fixed protocol;
2. discovery subset rule and checksums;
3. implementation and parity-test evidence;
4. every registered cell, exact configuration, status, Acc@0.5, mIoU, parse
   rate, and realized average visual-token count;
5. rejected attempts and reasons;
6. K20 tail breaking point and threshold calculation;
7. query-top-k versus random behavior as B decreases, including seed variation;
8. progressive quality/efficiency Pareto frontier and P7/P8/P9 decision;
9. oracle capacity curve, leakage-safety evidence, candidate-only average-token
   figures, and separately reported dense teacher token-layer work;
10. the strongest deployable candidate and optional comparator recommended for
    possible later 8,811-row confirmation, clearly marked **not run**;
11. paired bootstrap intervals for important contrasts;
12. any conditional exploratory timing and memory results, or the recorded gate
    reason for skipping them;
13. strongest deployable candidate;
14. strongest oracle candidate with average visual tokens <=64;
15. limitations, including 1,000-example exploratory scope, the fact that a
    one-point accuracy difference is only ten examples, validation-only
    development, and privileged oracle
    information; and
16. a direct recommended next step.

End the report with exactly one primary recommendation heading and one of these
verbatim recommendation labels:

- **A. Simple progressive pruning is already competitive enough to continue without training.**
- **B. Oracle early selection is strong, so a lightweight future-relevance predictor/router is justified.**
- **C. Even oracle early selection is weak, so selector prediction alone is insufficient and the next work should investigate information transfer, token recovery or another mechanism.**

Choose A/B/C as a **preliminary exploratory recommendation** from the accepted
evidence and explain the decision immediately below it. Marginal 0.5-1.0 point
differences are inconclusive; prefer clear several-point effects, consistent
mIoU/seed evidence, and the oracle-curve shape. State that any dissertation
claim or training decision should first consider the deferred 8,811-row
confirmation. Do not recommend or begin training without human review.

## Completion audit

Before declaring completion, derive a requirement checklist from this file and
verify every item against current artifacts. `FINAL_COMPLETION_REPORT.json`
must prove:

- the previous campaign has no active process and was not resumed;
- no forbidden dataset or testA/testB manifest appears in any accepted command
  or artifact;
- the discovery subset was frozen before new outcomes and has exactly 1,000
  unique ordered rows;
- all mandatory CPU, parity, GPU smoke, B=0, progressive, cache, and oracle
  leakage tests passed;
- O7/O8 use the registered B_early64 calculation and both realized restarted
  candidate averages are <=64, while every oracle cell separately reports its
  dense teacher token-layer work;
- Campaigns 1-3 contain exactly 14, 9, and 10 accepted 1,000-row cells
  respectively;
- no F1-F4 or other new 8,811-row confirmation was launched;
- E1/E2 either passed their registered gate and isolated timing validation or
  are explicitly `skipped_by_design` with the failed gate recorded;
- no more than 33 accepted quality jobs and two accepted timing jobs exist;
- every accepted run has a valid completion sentinel and independently
  recomputed metrics;
- registry totals and report tables match raw artifacts exactly, and the future
  full-validation recommendation is clearly marked not run;
- `RESEARCH_REPORT.md` contains the final A/B/C recommendation and all required
  sections; and
- no learned router/model was implemented or trained.

Run the complete test suite one final time from a clean worktree, record the
result, and create a final local commit containing code, tests, registry,
compact analysis artifacts, the single report, and completion audit. Do not
commit large raw prediction JSONLs or model data unless the repository already
tracks them by policy. Do not push.

Stop after the final report and completion audit. Await human review before any
learned-selector work.
