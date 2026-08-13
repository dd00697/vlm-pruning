# Goal: K=12, B=48 Primary Evaluation and Official-Baseline Expansion

## Status and authority

This is the active execution specification for the final dissertation experiment campaign.
It supersedes any stale K=20 experiment or reporting instructions in `AGENTS.md`, `GOAL.md`,
scripts, notes, tables, and report prose, but it does not erase the historical audit trail.
Before doing experiment work, update `AGENTS.md` so that future agents cannot accidentally
follow the superseded K=20 or restricted-baseline instructions. Preserve the completed
`GOAL.md` unchanged as a record of the earlier campaign.

The implementation agent must read this file in full, then read the complete repository,
the complete dissertation source, and `/workspace/dissertation_final.pdf` before changing
code or launching jobs. It must inspect existing artifacts before deciding that a run is
missing. Never manufacture, interpolate, copy, or silently relabel a result.

## Objective

Complete, validate, and report an end-to-end final evaluation of Q-TAC with:

1. one primary Q-TAC configuration pruning after decoder layer `K=12` and retaining exactly `B=48` visual
   tokens;
2. a clean RefCOCO validation mechanism ablation that includes a pure query top-k selector;
3. every relevant previously incomplete, partial, malformed, or configuration-mismatched
   run rerun to completion;
4. a broader multimodal benchmark suite;
5. fair comparisons against FastV and several recent localization/token-pruning methods,
   using only their official repositories for the method implementation;
6. a primary matched-final-token comparison at exactly 48 visual tokens;
7. complete artifact, metric, statistical, efficiency, provenance, and PDF validation; and
8. an updated dissertation and supporting documentation whose claims are exactly supported
   by verified artifacts.

The campaign is complete only when all mandatory gates in this document pass, the report is
rebuilt and visually inspected, and a machine-readable completion report proves the result.

## Non-negotiable scientific rules

### The primary configuration

Use one locked primary configuration throughout the headline evaluation:

- model/checkpoint: the repository's canonical LLaVA-1.5-7B checkpoint, pinned by exact
  identifier, revision, and local file hashes;
- precision: BF16 unless the official method genuinely requires another precision, in which
  case disclose it and do not mix its speed/memory figures without qualification;
- seed: 0 for the deterministic primary evaluation;
- pruning boundary: after decoder layer `K=12`;
- final/post-pruning visual-token budget: exactly `B=48`;
- query policy: `query_max`;
- selector: `single_layer`;
- scoring/fusion: `structured_tca`;
- role allocation: the existing verified K=12 `more_anchors` allocation of 21 target,
  12 context, and 15 anchor tokens;
- token ordering: gap-preserving;
- cache: layer-aware;
- generation, decoding, prompt templates, image processing, and stopping rules: lock before
  looking at new benchmark results and hold constant wherever method APIs permit.

If repository inspection shows that a field name differs, map the intended semantics to the
actual configuration and record the mapping. Do not silently change the scientific meaning.

### What “B=48” means

The headline pruning comparison is matched at exactly 48 final retained visual tokens.
For a method that prunes inside the language model, `B=48` means that its steady-state visual
sequence after the method's pruning event contains exactly 48 image tokens for every scored
example. For a method that selects before the language model, exactly 48 visual tokens must
enter the language model. Count and persist the actual visual-token sequence at every layer.

Matching final tokens is the primary quality comparison requested for the dissertation, but
it is not claimed to equalize computation: a method that processes 576 visual tokens in early
layers can cost more than a method that enters the LLM with 48. Therefore every headline row
must also report per-layer token counts, effective token-layer area, FLOPs or a defensible
compute proxy, peak memory, TTFT, and end-to-end latency.

No row may be called “matched B=48” on the basis of an approximate average, a nominal pruning
ratio, or 48 text-plus-image tokens. It must be 48 final visual tokens on every evaluated
sample. Adaptive official methods that cannot guarantee this are ineligible for the comparison.

### K=12 reporting rule

The dissertation's primary/frozen-evaluation section must discuss only `K=12`. It must not
say that K=20 was changed, replaced, removed, or used previously. It must not contain `K=20`,
`K20`, “changed from”, “previously K”, “original K”, or an equivalent change narrative.
An automated section-level check must enforce this before the PDF is accepted.

Do not create a false preregistration claim. Results produced before this campaign may be
described as verified existing K=12 results, and the K=12 protocol can be prospectively locked
for new datasets. If needed for research integrity, place a concise protocol-history entry in
the experiment ledger or a provenance appendix outside the primary/frozen-evaluation section.
That audit note must not distract from the K=12-only scientific presentation. Historical K=20
artifacts must remain immutable, but they are excluded from headline tables, figures, abstract,
main findings, discussion, conclusion, and K=12 primary evaluation prose.

### No result-driven tuning

Freeze protocol, datasets, prompts, scoring, checkpoint, and baseline configurations before
reading the new quality results. Use smoke tests only for compatibility and integrity. Smoke
examples must not be used to select a quality configuration. Log all amendments with time,
reason, files affected, and whether any outcome was already visible.

## Required comparison design

### Headline comparison: exact final B=48 first

The main cross-method table must prioritize the user's intended operating point:

- unpruned/full LLaVA-1.5-7B reference (`576` visual tokens, clearly labelled as the reference,
  not as a matched-budget method);
- random K=12/B=48 control;
- uniform/spatial K=12/B=48 control;
- Q-TAC primary K=12/B=48;
- every official-repository baseline that passes the officialness, parity, and exact-B48 gates.

Use the same canonical model weights, image preprocessing, prompts, output constraints,
dataset samples, and official metric semantics wherever scientifically possible. A method that
cannot pass these gates is not shown as though it were a comparable B=48 result.

### Scope boundary: exact B=48 only

This campaign contains no supplementary budget sweeps or extra operating points. Do not run
Q-TAC or published baselines at B=72, B=128, nominal paper defaults, or other token budgets as
part of this goal. Those experiments require a separate future protocol after this campaign is
complete. A published method that cannot use its official implementation to guarantee exactly
48 final visual tokens is excluded from the cross-method result table rather than run at a
different budget.

## Required datasets and metrics

Use official data, official split definitions, pinned revisions/checksums, and official metric
semantics. Record license/access constraints. Resolve the user's shorthand as follows:

| Requested name | Locked benchmark/split | Primary metric(s) |
|---|---|---|
| GQA | balanced test-dev, or the closest officially scoreable held-out split if test-dev labels are unavailable | official accuracy |
| MMB | MMBench English dev | official circular-evaluation accuracy where supported; otherwise disclose exact official dev protocol |
| MMMU | validation, all subjects | official accuracy |
| MME | full official perception and cognition suite | official perception, cognition, and total scores |
| VQAv2 | validation | official VQA soft accuracy |
| VQAtext | TextVQA validation | official VQA-style soft accuracy |
| POPE | random, popular, and adversarial subsets | official accuracy, precision, recall, F1, and yes-ratio |
| SQA | ScienceQA-IMG official image-only evaluation split | official accuracy |
| MMVet | official MM-Vet evaluation split | official judge-based score and evaluator version |
| one more RefCOCO dataset | RefCOCO+ val, testA, and testB | Acc@0.5 and mean IoU using the locked box parser |

Keep the existing RefCOCO val/testA/testB evaluation as the mechanism and held-out localization
study. RefCOCO+ is the additional referring-expression dataset and must not silently replace it.

If an official leaderboard-only split cannot be locally scored, use the explicitly named
official locally scoreable split and state the substitution before running. For MM-Vet, use
the official judge/evaluator and pinned judge model. Existing task-relevant credentials may be
used if already configured. If official scoring requires unavailable external credentials,
still produce and fully validate predictions, document the scoring blocker, exclude the score
from quantitative aggregation, and do not substitute exact match or a home-grown judge.

Do not use generic exact-match scoring for MM-Vet, VQAv2, TextVQA, MME, MMBench, or any task
whose official metric differs. Prefer the benchmark's official evaluator or a pinned official
implementation exposed through a trusted evaluation framework. Confirm sample IDs and answer
normalization against official examples.

## Required RefCOCO mechanism ablation

On the exact same RefCOCO validation manifest, checkpoint, prompts, seed, parser, K=12 pruning
policy, and B=48 budget, produce or verify all of:

1. full/unpruned reference;
2. random K=12/B=48;
3. uniform/spatial K=12/B=48;
4. pure query top-k K=12/B=48: rank patches only by query relevance and keep the top 48;
5. query top-k plus the locked NMS/diversity rule, K=12/B=48;
6. complete structured Q-TAC, K=12/B=48.

The pure query top-k run must not use anchors, anchor-conditioned context, neighborhood quotas,
NMS, structured allocation, gap reinsertion, or another hidden diversity mechanism. Verify its
retained indices on sampled records and save machine-readable selection traces. This ladder is
the causal test of whether anchors/context add value beyond query relevance alone.

Report Acc@0.5, mean IoU, paired sample count, paired differences versus pure query top-k and
full, bootstrap 95% confidence intervals, and a paired randomization/permutation result where
appropriate. Use paired per-example data, not independent standard errors.

If complete, schema-valid, configuration-exact versions of these runs already exist, reuse them
after audit rather than spending GPU time. If any artifact is partial, mislabelled, based on a
different manifest/configuration/parser, or lacks enough information to verify it, rerun it.

## Existing-run and “incomplete run” audit

Before launching full runs, inventory every experiment referenced by the dissertation, README,
technical report, viva notes, tables, figures, claim-evidence files, and prior goal. Build a
machine-readable ledger with one row per claimed result and these statuses:

- `verified_reusable`;
- `rerun_required_incomplete`;
- `rerun_required_corrupt`;
- `rerun_required_wrong_config`;
- `superseded_not_reported`;
- `development_only_not_claimed`;
- `blocked_external_scoring`.

For every artifact, verify at least: exact configuration fingerprint, checkpoint identity,
dataset and split revision, manifest hash, expected and actual row count, unique IDs, ordering,
JSON/JSONL schema, finite values, box format, parser version, per-sample token counts, completion
sentinel, process exit state, and independently recomputed metrics.

Rerun every incomplete or invalid experiment that remains relevant to a K=12/B=48 claim. Do not
rerun superseded K=20 experiments merely to fill an old table; remove those stale claims from
the current scientific presentation. If a prior K=12 result passes every audit gate, reuse it
with provenance. No “incomplete,” “partial,” “subset result presented as full,” mixed-manifest,
or undersized run may remain in the final report.

Specifically audit the existing K=12 RefCOCO val/testA/testB, random, uniform, pure-query,
NMS/diversity, structured, position-policy, error-analysis, and timing artifacts. Existing
development-configuration latency may not be attributed to the locked primary configuration.
If exact K=12/B=48 primary latency does not exist, measure it afresh under the serial timing
protocol below.

## Official-repository baseline policy

### Target methods

Attempt the following method implementations from their author-maintained official repositories:

1. FastV;
2. VisionZip;
3. FEATHER;
4. Nüwa/Nuwa (use the exact paper/repository spelling in citations);
5. TransPrune.

This yields FastV plus three recent localization/token-pruning baselines and the already relevant
VisionZip comparison. During literature verification, confirm that each paper, publication year,
official repository, checkpoint support, and method scope are correctly identified. Do not call
a preprint “published in 2026” unless the venue and year are verified. If a named work is not in
scope after verification, explain why and omit it rather than forcing a misleading comparison.

### What counts as official

- Clone only the author-linked official repository, record URL, immutable commit SHA, retrieval
  date, license, paper identifier, and environment lock.
- Keep each repository in an isolated, clearly named `third_party_official/` location or external
  immutable cache. Do not commit large weights or datasets.
- The pruning/localization/merging algorithm must execute from official source. Do not recreate
  it in the local Q-TAC runner and do not use the repository's older local FastV/VisionZip
  approximations for headline results.
- Thin dataset, prompt, logging, and output-conversion glue may wrap the official method without
  altering its selection logic. Store and review every glue layer.
- Minimal compatibility patches are allowed only for dependency/API compatibility. Save the
  patch, explain it, and prove that it does not change method semantics. Any patch to selection,
  scoring, token allocation, merging, or layer scheduling disqualifies the run as official.
- Use official checkpoints/configs when required. For the matched table, require the same
  canonical LLaVA-1.5-7B weights or demonstrate exact checkpoint equivalence.

### Baseline admission gates

Each method must pass all applicable gates before entering a result table:

1. official repository and paper identity verified;
2. environment reproducibly installs without contaminating the primary environment;
3. official minimal example completes;
4. one authors-reported reference behavior/result is reproduced within a preregistered and
   justified tolerance, or an equivalent deterministic parity test is passed;
5. unpruned/reference output parity against the canonical harness is within a locked tolerance;
6. canonical sample IDs, prompt, images, decoding, and output normalization are aligned;
7. token traces prove the claimed budget and pruning layer behavior;
8. outputs and official metrics pass the common artifact validator.

If the official implementation genuinely cannot be made to work, it may be skipped. Before
skipping, perform and record: documentation-based installation, the official example, an isolated
compatible dependency attempt, checkpoint verification, and diagnosis of the first real failure.
Record repository SHA, commands, environment, logs, error, remediation attempts, and the reason
further work would require changing method semantics or unavailable assets. Mark the method
`skipped_official_failure`; do not substitute a third-party port or local reimplementation, and
do not include an unverified number from its paper as if it were an experimental result.

Maintain `method_eligibility.json` with exactly one status per method:

- `matched_b48_eligible`;
- `excluded_no_exact_b48_support`;
- `skipped_official_failure`;
- `excluded_checkpoint_or_parity_mismatch`;
- `out_of_scope_after_literature_verification`.

The official-repository restriction applies to published pruning methods. The locally defined
random, uniform, pure-query, NMS, full, and Q-TAC rows are experimental controls/method variants,
not claims to reproduce another paper.

## Protocol freeze and run manifests

Create a dated campaign directory such as:

`experiments/k12_b48_expanded_official_baselines_20260813/`

It must contain, at minimum:

- `PROTOCOL.md` and a machine-readable protocol snapshot;
- `AMENDMENTS.jsonl`;
- `artifact_ledger.json`;
- `method_eligibility.json`;
- dataset/checkpoint/repository checksum manifests;
- one immutable run manifest per method/configuration/dataset/split;
- raw logs and raw predictions;
- canonical normalized predictions;
- selection/per-layer token traces;
- recomputed metric outputs and official evaluator outputs;
- integrity reports;
- timing/efficiency outputs;
- statistical-analysis outputs;
- report-input tables and figure data;
- `FINAL_COMPLETION_REPORT.json`.

Every run ID must encode or map unambiguously to method, method commit, model/checkpoint,
dataset/split, K where applicable, B, selector/config, prompt revision, seed, precision, and
protocol revision. Compute a configuration fingerprint and reject resume/merge across differing
fingerprints.

Predictions must be written to temporary/sharded paths, flushed and validated, then atomically
published. Never let a crashed process leave a filename that appears complete. Completion is
proved by a sentinel written only after row-count and schema validation. Preserve failed shards
for diagnosis with an explicit failed status.

## Dataset and harness preparation

1. Reuse already downloaded data only after checksum and split verification.
2. Obtain missing official datasets through their documented access path; pin revisions and
   checksums without committing data.
3. Build canonical sample manifests with stable IDs and explicit expected counts.
4. Lock one prompt/evaluation protocol per dataset before quality runs.
5. Validate image paths, answer annotations, label mappings, multi-image handling, and split
   leakage.
6. For each official method, translate canonical samples into its input API and normalize its
   raw output back to a canonical prediction record without changing content.
7. Run a tiny structural smoke test on every method/dataset pair; inspect output structure only.
8. Validate the official evaluator on a known fixture or published example before scoring runs.

Where a task framework such as lmms-eval is used, pin its exact version and task revision. A
framework score is acceptable only if its task implementation matches the benchmark's official
metric; verify this rather than assuming it.

## Execution matrix

### Stage A: K=12 RefCOCO completion

- Audit/reuse/rerun the full six-row RefCOCO validation mechanism ladder.
- Audit/reuse/rerun the clean K=12/B=48 primary and full reference on RefCOCO testA and testB.
- Produce paired confidence intervals and per-category/error analysis.
- Verify or rerun the exact primary position-ordering control if it is discussed in the report.
- Measure exact-primary serial efficiency; do not reuse timing from a different anchor allocation.

### Stage B: exact-B48 broad benchmark matrix

For every required dataset/split, run or verify:

- full reference;
- random K=12/B=48;
- uniform K=12/B=48;
- Q-TAC K=12/B=48;
- each `matched_b48_eligible` official method.

Use one primary deterministic seed for the complete quality matrix. Add multiple seeds only for
stochastic controls or methods, or for a focused variance study; never average incompatible
configs. Random and stochastic rows should use a preregistered small seed set (at least 3 where
feasible), with seed-level and pooled summaries retained.

### Stage C: qualitative/error analysis

Create a deterministic paired sample of successes, failures, and disagreement cases across
full, pure query top-k, structured Q-TAC, and eligible official methods. Verify every displayed
image, prompt, prediction, target, selected-token overlay, and caption against raw artifacts.
No cherry-picked qualitative claim may be presented without the selection rule being recorded.

## Parallelism and GPU safety

Up to three quality runs may execute in parallel only when GPU memory permits.

Before enabling three-way concurrency:

1. run each distinct method/environment alone on a short representative soak;
2. measure steady and peak allocated/reserved GPU memory;
3. verify device assignment and that processes do not silently share the wrong GPU;
4. reserve at least `max(10% of device VRAM, 4 GiB)` unallocated headroom per device;
5. ensure the sum of measured peaks plus headroom fits on each assigned GPU;
6. run a three-process soak and validate every output.

If any condition fails, reduce to two or one concurrent run. Never trade correctness for three-way
parallelism. Use an explicit scheduler/registry so no two processes claim the same run or output
path. OOM, worker death, truncated stdout, missing completion sentinel, row-count mismatch, or
GPU reset invalidates the affected run and triggers a clean rerun from a valid checkpoint/shard.

All latency and memory benchmarking must run serially with no other GPU workload. Parallel quality
runs must never be used for headline latency numbers.

## Efficiency protocol

For full, Q-TAC B=48, and every eligible official B=48 method, use a fixed stratified sample and:

- batch size 1, locked precision and generation settings;
- sufficient warm-up iterations before measurement;
- at least five measured repetitions per sample/configuration or another preregistered robust
  protocol if total generation makes this infeasible;
- synchronized GPU timing;
- identical hardware, clocks/power mode where observable, software stack, and no co-tenants;
- TTFT, end-to-end latency, tokens/second where meaningful, selector/pruner overhead, peak GPU
  memory, visual tokens per layer, token-layer area, KV-cache footprint, and FLOPs/compute proxy;
- median plus dispersion/confidence intervals, with raw repetitions retained.

Do not compare latency across separate incompatible environments without reporting the environment
difference. If an official method cannot run in the common environment, show quality and token
efficiency in the common table but place runtime in an environment-qualified table.

## Integrity and validation gates

No metric can enter a table until all of these pass:

- expected row count equals actual unique scored IDs;
- no missing, duplicate, extra, or reordered-without-key records;
- every record matches the locked manifest and configuration fingerprint;
- every image and annotation checksum resolves;
- predictions and metric inputs are finite and schema-valid;
- localization boxes use the declared coordinate convention and are in bounds after conversion;
- generation did not terminate with an error/empty output without explicit handling;
- actual token traces match the claimed budget and layer schedule;
- official evaluator completes and its raw output is preserved;
- an independent recomputation from canonical predictions matches the reported value within
  numerical tolerance;
- summary tables are generated from validated machine-readable results, never hand-transcribed;
- spot checks trace published table cells back to raw per-example records.

Create a validator that fails closed. Its failure must block report generation rather than emit
a warning. Add regression fixtures for known prior artifact failure modes: partial JSONL,
duplicate IDs, wrong manifest, mixed configs, missing terminal newline/truncated record, stale
summary, incorrect box scaling, and nominal versus actual token-count mismatch.

## Statistical analysis

- Use paired bootstrap confidence intervals for metric differences on shared samples.
- Use dataset-appropriate units: examples for VQA/localization, question-image pairs where
  official metrics do so, and benchmark-defined components for MME.
- Keep official aggregate scores primary; do not invent a cross-benchmark average over
  incomparable units.
- Report absolute quality, change from full, retained percentage where meaningful, and efficiency.
- For multiple methods/datasets, emphasize effect sizes and intervals rather than selectively
  declaring wins from point estimates.
- A claim that Q-TAC improves over a baseline is allowed only when the exact comparison and
  uncertainty support it. Otherwise use neutral wording such as “comparable,” “higher point
  estimate,” or “trade-off.”

## Report revision requirements

Update the dissertation sources and every derived/repeated claim, including abstract, introduction,
methods/protocol, experiment chapters, tables, figure captions, discussion, limitations, conclusion,
appendices, README, technical report, viva notes, and claim-evidence ledger.

The final narrative must:

1. present K=12/B=48 as the sole primary Q-TAC setting;
2. contain no K=20 results or transition story in the primary/frozen-evaluation section and no
   K=20 headline claim elsewhere;
3. explain the exact-B48 comparison first and why final-token matching does not itself equalize
   compute;
4. keep the 576-token full reference visibly distinct from the exact-B48 method comparison;
5. show the pure query top-k mechanism ablation and interpret what the anchor/context increment
   does or does not establish;
6. identify official repository and commit for every published method result;
7. mark skipped methods transparently without presenting unofficial substitutes;
8. use only verified complete runs and official metrics;
9. distinguish prospective new-dataset protocol locking from retrospective reuse of verified
   existing K=12 artifacts;
10. report limitations including evaluator dependence, method/environment differences, and any
    unscored externally blocked task;
11. avoid claiming universal superiority from a heterogeneous benchmark suite; and
12. ensure all numbers, rounding, sample counts, split names, labels, and citations agree across
    prose, tables, figures, and appendices.

Remove stale K=20 macros, cached tables, plots, captions, labels, auxiliary text, and claim-ledger
entries from the current report inputs. Preserve historical experiment artifacts outside the
presentational path. Search both source and extracted final-PDF text for prohibited stale claims.

For the primary/frozen-evaluation section specifically, extract its final text and fail the build
if it contains any forbidden K=20/change wording. It may say only what was evaluated at K=12; it
must not imply that K=12 was preregistered before results that were already observed.

## Report build and visual QA

1. Back up the current dissertation PDF and record its checksum.
2. Generate tables and figures only from validated derived artifacts in a dedicated directory,
   such as `Dissertation_Draft/analysis/derived/k12_b48_expanded_official_baselines/`.
3. Rebuild the dissertation from a clean auxiliary-file state using the documented build process.
4. Fail on missing citations/references, undefined labels, missing figures, stale generated inputs,
   or material LaTeX errors.
5. Review the build log for overfull boxes and resolve all visible/material layout problems.
6. Render every PDF page and visually inspect tables, plots, captions, page breaks, equations,
   images, overlays, font embedding, cropping, blank pages, and legibility at normal reading size.
7. Extract final PDF text and cross-check key numbers, K=12-only section wording, method names,
   split names, and references.
8. Atomically publish `/workspace/dissertation_final.pdf` only after every quality gate passes.
9. Reopen and checksum the published PDF; confirm it is the same approved build.

Do not overwrite the last known-good PDF until the replacement passes all gates.

## Reproducibility and repository hygiene

- Keep datasets, model weights, large raw caches, and cloned third-party git histories out of the
  main repository and `.gitignore` them where appropriate.
- Track code, configs, manifests, hashes, summaries, validators, patches, report sources, and small
  evidence artifacts needed to reproduce the analysis.
- Preserve unrelated user changes and do not use destructive git commands.
- Record all package versions, CUDA/driver/GPU information, environment variables relevant to
  behavior (without secrets), and official repository commits.
- Never log credentials or commit secrets.
- Run repository tests, new validator tests, and relevant end-to-end smoke tests.
- Make one or more intentional local commits with clear messages after validation; do not push.

## Mandatory final completion report

Write `experiments/k12_b48_expanded_official_baselines_20260813/FINAL_COMPLETION_REPORT.json`
and a concise Markdown companion. They must include:

- protocol/configuration fingerprints;
- exact dataset splits, revisions, checksums, and expected/scored counts;
- every run ID and status;
- reused-versus-rerun provenance;
- every previously incomplete artifact and its resolution;
- baseline official repo URLs, SHAs, environments, compatibility patches, parity evidence, and
  eligibility status;
- actual final and per-layer token-count audits;
- metric recomputation and official-evaluator checks;
- statistical output paths;
- efficiency/timing validation;
- report table/figure lineage;
- report source and final PDF checksums;
- source/text/visual PDF QA results;
- test results;
- amendment log summary;
- skipped or externally blocked items and precise reasons;
- local git commit IDs; and
- an overall `complete: true` only if every mandatory gate is satisfied.

The Markdown companion must state plainly what was run, what was reused, what was skipped, what
the exact-B48 comparison shows, and any limitations. It must not overstate results.

## Definition of done

This goal is done only when:

- the active repository instructions agree with this K=12 campaign;
- all relevant prior incomplete runs have either been validly rerun or removed as superseded and
  unclaimed;
- the six-way RefCOCO val mechanism ladder is complete and validated;
- clean K=12/B=48 RefCOCO testA/testB evidence is complete;
- the required broader datasets have complete validated full/control/Q-TAC predictions and official
  scores, except an explicitly documented external official-scoring blocker;
- every published baseline is official-source verified and is either validly reported at exact
  B=48, excluded because its official implementation cannot guarantee exact B=48, or transparently
  skipped under the failure policy;
- all reported B=48 methods prove exactly 48 final visual tokens on every sample;
- no non-B48 pruning result is mixed into the exact-B48 comparison;
- all efficiency measurements used in claims were obtained serially under the locked protocol;
- every reported number traces to a complete artifact and independent metric validation;
- the dissertation and all supporting documentation consistently report only the K=12 primary
  setting, with the special K=12-only section rule passing;
- the final PDF passes source, build-log, text, and page-by-page visual QA;
- the completion report says `complete: true` truthfully; and
- clean local commits exist, with nothing pushed remotely.

If any mandatory condition cannot be completed without unavailable access, credentials, or a
method-semantic rewrite, do not fabricate completion. Exhaust safe in-scope diagnostics, record
the blocker precisely, use only the explicitly permitted skip/exclusion rules above, and keep
the corresponding claim out of the report.
