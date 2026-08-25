# Repository scope

`GOAL.md` in the repository root is the only active execution specification. Read it completely
before taking task actions. This RunPod goal authorizes one new, frozen IVC-Prune campaign on an
NVIDIA GPU, followed by evidence-bound dissertation updates. It supersedes the previous
no-new-experiments goal only after a human deliberately copies this file and the adjacent
`GOAL.md` into the repository root.

Never read or modify `legacy_goal.md`, `legacy_goal2.md` or
`PRO_REASONING_RESEARCH_HANDOFF.md`. Never push, publish, open a pull request, or alter remote
state. Local focused commits are allowed.

# Scientific boundaries

- The selected dissertation method remains `Z_M448_0`; its validation selection and held-out
  testA/testB result are final and immutable.
- IVC-Prune is a post-hoc external reference replacing FEATHER. It cannot tune, select, reject,
  or otherwise change the dissertation method.
- Run only `IVC_STACK_FULL` and `IVC_PRUNE` on the frozen RefCOCO validation and timing manifests.
  Never run IVC on testA, testB, another benchmark, subgroup, prompt, layer, or parameter setting.
- Use official IVC-Prune commit `03afd9cee41f60bb30b51a4105839f2b5d467814`, its LLaVA
  configuration (`ivcp_k=16`, image start 5, image length 576), and Hugging Face checkpoint
  revision `a272c74b2481d8aff3aa6fc2c4bf891fe57334fb`.
- The only permitted upstream source modification is the registered removal of the unconditional
  debug print from the timed forward path. Do not change model arithmetic or selection logic.
- Completed experiment artifacts, predictions, manifests, configs, ledgers, reports, hashes, and
  previous FEATHER evidence are immutable. Correct the dissertation or wrapper, never evidence.
- New outputs belong only under `experiments/ivc_prune_refcoco_val_20260825/` and registered
  dissertation output paths.
- GPU jobs must be serial. Stop if another compute process is present. Never run two model jobs in
  parallel.

# Execution discipline

- Freeze and hash source, manifests, model revision, prompt, code, and protocol before accepted GPU
  execution.
- Smoke-test both methods before quality or timing. A smoke run is diagnostic only and cannot be
  reported as a result.
- Independently align and rescore all 8,811 quality rows. Never trust copied summary values.
- Record the actual retained visual indices and 32-layer token trace for every IVC request. IVC is
  adaptive; do not label it with an invented fixed budget.
- Time five complete natural-generation repetitions on the frozen 1,000-row subset, with 20
  warmups and the registered alternating order. Include all IVC scoring, selection, cache pruning,
  and gather overhead.
- Run the fixed-32 diagnostic only if the frozen analysis trigger requires it.
- Compare IVC speed and paired quality only with `IVC_STACK_FULL`. Cross-stack comparisons with the
  canonical, FastV, or Nüwa stacks are descriptive only.
- Do not edit the dissertation until experiment analysis and audit pass. Snapshot and hash the
  dissertation, current PDF, and active FEATHER report assets before editing.
- Remove FEATHER from active evaluated-baseline results, tables, figures, and claims, but preserve
  its experiment evidence. A relevant literature-review discussion of the FEATHER paper may remain
  if it is clearly literature rather than an experiment performed in this dissertation.
- Bind every changed number and experimental statement to an immutable artifact. Zero unsupported
  active claims may remain.

# Completion

Run the complete test suite, experiment audit, recursive active-TeX claim audit, dissertation build,
word-count check, textual PDF checks, and visual inspection of every rebuilt page. Publish nowhere.
Write `experiments/ivc_prune_refcoco_val_20260825/FINAL_COMPLETION_REPORT.json` with an empty
`issues` array and status `IVC_CAMPAIGN_COMPLETE_HUMAN_REVIEW_REQUIRED`, then stop for human review.
