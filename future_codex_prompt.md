Work autonomously to complete the active goal in `/workspace/vlm-pruning/GOAL.md`. First read
`GOAL.md` and `AGENTS.md` completely, inspect the current repository and completed campaign state,
and continue until the required terminal report says `GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED`.

This is a CPU-only evidence-analysis and dissertation goal. Do not run any new model inference,
GPU quality/timing job, tuning, ablation, external method, transfer task or exploratory subgroup
search. Treat all completed experiment artifacts as immutable. Use only the accepted Full and
frozen `Z_M448_0` testA/testB rows from
`experiments/frozen_z_m448_0_refcoco_testab_20260820/`.

Before editing, freeze the new protocol and analysis schema and create the required atomic
dissertation/PDF backup. Implement exactly one post-hoc paired localization threshold-crossing
analysis: the 2x2 correctness transitions at IoU 0.5, fixed IoU-threshold sweep 0.30--0.70 by 0.05
with deterministic 10,000-resample paired intervals per split, and the specified paired-IoU
summaries. Produce only one compact table, one two-panel figure and a short clearly labelled
post-hoc discussion. Reproduce the frozen transition counts in `GOAL.md` or fail closed. Do not
claim that any single pruning stage caused the result.

Remove the obsolete earlier-method error analysis from the active dissertation dependency graph
and asset namespace, but preserve byte-hashed recoverable copies and never delete underlying
experiment evidence. Then recursively audit every active experimental number, table, figure and
claim in the dissertation against accepted physical artifacts. Independently recompute values from
raw rows where possible. Correct or qualify dissertation prose/rounding/labels; never alter raw
evidence. Leave no unsupported active result.

Review the whole dissertation for clear examiner-friendly explanation, consistent research
questions, methods, results, discussion, limitations and conclusions. Make focused evidence-backed
edits, not gratuitous rewriting. Recompute word count, rebuild with `latexmk`, run textual QA, render
and visually inspect every PDF page, then atomically publish a byte-identical
`/workspace/dissertation_final.pdf`.

Use up to three agents only for independent read-only audits or disjoint page inspection, and
reconcile full coverage yourself. Compile changed Python, run the complete unit suite,
`git diff --check`, protected/source checks and the fail-closed terminal audit. Do not stop after a
plan, partial analysis or draft. Stop only after
`experiments/frozen_z_error_analysis_report_audit_20260820/FINAL_COMPLETION_REPORT.json` exists with
issues empty and status `GOAL_COMPLETE_HUMAN_REVIEW_REQUIRED`; then report the mixed held-out result
unchanged and hand off for human review.
