# Codex goal restart handoff — adaptive RefCOCO efficiency campaign

Use this file as the authoritative restart handoff. Start or resume a Codex goal with:

```text
/goal Resume and complete /workspace/vlm-pruning/GOAL.md end-to-end. First read /workspace/vlm-pruning/future_codex_prompt.md, /workspace/vlm-pruning/AGENTS.md, and the complete GOAL.md. Treat only GOAL.md as active; never read or execute legacy_goal.md. Audit the persisted state and storage before launching anything, then continue from the valid C480 physical prefix exactly as documented. Do not stop until every adaptive branch is terminal, all required artefacts independently validate, the single cumulative report and registry are complete, and GOAL.md's stopping condition is genuinely satisfied.
```

## Non-negotiable scope

- Repository: `/workspace/vlm-pruning`
- Active specification: `/workspace/vlm-pruning/GOAL.md`
- Never read or execute `legacy_goal.md`; it is deliberately postponed.
- Use only the frozen 1,000-example RefCOCO validation discovery manifest for validation-quality work. Do not run the 8,811-record validation set, testA/testB, transfer datasets, or extra experiments.
- Preserve all historical and current artefacts. Do not overwrite or splice runs.
- Follow the adaptive state machine and registered gates exactly. Do not add budgets, layers, selectors, architectures, datasets, or post-hoc tuning.
- Complete implementation, tests, source-native official comparisons/skips, fixed-route selection, train-only route-label generation, the registered three-seed tiny gate, conditional timing, registry/report updates, and final independent audit.

## Restart/storage facts

At the handoff on 2026-08-15:

- `/workspace` contained about 40 GB and had reached its effective allocation quota even though `df` showed free capacity on the underlying shared filesystem.
- `/` was 20 GB and 98% full, with about 475 MB free.
- The user planned to restart/recreate the RunPod with more storage. Prefer at least 80–100 GB for `/workspace` and at least 40 GB for the container/root disk.
- All irreplaceable project code, data, manifests, experiment artefacts, official-source locks, registry, state, report, and this handoff are under `/workspace`.
- `/workspace/.followup_venv` is under persistent workspace storage and was the canonical Python environment.
- The reproducible Hugging Face model cache was at `/root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf`. It may be lost if the pod is recreated. Redownload it if needed; it is not a scientific result.
- An incomplete scratch venv at `/workspace/.adaptive_reroute_venv` was removed after its installation hit quota. It contained no experiment data.
- No scientific artefact was deleted or moved.

Before any GPU work, verify that this file, `GOAL.md`, the campaign directory, frozen manifest, and Git history still exist; inspect storage headroom; inspect CUDA/OS processes; and run the saved integrity checks. If `/workspace` is not the same persisted volume, stop immediately.

## Repository state at handoff

- Branch restored to `main`.
- Main HEAD before adding this handoff: `fcd397ec772fcdc62a1d7288332b9dfc611c9a4c`.
- The only pre-existing user worktree item was untracked `PRO_REASONING_RESEARCH_HANDOFF.md`. Preserve it and never add, edit, delete, or commit it.
- No experiment CUDA process was active when audited. The only Python process was the normal Jupyter server.
- `GOAL.md` SHA-256: `11bd7b77674e94c45fb1968c83e8aa2e982f05ecf1deaf8d17cb0da837ce9637`
- Frozen validation manifest: `experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl`
- Frozen manifest SHA-256: `98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd`
- Campaign root: `experiments/adaptive_efficiency_refcoco_val_20260815`

Important completed commits:

- `3d7b7b6` — adaptive A0 analysis and campaign initialization
- `98ba0fd` — registered mass-preserving pruning merge
- `048578b` — adaptive campaign preflight audit
- `7960e51` — adaptive fixed-method controller
- `79506a6` — source-native official Reroute adapter
- `fcd397e` — official adapter provenance bug fix

Run the full test suite after restart. It passed 172 tests before the final one-line provenance fix, and the targeted official tests passed afterward.

## Completed scientific work

### A0 and preflight

A0 and preflight are complete and recorded in the campaign registry/report. Preflight passed implementation, exact single-stage parity, merge invariants, interruption/resume, two-job memory soak, and independent validation. The registered safe quality concurrency is two jobs; timing must be serial.

Historical rescored discovery metrics:

- original structured K12/B48: Acc@0.5 `0.439`, mIoU `0.42159786358291806`, average tokens `246`
- pure query-top-k K12/B48: `0.441`, `0.42773466727086573`, average `246`
- P3: `0.408`, `0.4246506509420894`, average `198`
- P5: `0.406`, `0.41979018313550226`, average `186`
- P8: `0.401`, `0.41791704893558784`, average `192`

### C448 accepted

`C448` completed and independently revalidated:

- schedule: K4→448 query top-k, K12→48 query top-k
- 1,000/1,000 rows
- Acc@0.5 `0.415`
- mIoU `0.4265234301573266`
- parse rate `1.0`
- average visual tokens `214.0`
- accepted run: `experiments/adaptive_efficiency_refcoco_val_20260815/runs/c448_attempt1`
- predictions SHA-256: `54c8f48c3d4581529b18afb160e992301d9cc5b4b8016239bc97936659da84ef`
- completion SHA-256: `734f96db6dc576091e241aa2dfbf257f9f0bf4bd4bacd11c4c655b0edb9f34e2`

Relative to P3, C448 changed Acc by `+0.007` and mIoU by `+0.0018727792`. This is the registered ambiguous/discordant region, so both C480 and D1 were triggered. Do not revisit that decision.

## Exact C480 resume state

C480 was first interrupted by `Errno 122: Disk quota exceeded`. After removing only the failed scratch venv, it was resumed under its original exact inference commit and deliberately interrupted for the pod restart.

The valid persisted state is:

- cell: `C480`
- schedule: `4:480:query_topk,12:48:query_topk`
- run directory: `experiments/adaptive_efficiency_refcoco_val_20260815/runs/c480_attempt1`
- valid physical prefix: exactly **586 JSON rows**
- last row sample ID: `refcoco_val_005241_0`
- config SHA-256: `615387bc724d958ae802a8776527016956cbee84ae7db7911e7f7999d3f1b609`
- partial SHA-256 at handoff: `c74ab653c0ea2fb3fc9dee7c106f2161f46913fe92475b15972835244412e106`
- the 586 IDs exactly equal the first 586 frozen-manifest IDs, are unique, parse as complete JSON, and every row contains the recorded config hash
- config source commit: `7960e51b255246cc7d5f95d87da758d0452a8491`

The campaign state intentionally still says `active_workers: ["C480"]` and the registry says C480 is `running`, because the top-level controller was interrupted/crashed before it could clear its state. This is stale bookkeeping, not an active OS process and not corrupt data. Do not start the controller blindly: it will reject the stale active-worker flag.

Resume C480 without creating a new attempt or code epoch:

1. Confirm no process command contains `c480_attempt1` and no CUDA worker is active.
2. Re-run the independent 586-row prefix checks described above.
3. Temporarily use the exact recorded commit with `git switch --detach 7960e51b255246cc7d5f95d87da758d0452a8491`. Ensure there are no tracked worktree edits first; preserve the untracked user handoff.
4. Run:

```bash
/workspace/.followup_venv/bin/python scripts/run_progressive_oracle_refcoco.py \
  --cell-id C480 \
  --manifest /workspace/vlm-pruning/experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl \
  --output-root /workspace/vlm-pruning/experiments/adaptive_efficiency_refcoco_val_20260815/runs \
  --run-name c480_attempt1 \
  --schedule 4:480:query_topk,12:48:query_topk \
  --seed 0 \
  --resume
```

5. Independently validate the completed 1,000-row run.
6. Return to `main` with `git switch main`.
7. Repair the controller's stale-worker recovery transparently: after proving the old process is absent, record a recovery event, clear the stale active-worker marker, and teach/allow the controller to recognize and accept the already completed physical C480 attempt rather than marking it failed. Add tests and commit this controller-only recovery. Preserve C480 as attempt 1.
8. Restart `scripts/run_adaptive_fixed_campaign.py`; it must accept C480 and continue the already-triggered D1 and every later conditional branch exactly according to `GOAL.md`.

Do not resume C480 directly at the newer main commit: the runner's immutable resume contract includes the original Git revision. The relevant inference source hashes are preserved at the old commit.

## Official-source branch status

Official source checkouts and locks exist under the campaign directory and were clean at handoff:

- Nüwa commit `175b7d57a7890b9818dd26d82f2d91d9868b0d80`
- Reroute commit `9d422042076de678b2ad67c4a6660e4b97971a1e`
- `official_sources/source_lock.json` SHA-256 `5cc6aabc3190a6437eb506965c4cdaacd5147e9538e174b012a53acb98c77b68`
- `official_sources/compatibility_audit.json` SHA-256 `ec618703d3a8dc1f2038f463a861d309aad65f809d41bcdbdc13cb96428093cf`

The source audit established:

- official Nüwa is `SKIPPED_OFFICIAL_INCOMPATIBLE` after three documented bounded checks: no published average-192 HF operating point, missing legacy `llava` dependency/API, and a monkeypatch targeting the incompatible native LLaVA class tree. Do not approximate it.
- conditional Nüwa+Reroute is skipped because the official Reroute release has no direct Nüwa implementation. Do not create a hybrid.
- official PDrop and PDrop+Reroute have pinned published near-192 configs and a committed thin source-native adapter.
- PDrop realized average: `191.34375` tokens; PDrop+Reroute realized average: `191.46875`. Label these realized values, not exact budget-matched 192.

The first official unpruned parity invocation exposed a local provenance-key bug before loading the model or writing predictions. It was fixed in commit `fcd397e`; no official quality cell has run yet. The `official_runs/parity_full_20` directory may exist but had no scientific files. Run the mandatory 20-example unpruned canonical parity gate, then bounded method smokes, before accepting any 1,000-row official run. Compare the unpruned outputs against the canonical existing full predictions mapped by frozen sample ID from:

`experiments/progressive_oracle_refcoco_val_20260814/preflight/existing_full/existing_full/per_sample_results.jsonl`

Use an isolated third-party environment if dependencies require it, without changing the canonical environment. If official source fails the three registered bounded attempts, preserve evidence and skip cleanly rather than reimplementing the algorithm.

## Work still required after fixed and official branches

Continue all remaining phases in `GOAL.md`, not merely C480:

1. Finish the adaptive fixed-method state machine and freeze the safe/aggressive routes.
2. Complete or cleanly skip each official cell under the source-native acceptance gate.
3. Create a deterministic grouped 3,000-record RefCOCO **train** manifest and registered 2,160 fit / 240 early-stop / 600 calibration split with leakage proof against validation.
4. Generate paired safe/aggressive physical route labels and inference-available K2 features only.
5. Implement, preregister, train, and evaluate exactly one ≤1M-parameter MLP for seeds 0/1/2; calibration uses only the 600 records and the frozen constraints.
6. Run three physical dynamic validation seeds and assemble the matched-rate random and privileged oracle controls from physical source rows.
7. Select the project candidate and report official methods separately.
8. Run matched serial timing only if a deployable project candidate passes the registered K12-like gate; otherwise record `SKIPPED_NO_QUALITY_CANDIDATE`.
9. Independently validate every artefact, update the single `RESEARCH_REPORT.md` and machine-readable registry, run the final audit, and stop at the exact GOAL.md stopping condition.

## Restart verification checklist

Do not trust this handoff alone. At startup verify:

- `git status`, branch, history, and user-owned untracked file
- `GOAL.md` and frozen-manifest hashes above
- no stale experiment OS/CUDA processes
- C448 with `scripts/validate_progressive_oracle_run.py`
- all 586 C480 partial rows, order, uniqueness, JSON completeness, and config hashes
- controller state/registry/report readability and hashes
- official checkout commits, cleanliness, and recursive source lock
- the canonical Python environment and model checkpoint availability
- enough persistent and root storage for the remaining physical runs, train images/labels, and optional timing

Work autonomously after these checks. Preserve failures and report them honestly. Do not claim the goal complete until every required branch is accepted or validly skipped and the cumulative final audit passes.
