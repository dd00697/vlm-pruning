# Progressive Visual-Token Pruning and Oracle Feasibility Campaign

## Scope and protocol

This is an exploratory RefCOCO-validation campaign. All quality cells use the same frozen 1,000-example validation subset. No testA/testB, transfer dataset, new 8,811-example confirmation, learned router, or dissertation edit was run.

## Frozen discovery subset and historical reference

- Discovery rows: 1,000
- Manifest SHA-256: `98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd`
- REF_K20_B48: Acc@0.5=0.4390, mIoU=0.4328, parse=1.0000

The reference was independently rescored from the complete physical historical JSONL. Its absent historical completion sentinel and redundant partial are disclosed; neither historical aggregate metadata nor the partial was trusted.

## Implementation and preflight evidence

All 147 preregistered preflight CPU tests passed. Real-model empty-schedule, K12/B48, and K20/B48 paths matched the current canonical implementations exactly. B=0, two-stage, three-stage, oracle restart, and strict index-only replay passed. Two-way quality concurrency passed exact semantic parity with 10% measured memory headroom; three-way concurrency was rejected by capacity. Oracle remained serial.

Two preflight-only attempts failed before scientific inference (wrong interpreter; inert helper probe configuration). They were preserved and did not become quality cells.

## Campaign 1 — K20 extreme tail

| Cell | Configuration | Status | Acc@0.5 | mIoU | Parse | Candidate avg tokens | Teacher token-layer work |
|---|---|---|---:|---:|---:|---:|---:|
| T1 | K20->32 query_topk | accepted | 0.4640 | 0.4421 | 1.0000 | 372.000 | 0 |
| T2 | K20->16 query_topk | accepted | 0.4650 | 0.4432 | 1.0000 | 366.000 | 0 |
| T3 | K20->8 query_topk | accepted | 0.4580 | 0.4414 | 1.0000 | 363.000 | 0 |
| T4 | K20->4 query_topk | accepted | 0.4520 | 0.4413 | 1.0000 | 361.500 | 0 |
| T5 | K20->0 none | accepted | 0.4520 | 0.4405 | 1.0000 | 360.000 | 0 |
| R16-S0 | K20->16 random | accepted | 0.4600 | 0.4396 | 1.0000 | 366.000 | 0 |
| R16-S1 | K20->16 random | accepted | 0.4630 | 0.4383 | 1.0000 | 366.000 | 0 |
| R16-S2 | K20->16 random | accepted | 0.4480 | 0.4374 | 1.0000 | 366.000 | 0 |
| R8-S0 | K20->8 random | accepted | 0.4570 | 0.4400 | 1.0000 | 363.000 | 0 |
| R8-S1 | K20->8 random | accepted | 0.4550 | 0.4414 | 1.0000 | 363.000 | 0 |
| R8-S2 | K20->8 random | accepted | 0.4570 | 0.4394 | 1.0000 | 363.000 | 0 |
| R4-S0 | K20->4 random | accepted | 0.4570 | 0.4401 | 1.0000 | 361.500 | 0 |
| R4-S1 | K20->4 random | accepted | 0.4600 | 0.4399 | 1.0000 | 361.500 | 0 |
| R4-S2 | K20->4 random | accepted | 0.4580 | 0.4377 | 1.0000 | 361.500 | 0 |

The provisional discovery `B_tail` is **0** under the registered 1.0-point Acc@0.5 and 0.005 mIoU development tolerances. This is not an equivalence claim and requires 8,811-row confirmation.

Relative to REF_K20_B48, T5 (B=0) changes Acc@0.5 by +1.3 percentage points and mIoU by +0.0077 on this subset. The rule therefore selects zero, but its 1,000-row uncertainty makes this a breaking-point hypothesis rather than a definitive threshold.

Dense-prefix query-top-k versus random behavior:

| B | Query Acc | Random Acc mean (seed range) | Query mIoU | Random mIoU mean (seed range) |
|---:|---:|---:|---:|---:|
| 16 | 0.4650 | 0.4570 (0.4480-0.4630) | 0.4432 | 0.4385 (0.4374-0.4396) |
| 8 | 0.4580 | 0.4563 (0.4550-0.4570) | 0.4414 | 0.4403 (0.4394-0.4414) |
| 4 | 0.4520 | 0.4583 (0.4570-0.4600) | 0.4413 | 0.4392 (0.4377-0.4401) |

Query-top-k is not consistently above the random seed mean at B=16/8/4; the paired intervals below determine whether any observed difference is distinguishable from zero. Random seed spread is retained rather than hidden behind the mean.

Registered paired contrasts (10,000 resamples, seed 20260814):

| Contrast (left minus right) | Metric | Delta | 95% paired bootstrap CI | Unit |
|---|---|---:|---:|---|
| T1-minus-REF_K20_B48 | acc_at_0_5 | 2.5000 | [0.8000, 4.2000] | percentage_points |
| T1-minus-REF_K20_B48 | mean_iou | 0.0093 | [0.0036, 0.0150] | 0_to_1 |
| T2-minus-REF_K20_B48 | acc_at_0_5 | 2.6000 | [0.8000, 4.4000] | percentage_points |
| T2-minus-REF_K20_B48 | mean_iou | 0.0104 | [0.0047, 0.0161] | 0_to_1 |
| T3-minus-REF_K20_B48 | acc_at_0_5 | 1.9000 | [0.1000, 3.6000] | percentage_points |
| T3-minus-REF_K20_B48 | mean_iou | 0.0086 | [0.0024, 0.0147] | 0_to_1 |
| T4-minus-REF_K20_B48 | acc_at_0_5 | 1.3000 | [-0.4000, 3.1000] | percentage_points |
| T4-minus-REF_K20_B48 | mean_iou | 0.0085 | [0.0028, 0.0144] | 0_to_1 |
| T5-minus-REF_K20_B48 | acc_at_0_5 | 1.3000 | [-0.5000, 3.1000] | percentage_points |
| T5-minus-REF_K20_B48 | mean_iou | 0.0077 | [0.0019, 0.0137] | 0_to_1 |
| query_topk-minus-random_seed_mean_B16 | acc_at_0_5 | 0.8000 | [-0.2333, 1.8333] | percentage_points |
| query_topk-minus-random_seed_mean_B16 | mean_iou | 0.0047 | [0.0018, 0.0078] | 0_to_1 |
| query_topk-minus-random_seed_mean_B8 | acc_at_0_5 | 0.1667 | [-0.8333, 1.2000] | percentage_points |
| query_topk-minus-random_seed_mean_B8 | mean_iou | 0.0011 | [-0.0019, 0.0043] | 0_to_1 |
| query_topk-minus-random_seed_mean_B4 | acc_at_0_5 | -0.6333 | [-1.7000, 0.4000] | percentage_points |
| query_topk-minus-random_seed_mean_B4 | mean_iou | 0.0021 | [-0.0008, 0.0053] | 0_to_1 |

## Campaign 2 — progressive pruning

| Cell | Configuration | Status | Acc@0.5 | mIoU | Parse | Candidate avg tokens | Teacher token-layer work |
|---|---|---|---:|---:|---:|---:|---:|
| P1 | K8->256 query_topk, K12->48 query_topk | accepted | 0.3910 | 0.4105 | 1.0000 | 206.000 | 0 |
| P2 | K8->192 query_topk, K12->48 query_topk | accepted | 0.3770 | 0.4038 | 1.0000 | 198.000 | 0 |
| P3 | K4->384 query_topk, K12->48 query_topk | accepted | 0.4080 | 0.4247 | 1.0000 | 198.000 | 0 |
| P4 | K4->256 query_topk, K12->48 query_topk | accepted | 0.3820 | 0.4081 | 1.0000 | 166.000 | 0 |
| P5 | K2->384 query_topk, K12->48 query_topk | accepted | 0.4060 | 0.4198 | 1.0000 | 186.000 | 0 |
| P6 | K2->288 query_topk, K12->48 query_topk | accepted | 0.3680 | 0.4045 | 1.0000 | 156.000 | 0 |
| P7 | K2->384 query_topk, K12->96 query_topk, K20->0 none | accepted | 0.3900 | 0.4122 | 1.0000 | 180.000 | 0 |
| P8 | K4->384 query_topk, K12->96 query_topk, K20->0 none | accepted | 0.4010 | 0.4179 | 1.0000 | 192.000 | 0 |
| P9 | K4->384 query_topk, K12->96 query_topk, K20->4 query_topk | accepted | 0.4000 | 0.4160 | 1.0000 | 193.500 | 0 |

The registered base/refinement selection chose **P8** before P9. The full decision traces, Pareto inputs, and paired contrasts are in `decisions/` and `analysis/campaign2.json`.

The non-dominated Acc@0.5/average-token frontier among P1-P9 is **P6, P4, P7, P5, P3** (in increasing token order). P3 has the highest progressive accuracy (0.4080); P5 is the registered near-best efficiency choice because it is only 0.2 points lower at 186 rather than 198 average tokens. Both remain clearly below REF_K20_B48.

P8 versus P9 is the preregistered zero-tail versus four-token-tail ablation because `B_tail=0`; it is not described as a selector comparison. P8 leads by only 0.1 point Acc@0.5 and its interval includes zero.

| Contrast (left minus right) | Metric | Delta | 95% paired bootstrap CI | Unit |
|---|---|---:|---:|---|
| P7-minus-P8 | acc_at_0_5 | -1.1000 | [-3.2000, 1.0000] | percentage_points |
| P7-minus-P8 | mean_iou | -0.0057 | [-0.0137, 0.0020] | 0_to_1 |
| P8-minus-P9 | acc_at_0_5 | 0.1000 | [-1.1000, 1.3000] | percentage_points |
| P8-minus-P9 | mean_iou | 0.0019 | [-0.0013, 0.0054] | 0_to_1 |
| P8-minus-T5 | acc_at_0_5 | -5.1000 | [-7.8000, -2.3000] | percentage_points |
| P8-minus-T5 | mean_iou | -0.0226 | [-0.0327, -0.0125] | 0_to_1 |

## Campaign 3 — oracle diagnostics

| Cell | Configuration | Status | Acc@0.5 | mIoU | Parse | Candidate avg tokens | Teacher token-layer work |
|---|---|---|---:|---:|---:|---:|---:|
| O1 | pre-LLM oracle->64 | accepted | 0.2630 | 0.3513 | 1.0000 | 64.000 | 6912 |
| O2 | pre-LLM oracle->96 | accepted | 0.2980 | 0.3666 | 1.0000 | 96.000 | 6912 |
| O3 | pre-LLM oracle->128 | accepted | 0.3140 | 0.3836 | 1.0000 | 128.000 | 6912 |
| O4 | pre-LLM oracle->64 | accepted | 0.2910 | 0.3517 | 1.0000 | 64.000 | 11520 |
| O5 | pre-LLM oracle->96 | accepted | 0.3500 | 0.3815 | 1.0000 | 96.000 | 11520 |
| O6 | pre-LLM oracle->128 | accepted | 0.3720 | 0.4018 | 1.0000 | 128.000 | 11520 |
| O7 | pre-LLM oracle->102; K20->0 none | accepted | 0.3000 | 0.3680 | 1.0000 | 63.750 | 6912 |
| O8 | pre-LLM oracle->102; K20->0 none | accepted | 0.3510 | 0.3881 | 1.0000 | 63.750 | 11520 |
| O9 | pre-LLM random->64 | accepted | 0.1680 | 0.2573 | 1.0000 | 64.000 | 0 |
| O10 | pre-LLM uniform->64 | accepted | 0.1980 | 0.2818 | 1.0000 | 64.000 | 0 |

Oracle candidate-average token figures describe only the freshly restarted counterfactual candidate. The dense privileged teacher is not efficient, is not deployable, and its 576×K visual-token layer-work is reported separately.

The K12-teacher capacity curve is 0.2630/0.3513, 0.2980/0.3666, and 0.3140/0.3836 Acc@0.5/mIoU at 64/96/128 patches. The K20-teacher curve is stronger—0.2910/0.3517, 0.3500/0.3815, and 0.3720/0.4018—but even privileged K20 oracle-128 is 6.7 accuracy points and 0.0310 mIoU below REF_K20_B48.

O7/O8 use the preregistered dynamic `B_early64=102` calculation and realize exactly 63.75 candidate-average tokens. O8 is the strongest ≤64-average oracle at 0.3510 Acc@0.5 and 0.3881 mIoU, still 8.8 points and 0.0447 mIoU below the reference. Oracle-64 nevertheless clearly exceeds random-64 (0.1680/0.2573) and uniform-64 (0.1980/0.2818), so patch identity helps but is not sufficient to replace early dense processing.

Oracle computations are privileged diagnostics: candidate-average figures exclude the discarded dense teacher pass, while teacher work is separately shown as 6,912 (K12) or 11,520 (K20) visual-token layers.

| Contrast (left minus right) | Metric | Delta | 95% paired bootstrap CI | Unit |
|---|---|---:|---:|---|
| O1-minus-O9 | acc_at_0_5 | 9.5000 | [6.4000, 12.6000] | percentage_points |
| O1-minus-O9 | mean_iou | 0.0939 | [0.0788, 0.1096] | 0_to_1 |
| O1-minus-O10 | acc_at_0_5 | 6.5000 | [3.3000, 9.6000] | percentage_points |
| O1-minus-O10 | mean_iou | 0.0695 | [0.0542, 0.0843] | 0_to_1 |
| O4-minus-O9 | acc_at_0_5 | 12.3000 | [9.1000, 15.4000] | percentage_points |
| O4-minus-O9 | mean_iou | 0.0944 | [0.0793, 0.1093] | 0_to_1 |
| O4-minus-O10 | acc_at_0_5 | 9.3000 | [6.1000, 12.5000] | percentage_points |
| O4-minus-O10 | mean_iou | 0.0699 | [0.0547, 0.0853] | 0_to_1 |
| O2-minus-O1 | acc_at_0_5 | 3.5000 | [1.2000, 5.8000] | percentage_points |
| O2-minus-O1 | mean_iou | 0.0153 | [0.0072, 0.0237] | 0_to_1 |
| O3-minus-O2 | acc_at_0_5 | 1.6000 | [-0.2000, 3.5000] | percentage_points |
| O3-minus-O2 | mean_iou | 0.0170 | [0.0095, 0.0247] | 0_to_1 |
| O5-minus-O4 | acc_at_0_5 | 5.9000 | [3.6000, 8.2000] | percentage_points |
| O5-minus-O4 | mean_iou | 0.0298 | [0.0216, 0.0376] | 0_to_1 |
| O6-minus-O5 | acc_at_0_5 | 2.2000 | [0.2000, 4.3000] | percentage_points |
| O6-minus-O5 | mean_iou | 0.0203 | [0.0131, 0.0278] | 0_to_1 |
| O7-minus-O9 | acc_at_0_5 | 13.2000 | [10.2000, 16.2000] | percentage_points |
| O7-minus-O9 | mean_iou | 0.1106 | [0.0956, 0.1263] | 0_to_1 |
| O8-minus-O10 | acc_at_0_5 | 15.3000 | [11.9000, 18.7000] | percentage_points |
| O8-minus-O10 | mean_iou | 0.1063 | [0.0912, 0.1217] | 0_to_1 |

## Rejected attempts and artifact validity

All 33 scientific cells were accepted on their first attempt. Every accepted cell has exactly 1,000 ordered physical predictions, a strict source/config epoch, independently recomputed raw-output metrics, validated stage indices/positions/cache lengths, and a hash-bearing completion sentinel. No partial was spliced or reconstructed.

The two rejected preflight-only attempts occurred before scientific inference: the default interpreter lacked `transformers`, and an unused inherited helper probe configuration was rejected. Both are recorded in `preflight/PREFLIGHT_REPORT.json`; neither counts as a scientific cell.

## Conditional exploratory timing

### E1: accepted

Candidate: **T5**; paired rows: 1000; 20 warm-up examples per path; alternating within-pair order; matched full reference measured in the same process.

| Quantity | Full mean | Candidate mean | Candidate/full or reduction |
|---|---:|---:|---:|
| Vision latency (s) | 0.010720 | 0.010405 | 0.9706× |
| Projector latency (s) | 0.000355 | 0.000369 | 1.0384× |
| LLM prefill latency (s) | 0.149224 | 0.126458 | 0.8474× |
| Selector latency (s) | 0.000000 | 0.059788 | — |
| Decode latency (s) | 0.693883 | 0.687900 | 0.9914× |
| End-to-end latency (s) | 0.868920 | 0.839877 | 0.9666× |
| Generated tokens | 23.469000 | 23.458000 | 0.9995× |
| Prompt KV bytes | 349390766.080000 | 236138790.912000 | 0.6759× |
| Final KV bytes | 349390766.080000 | 236138790.912000 | 0.6759× |
| Average visual tokens | 576.000000 | 360.000000 | 0.6250× |
| Peak allocated GPU bytes | 14552203264 | 14420242944 | — |
| Peak reserved GPU bytes | 14755561472 | 14755561472 | — |

Prompt/final KV storage is also preserved by decoder-layer segment in the physical timing JSONL. These are exploratory one-pass measurements, not the dissertation's five-repetition final timing protocol.

### E2: skipped_by_design

Candidate: **P5**. Skipped by design: Registered progressive candidate exceeds the Acc@0.5 and mIoU loss tolerances.

Registered gate evidence: `{"acc_at_0_5_loss": 0.032999999999999974, "average_visual_tokens": 186.0, "candidate_cell_id": "P5", "mean_iou_loss": 0.013041949388377938, "parsing_ok": true, "passed": false, "thresholds": {"acc_loss_lte": 0.02, "average_visual_tokens_lt": 246, "miou_loss_lte": 0.01}, "validation_ok": true}`

## Deferred full-validation recommendation

Strongest deployable candidate: **T2**. Strongest oracle candidate at ≤64 average tokens: **O8**.

The future 8,811-row recommendation is T2 (K20→16 query top-k) with R16-S0 (K20→16 random, seed 0) as its single selector comparator. Both are frozen in the registry as **not run**. No F1-F4, testA/testB, or other full-validation job was launched; a new human-authorized goal is required.

## Limitations

These are validation-only exploratory results on 1,000 examples; one percentage point of accuracy is only ten examples. Marginal 0.5–1.0-point movements are inconclusive. Oracle results use privileged future information. Any dissertation claim or training decision should first consider the deferred full-validation confirmation.

## Final validation and completion audit

The final clean-worktree CPU suite passed all 160 tests with exit code 0. Command: `/workspace/.followup_venv/bin/python -m unittest discover -s tests -v`. Its captured output SHA-256 is `fdcceb6ceb5649d87310411426af6b415ccd7ad8c6d841010dc7b37832678835`.

The machine-readable completion audit independently revalidates every accepted physical quality and timing artifact and records requirement-level evidence in `FINAL_COMPLETION_REPORT.json`.

An initial audit command omitted the explicit `tests/` discovery root and therefore discovered zero tests. It was rejected, preserved in the registry, and replaced by the clean-worktree command above; zero-test discovery is never accepted as test evidence.

## Primary recommendation

**C. Even oracle early selection is weak, so selector prediction alone is insufficient and the next work should investigate information transfer, token recovery or another mechanism.**

The strongest deployable cell (T2) is accurate but averages 366.0 visual tokens, while the strongest ≤64-average oracle (O8) reaches only 0.3510 Acc@0.5 and 0.3881 mIoU. Even oracle-128 remains well below the reference. The evidence therefore points to early-layer information transfer/mixing—not merely imperfect patch ranking—as the limiting factor.

This is a preliminary exploratory recommendation. Do not begin training without human review; any dissertation claim should first consider the deferred 8,811-row confirmation.
