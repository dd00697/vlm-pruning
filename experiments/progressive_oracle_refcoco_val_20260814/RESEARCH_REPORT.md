# Progressive Visual-Token Pruning and Oracle Feasibility Campaign

## Scope and protocol

This is an exploratory RefCOCO-validation campaign. Every quality cell uses the
same frozen 1,000-example validation subset. No RefCOCO testA/testB, transfer
dataset, new 8,811-example confirmation, learned router, or dissertation edit is
part of this campaign.

## Frozen discovery subset

- Rows: 1,000
- Manifest SHA-256: `98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd`
- Sampling rule and provenance: `manifests/discovery_manifest_lock.json`

## Historical reference

`REF_K20_B48` was independently recomputed from the complete physical historical
prediction JSONL, rather than trusted from its aggregate metrics file.

| Reference | Rows | Acc@0.5 | mIoU | Parse rate |
|---|---:|---:|---:|---:|
| REF_K20_B48 (frozen subset) | 1,000 | 0.4390 | 0.4328 | 1.0000 |

The historical run has no `COMPLETED.json` sentinel; it is accepted as read-only
reference evidence only because this campaign independently validated its exact
physical JSONL, configuration, source hashes, row order, budgets, caches, parser,
and metrics. Its redundant partial file is not used.

## Implementation and preflight evidence

The generalized path physically prunes an ordered sequence of boundaries,
recomputes query relevance from each boundary's current hidden states, preserves
original raster order and gap-aware position IDs, and maintains one validated
dynamic-cache bank per decoder-layer segment. B=0 removes every visual state
without a dummy replacement. Oracle restarts accept a strict integer-index
artifact only and reconstruct fresh projector embeddings.

All 147 CPU tests passed. On two fixed real-model examples, the generalized
unpruned, K12/B48, and K20/B48 paths exactly matched the current canonical paths
in output text, boxes, IoU, and (for pruned cells) retained indices. B=0,
two-stage, three-stage, oracle-restart, and index-only oracle-replay smokes all
passed independent artifact validation. Replaying stored oracle indices produced
identical rankings, selections, text, boxes, and IoU without a teacher state.

The maximum measured reservation was 14,040 MiB. Two jobs plus 10% headroom
require 30,888 MiB and passed an exact serial-versus-concurrent semantic soak.
Three would require 46,332 MiB, exceeding the measured 45,489 MiB free, so the
quality controller is capped at two. Oracle execution remains serial unless a
separate oracle two-job soak passes.

## Campaign 1 — K20 extreme tail

Pending.

## Campaign 2 — progressive pruning

Pending.

## Campaign 3 — oracle diagnostics

Pending. Oracle candidate-average token figures will exclude the dense teacher
pass and will report teacher visual-token layer-work separately.

## Conditional exploratory timing

Pending registered gates.

## Deferred full-validation recommendation

Not run. Candidate selection is pending discovery outcomes.

## Limitations

These are validation-only exploratory results on 1,000 examples. One percentage
point of Acc@0.5 is only ten examples. Marginal differences are inconclusive and
all dissertation claims require separately authorized full-validation evidence.

## Primary recommendation

Pending completion of all accepted discovery evidence.
