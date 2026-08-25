# Installing the RunPod handoff

These files are templates. They do not supersede the current local root goal merely by existing in
this directory.

On the RunPod clone, first preserve any existing root instructions, then deliberately install this
campaign specification:

```bash
mkdir -p experiments/ivc_prune_refcoco_val_20260825/pre_goal_snapshot
cp AGENTS.md GOAL.md experiments/ivc_prune_refcoco_val_20260825/pre_goal_snapshot/ 2>/dev/null || true
cp runpod_handoff/ivc_prune/AGENTS.md ./AGENTS.md
cp runpod_handoff/ivc_prune/GOAL.md ./GOAL.md
```

Then start a fresh Codex turn with: `Read AGENTS.md and GOAL.md completely, then execute GOAL.md
until its terminal completion condition. Do not push.`
