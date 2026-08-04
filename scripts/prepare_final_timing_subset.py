#!/usr/bin/env python
"""Create the registered 1,000-example RefCOCO testA timing subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path("data/manifests/final_evaluation/refcoco_testA_unique.jsonl")
OUTPUT = Path("data/manifests/final_evaluation/refcoco_testA_timing1000_seed20260729.jsonl")
SEED = 20260729
COUNT = 1000


def main() -> int:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    rng = np.random.default_rng(SEED)
    selected = [rows[int(index)] for index in rng.choice(len(rows), size=COUNT, replace=False)]
    selected.sort(key=lambda row: str(row["sample_id"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(json.dumps({"path": str(OUTPUT), "rows": COUNT, "seed": SEED, "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
