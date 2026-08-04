#!/usr/bin/env python
"""Run one serial, warmed-up trial from the registered efficiency campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for path in (str(ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from run_baseline import prepare_compacted_inputs  # noqa: E402
from run_official_baseline import (  # noqa: E402
    generate_answer_true_removal_position_mode,
    generate_fastv_answer,
    generate_visionzip_answer,
    load_manifest,
    load_model,
    make_prompt,
)
from run_ug_prune_single_pass import (  # noqa: E402
    generate_mid_layer_answer,
    query_text_for_sample,
)
from vlm_pruning.features import resolve_query_token_positions  # noqa: E402
from vlm_pruning.llava_hooks import MidLayerUGPruneForwarder, ug_gap58_config_for_prune_layer  # noqa: E402
from vlm_pruning.selectors import selector_for_method  # noqa: E402


METHODS = ("full", "k12", "final", "fastv", "visionzip")
MANIFEST = Path("data/manifests/final_evaluation/refcoco_testA_timing1000_seed20260729.jsonl")
WARMUP = 20


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def role_ratios(role: str) -> tuple[float, float, float]:
    return {
        "more_context": (0.45, 0.40, 0.15),
        "more_anchors": (0.45, 0.25, 0.30),
    }[role]


class Trial:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.torch, self.processor, self.model, self.device, self.model_env = load_model(
            args.model_path, "bf16"
        )
        self.full_selector = selector_for_method("full")
        self.forwarder = None
        self.query_scoring = None
        self.role = None
        if args.method in {"k12", "final"}:
            if args.method == "k12":
                layer = 12
                self.query_scoring = "query_max"
                self.role = "more_context"
            else:
                frozen = json.loads(
                    (ROOT / "experiments/final_evaluation/frozen_method.json").read_text(
                        encoding="utf-8"
                    )
                )
                layer = 20
                self.query_scoring = frozen["method"]["query_scoring"]
                self.role = frozen["method"]["role_allocation"]
            target, context, anchor = role_ratios(self.role)
            config = ug_gap58_config_for_prune_layer(
                layer,
                map_source=self.query_scoring,
                fusion_mode="single_layer",
                probe_layers=(layer,),
                position_mode="gap_preserving",
                target_ratio=target,
                context_ratio=context,
                anchor_ratio=anchor,
            )
            self.forwarder = MidLayerUGPruneForwarder(
                self.torch,
                self.model,
                image_token_id=int(getattr(self.model.config, "image_token_index", 32000)),
                grid=24,
                prune_after_layer=layer,
                target_visual_tokens=48,
                selector_config=config,
                diagnostics_mode="benchmark",
                synchronise_subtimings=True,
            )

    def infer(self, sample: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        image = Image.open(Path(sample["image_path"])).convert("RGB")
        prompt_args = argparse.Namespace(
            task=self.args.task,
            dataset=self.args.dataset,
            box_coordinate_mode="normalized_0_1",
            # Timing uses the registered evaluation prompt.  This attribute
            # became explicit when paper-verbatim MME reproduction was added
            # to the quality runner, so keep it explicit here as well.
            prompt_profile="registered",
        )
        prompt = make_prompt(prompt_args, sample)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        if self.args.method == "full":
            selection = self.full_selector.select(
                sample_id=str(sample["sample_id"]),
                budget=576,
                original_tokens=576,
                grid=24,
                seed=0,
            )
            raw, generated, _reported, diagnostics = generate_answer_true_removal_position_mode(
                torch,
                self.processor,
                self.model,
                self.device,
                image,
                prompt,
                self.args.max_new_tokens,
                selection.selected_indices,
                576,
                position_mode="gap_preserving",
            )
            actual = 576
            selector_inside_prefill = False
        elif self.args.method == "fastv":
            raw, generated, _reported, diagnostics, selection = generate_fastv_answer(
                torch,
                self.processor,
                self.model,
                self.device,
                image,
                prompt,
                self.args.max_new_tokens,
                576,
                48,
                synchronise_subtimings=True,
            )
            actual = 48
            selector_inside_prefill = True
        elif self.args.method == "visionzip":
            raw, generated, _reported, diagnostics, selection = generate_visionzip_answer(
                torch,
                self.processor,
                self.model,
                self.device,
                image,
                prompt,
                self.args.max_new_tokens,
                576,
                48,
            )
            actual = 48
            selector_inside_prefill = False
        else:
            assert self.forwarder is not None
            full_inputs, prep = prepare_compacted_inputs(
                torch,
                self.processor,
                self.model,
                self.device,
                image,
                prompt,
                list(range(576)),
                576,
            )
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            text_prompt = (
                self.processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                if hasattr(self.processor, "apply_chat_template")
                else prompt
            )
            query = query_text_for_sample(prompt_args, sample)
            resolution = resolve_query_token_positions(
                tokenizer,
                text_prompt,
                query,
                input_ids=full_inputs["input_ids"][0].detach().cpu().tolist(),
                image_token_id=int(getattr(self.model.config, "image_token_index", 32000)),
            )
            if not resolution.token_positions:
                raise RuntimeError(f"Query-token resolution failed for {sample['sample_id']}.")
            raw, generated, _reported, selection, diagnostics = generate_mid_layer_answer(
                torch,
                self.processor,
                self.model,
                full_inputs=full_inputs,
                prep_diag=prep,
                sample_id=str(sample["sample_id"]),
                query_text=query,
                query_token_status=resolution.resolver_status,
                query_token_positions=resolution.token_positions,
                seed=0,
                max_new_tokens=self.args.max_new_tokens,
                original_tokens=576,
                forwarder=self.forwarder,
                decode_mode="layer_aware_cache",
            )
            actual = 48
            selector_inside_prefill = True
        torch.cuda.synchronize()
        end_to_end = time.perf_counter() - start
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        if len(selection.selected_indices) != actual:
            raise RuntimeError(f"{self.args.method}: selected {len(selection.selected_indices)}, expected {actual}.")

        vision = float(diagnostics.get("vision_encoder_latency_s") or 0.0)
        projector = float(diagnostics.get("projector_latency_s") or 0.0)
        selector = float(diagnostics.get("pruning_selection_latency_s") or 0.0)
        prefill = float(diagnostics.get("llm_prefill_latency_s") or 0.0)
        decode = float(diagnostics.get("decode_latency_s") or 0.0)
        reconciled = vision + projector + prefill + decode
        if not selector_inside_prefill:
            reconciled += selector
        return {
            "sample_id": str(sample["sample_id"]),
            "method": self.args.method,
            "repetition": self.args.repetition,
            "order_position": self.args.order_position,
            "raw_model_output": raw,
            "requested_visual_tokens": actual,
            "actual_visual_tokens": actual,
            "generated_token_count": generated,
            "vision_encoder_latency_s": vision,
            "projector_latency_s": projector,
            "selector_latency_s": selector,
            "selector_inside_prefill": selector_inside_prefill,
            "llm_prefill_latency_s": prefill,
            "decode_latency_s": decode,
            "decode_time_per_token_s": decode / generated if generated else None,
            "end_to_end_latency_s": end_to_end,
            "reconciled_component_latency_s": reconciled,
            "host_overhead_remainder_s": end_to_end - reconciled,
            "peak_allocated_gpu_bytes": peak_allocated,
            "peak_reserved_gpu_bytes": peak_reserved,
            "prompt_kv_cache_bytes": diagnostics.get("cache/prompt_total_bytes"),
            "final_kv_cache_bytes": diagnostics.get("cache/final_total_bytes"),
            "prompt_early_bank_bytes": diagnostics.get("cache/prompt_early_bank_bytes"),
            "prompt_late_bank_bytes": diagnostics.get("cache/prompt_late_bank_bytes"),
            "final_early_bank_bytes": diagnostics.get("cache/final_early_bank_bytes"),
            "final_late_bank_bytes": diagnostics.get("cache/final_late_bank_bytes"),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--repetition", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--order-position", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--model-path", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dataset", default="refcoco")
    parser.add_argument("--task", choices=["grounding", "vqa"], default="grounding")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/final_evaluation/timing"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--warmup-examples", type=int, default=WARMUP)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=os.environ.get("WANDB_MODE", "offline"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "vlm-pruning"))
    parser.add_argument("--wandb-group", default="dissertation-final-efficiency-20260729")
    args = parser.parse_args()
    frozen_path = ROOT / "experiments/final_evaluation/frozen_method.json"
    protocol_path = ROOT / "experiments/final_evaluation/protocol/protocol.json"
    if not frozen_path.exists() or not protocol_path.exists():
        raise SystemExit("Timing requires the registered frozen method and protocol.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if (
        hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        != frozen.get("registered_protocol_sha256")
    ):
        raise SystemExit("Timing refused: frozen method and registered protocol hashes differ.")
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be in [1, 1000].")
    if not 0 <= args.warmup_examples <= 1000:
        raise SystemExit("--warmup-examples must be in [0, 1000].")
    run_dir = args.output_root / f"rep{args.repetition}_pos{args.order_position}_{args.method}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        provenance_path = run_dir / "wandb_run.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.exists()
            else {}
        )
        provenance_complete = (
            args.wandb_mode == "disabled"
            or (
                bool(provenance.get("enabled"))
                and provenance.get("mode") in {"online", "offline"}
                and bool(provenance.get("id"))
                and (
                    provenance.get("mode") == "online"
                    or bool(list((run_dir / "wandb").glob("offline-run-*")))
                )
            )
        )
        if (
            int(existing.get("samples", -1)) == args.limit
            and provenance_complete
        ):
            print(f"Skipping complete timing trial {run_dir}")
            return 0
    rows = load_manifest(args.manifest, None)
    if len(rows) != 1000:
        raise SystemExit(f"Timing manifest has {len(rows)} rows, expected 1000.")
    measured_rows = rows[: args.limit]
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    trial = Trial(args)
    write_json(
        run_dir / "run_environment.json",
        {
            **trial.model_env,
            "python": sys.version,
            "torch": trial.torch.__version__,
            "cuda": trial.torch.version.cuda,
            "wandb_mode": args.wandb_mode,
        },
    )
    for sample in rows[: args.warmup_examples]:
        trial.infer(sample)
    trial.torch.cuda.synchronize()
    trial.torch.cuda.reset_peak_memory_stats()
    measured: list[dict[str, Any]] = []
    output = run_dir / "per_sample_timing.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(measured_rows, start=1):
            result = trial.infer(sample)
            measured.append(result)
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            if index % 25 == 0:
                handle.flush()
                print(f"Timed {index}/{len(measured_rows)}")
    fields = (
        "vision_encoder_latency_s",
        "projector_latency_s",
        "selector_latency_s",
        "llm_prefill_latency_s",
        "decode_latency_s",
        "end_to_end_latency_s",
        "host_overhead_remainder_s",
        "generated_token_count",
        "prompt_kv_cache_bytes",
        "final_kv_cache_bytes",
    )
    metrics: dict[str, Any] = {
        "method": args.method,
        "repetition": args.repetition,
        "order_position": args.order_position,
        "samples": len(measured),
        "warmup_examples": args.warmup_examples,
        "peak_allocated_gpu_bytes": max(
            int(row["peak_allocated_gpu_bytes"]) for row in measured
        ),
        "peak_reserved_gpu_bytes": max(
            int(row["peak_reserved_gpu_bytes"]) for row in measured
        ),
    }
    for field in fields:
        values = [float(row[field]) for row in measured if row.get(field) is not None]
        metrics[f"{field}/mean"] = statistics.fmean(values) if values else None
        metrics[f"{field}/sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
        metrics[f"{field}/median"] = statistics.median(values) if values else None
        metrics[f"{field}/p95"] = percentile(values, 0.95) if values else None
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    if args.method == "k12":
        method_configuration: Any = {
            "prune_after_layer": 12,
            "tokens": 48,
            "query_scoring": "query_max",
            "role_allocation": "more_context",
            "fusion": "single_layer",
            "probe_layer": 12,
            "positions": "gap_preserving",
            "decode": "layer_aware_cache",
        }
    elif args.method == "final":
        method_configuration = frozen["method"]
    elif args.method == "fastv":
        method_configuration = {"k": 3, "tokens": 48, "source": "pinned FastV adapter"}
    elif args.method == "visionzip":
        method_configuration = {
            "tokens": 48,
            "dominant": 40,
            "contextual": 8,
            "source": "pinned VisionZip adapter",
        }
    else:
        method_configuration = {"tokens": 576, "source": "full LLaVA"}
    write_json(
        run_dir / "config.json",
        config
        | {
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "checkpoint": args.model_path,
            "precision": "bf16",
            "batch_size": 1,
            "decoding": "deterministic greedy",
            "method_configuration": method_configuration,
            "warmup": args.warmup_examples,
        },
    )
    write_json(metrics_path, metrics)
    # Initialise provenance only after measurement so its background worker
    # cannot perturb the controlled timing loop.
    if args.wandb_mode != "disabled":
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_dir.name,
                group=args.wandb_group,
                job_type="serial-timing-trial",
                mode=args.wandb_mode,
                dir=str(run_dir),
                config={
                    "method": args.method,
                    "repetition": args.repetition,
                    "order_position": args.order_position,
                    "samples": args.limit,
                    "warmup": args.warmup_examples,
                    "manifest": str(args.manifest),
                },
                reinit=True,
                tags=["dissertation", "final", "efficiency", args.method, "serial"],
            )
        except Exception as exc:
            print(f"W&B unavailable; local timing remains authoritative: {type(exc).__name__}: {exc}")
    write_json(
        run_dir / "wandb_run.json",
        {
            "enabled": wandb_run is not None,
            "mode": args.wandb_mode,
            "project": args.wandb_project,
            "id": getattr(wandb_run, "id", None),
            "name": getattr(wandb_run, "name", None),
            "group": args.wandb_group,
            "initialised_after_measurement": True,
        },
    )
    if wandb_run is not None:
        try:
            wandb_run.log(metrics)
            wandb_run.finish()
        except Exception as exc:
            print(f"W&B final logging failed after local save: {type(exc).__name__}: {exc}")
    print(f"Saved timing trial {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
