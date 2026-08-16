#!/usr/bin/env python
"""Run the two bounded compilation feasibility attempts for selected M448_OPT."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from run_baseline import apply_chat_template, eos_token_ids  # noqa: E402
from run_official_baseline import load_manifest, load_model, load_sample_images  # noqa: E402
from run_progressive_oracle_refcoco import (  # noqa: E402
    DEFAULT_MODEL,
    MAX_NEW_TOKENS,
    REGISTERED_PROMPT,
    prepare_full_inputs,
    query_resolution,
)
from vlm_pruning.artifacts import sha256_file, source_provenance  # noqa: E402
from vlm_pruning.llava_hooks import (  # noqa: E402
    FusedNativeProgressiveForwarder,
    PruningEvent,
)


CAMPAIGN_ROOT = REPO_ROOT / "experiments/hardware_aware_zero_tail_refcoco_val_20260816"
MANIFEST = REPO_ROOT / "experiments/progressive_oracle_refcoco_val_20260814/manifests/refcoco_val_discovery1000.jsonl"
MANIFEST_SHA256 = "98577029365776a76bd5ec798da4ef06c546e298fd1e950bd238d6b9d127f1cd"
SELECTION = CAMPAIGN_ROOT / "runtime_repairs/aggregate/M448_OPT_SELECTION.json"
DEFAULT_OUTPUT = CAMPAIGN_ROOT / "runtime_repairs/compiled"
M448_SCHEDULE = (
    PruningEvent(4, 448, "query_topk_merge"),
    PruningEvent(12, 48, "query_topk"),
)
SOURCE_PATHS = (
    Path("scripts/attempt_runtime_compilation.py"),
    Path("scripts/run_progressive_oracle_refcoco.py"),
    Path("vlm_pruning/llava_hooks/mid_layer_pruning.py"),
    Path("vlm_pruning/llava_hooks/progressive_pruning.py"),
    Path("vlm_pruning/llava_hooks/runtime_repairs.py"),
)


class CompilationAttemptError(RuntimeError):
    """Raised when the bounded attempt contract is malformed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def generation_kwargs(torch: Any, processor: Any, model: Any, device: Any) -> tuple[dict[str, Any], str]:
    sample = load_manifest(MANIFEST, 1)[0]
    _, images = load_sample_images(sample)
    if len(images) != 1:
        raise CompilationAttemptError("Compilation feasibility sample must contain one image")
    prompt = REGISTERED_PROMPT.format(query=str(sample["query"]))
    text_prompt = apply_chat_template(processor, prompt, 1)
    inputs, _ = prepare_full_inputs(torch, processor, model, device, images, prompt)
    for image in images:
        image.close()
    image_token_id = int(getattr(model.config, "image_token_index", 32000))
    resolution = query_resolution(
        processor,
        text_prompt,
        str(sample["query"]),
        inputs["input_ids"],
        image_token_id,
    )
    return {
        "inputs_embeds": inputs["inputs_embeds"],
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "sample_id": str(sample["sample_id"]),
        "seed": 0,
        "query_original_positions": resolution.token_positions,
        "max_new_tokens": MAX_NEW_TOKENS,
        "eos_ids": eos_token_ids(model, processor),
    }, str(sample["sample_id"])


def make_forwarder(torch: Any, model: Any) -> FusedNativeProgressiveForwarder:
    return FusedNativeProgressiveForwarder(
        torch,
        model,
        image_token_id=int(getattr(model.config, "image_token_index", 32000)),
        schedule=M448_SCHEDULE,
        diagnostics_mode="benchmark",
    )


def same_generation(actual: Any, reference: Any) -> bool:
    def semantic_traces(result: Any) -> list[dict[str, Any]]:
        ignored = {"selection_latency_s", "merge_latency_s"}
        return [
            {key: value for key, value in trace.items() if key not in ignored}
            for trace in result.stage_traces
        ]

    return (
        actual.generated_ids.detach().cpu().tolist()
        == reference.generated_ids.detach().cpu().tolist()
        and actual.final_retained_indices == reference.final_retained_indices
        and semantic_traces(actual) == semantic_traces(reference)
    )


def worker(mode: str) -> int:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "mode": mode,
        "status": "started",
        "started_at_utc": utc_now(),
        "backend": "inductor" if mode == "torch_compile" else "torch.cuda.CUDAGraph",
    }
    try:
        torch, processor, model, device, model_environment = load_model(DEFAULT_MODEL, "bf16")
        if not torch.cuda.is_available():
            record.update(status="skipped_unsupported", reason="CUDA is unavailable")
        else:
            kwargs, sample_id = generation_kwargs(torch, processor, model, device)
            forwarder = make_forwarder(torch, model)
            with torch.inference_mode():
                reference = forwarder.generate_with_segment_caches(**kwargs)
            torch.cuda.synchronize()
            record.update(
                sample_id=sample_id,
                torch_version=str(torch.__version__),
                cuda_version=str(torch.version.cuda),
                model_environment=model_environment,
                reference_generated_ids=reference.generated_ids.detach().cpu().tolist(),
            )
            attempt_started = time.perf_counter()
            if mode == "torch_compile":
                compiled = torch.compile(
                    forwarder.generate_with_segment_caches,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=True,
                )
                with torch.inference_mode():
                    first = compiled(**kwargs)
                torch.cuda.synchronize()
                compile_and_first_call_s = time.perf_counter() - attempt_started
                steady_started = time.perf_counter()
                with torch.inference_mode():
                    second = compiled(**kwargs)
                torch.cuda.synchronize()
                record.update(
                    status=(
                        "feasible_exact"
                        if same_generation(first, reference) and same_generation(second, reference)
                        else "rejected_semantic_drift"
                    ),
                    compilation_and_first_call_s=compile_and_first_call_s,
                    steady_state_call_s=time.perf_counter() - steady_started,
                    exact_first_call=same_generation(first, reference),
                    exact_second_call=same_generation(second, reference),
                )
            elif mode == "cuda_graph":
                warm_stream = torch.cuda.Stream()
                warm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warm_stream), torch.inference_mode():
                    make_forwarder(torch, model).generate_with_segment_caches(**kwargs)
                torch.cuda.current_stream().wait_stream(warm_stream)
                torch.cuda.synchronize()
                capture_forwarder = make_forwarder(torch, model)
                graph = torch.cuda.CUDAGraph()
                capture_started = time.perf_counter()
                with torch.cuda.graph(graph), torch.inference_mode():
                    captured = capture_forwarder.generate_with_segment_caches(**kwargs)
                torch.cuda.synchronize()
                capture_s = time.perf_counter() - capture_started
                replay_started = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                record.update(
                    status=("feasible_exact" if same_generation(captured, reference) else "rejected_semantic_drift"),
                    capture_s=capture_s,
                    replay_s=time.perf_counter() - replay_started,
                    exact_after_replay=same_generation(captured, reference),
                )
            else:
                raise CompilationAttemptError(f"Unsupported worker mode: {mode}")
    except Exception as exc:  # The failure itself is required feasibility evidence.
        record.update(
            status="rejected_runtime_failure",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(),
        )
    record["total_worker_wall_s"] = time.perf_counter() - started
    record["finished_at_utc"] = utc_now()
    print(json.dumps(record, sort_keys=True))
    return 0


def last_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_parent(output_dir: Path) -> int:
    if output_dir.exists():
        raise CompilationAttemptError(f"Refusing to overwrite compilation evidence: {output_dir}")
    if sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise CompilationAttemptError("Frozen validation manifest hash differs")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if selection.get("m448_opt") != "R_FUSED_NATIVE":
        raise CompilationAttemptError("Compilation attempt is registered only for selected R_FUSED_NATIVE")
    source = source_provenance(REPO_ROOT, SOURCE_PATHS)
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if status:
        raise CompilationAttemptError("Compilation attempts require clean committed executable source")
    output_dir.mkdir(parents=True)
    results: dict[str, Any] = {}
    for mode in ("torch_compile", "cuda_graph"):
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=300,
                check=False,
            )
            result = last_json_object(completed.stdout) or {
                "mode": mode,
                "status": "rejected_missing_worker_record",
            }
            result["worker_returncode"] = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            result = {
                "mode": mode,
                "status": "rejected_timeout",
                "timeout_s": 300,
            }
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
        result["parent_observed_wall_s"] = time.perf_counter() - started
        atomic_write(output_dir / f"{mode}.stdout.log", str(stdout).encode())
        atomic_write(output_dir / f"{mode}.stderr.log", str(stderr).encode())
        result["stdout_sha256"] = sha256_file(output_dir / f"{mode}.stdout.log")
        result["stderr_sha256"] = sha256_file(output_dir / f"{mode}.stderr.log")
        results[mode] = result
    artifact = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "platform": platform.platform(),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "manifest_sha256": MANIFEST_SHA256,
        "m448_opt_selection": str(SELECTION.relative_to(REPO_ROOT)),
        "m448_opt_selection_sha256": sha256_file(SELECTION),
        "source_git_commit": source["git_commit"],
        "source_sha256": source["source_sha256"],
        "attempts": results,
        "bounded_search": {
            "torch_compile": "one inductor/fullgraph/dynamic attempt",
            "cuda_graph": "one full natural-generation capture/replay attempt",
            "additional_backends_or_modes_attempted": False,
        },
    }
    atomic_write(output_dir / "ATTEMPT_RESULTS.json", canonical_bytes(artifact))
    atomic_write(
        output_dir / "COMPLETED.json",
        canonical_bytes(
            {
                "status": "complete",
                "attempt_results_sha256": sha256_file(output_dir / "ATTEMPT_RESULTS.json"),
                "terminal_statuses": {mode: result["status"] for mode, result in results.items()},
            }
        ),
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker", choices=["torch_compile", "cuda_graph"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker(args.worker)
    return run_parent(args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
