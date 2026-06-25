param(
    [string]$Python = ".\.venv313\Scripts\python.exe",
    [string]$WandbProject = "vlm-pruning",
    [switch]$NoWandb
)

$ErrorActionPreference = "Stop"

$runs = @(
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "visionzip"; keep = "0.75"; name = "refcoco500_llava15_grounding_visionzip_keep075_prior_seed0" },
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "visionzip"; keep = "0.50"; name = "refcoco500_llava15_grounding_visionzip_keep050_prior_seed0" },
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "visionzip"; keep = "0.25"; name = "refcoco500_llava15_grounding_visionzip_keep025_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "visionzip"; keep = "0.75"; name = "pope500_llava15_vqa_visionzip_keep075_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "visionzip"; keep = "0.50"; name = "pope500_llava15_vqa_visionzip_keep050_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "visionzip"; keep = "0.25"; name = "pope500_llava15_vqa_visionzip_keep025_prior_seed0" },
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "fastv"; keep = "0.75"; name = "refcoco500_llava15_grounding_fastv_keep075_prior_seed0" },
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "fastv"; keep = "0.50"; name = "refcoco500_llava15_grounding_fastv_keep050_prior_seed0" },
    @{ task = "grounding"; dataset = "refcoco"; manifest = "data\manifests\refcoco_val.jsonl"; method = "fastv"; keep = "0.25"; name = "refcoco500_llava15_grounding_fastv_keep025_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "fastv"; keep = "0.75"; name = "pope500_llava15_vqa_fastv_keep075_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "fastv"; keep = "0.50"; name = "pope500_llava15_vqa_fastv_keep050_prior_seed0" },
    @{ task = "vqa"; dataset = "pope"; manifest = "data\manifests\pope_val.jsonl"; method = "fastv"; keep = "0.25"; name = "pope500_llava15_vqa_fastv_keep025_prior_seed0" }
)

$env:WANDB_PROJECT = $WandbProject

foreach ($run in $runs) {
    $args = @(
        "scripts\run_prior_methods.py",
        "--task", $run.task,
        "--dataset", $run.dataset,
        "--manifest", $run.manifest,
        "--method", $run.method,
        "--keep-ratio", $run.keep,
        "--limit", "500",
        "--run-name", $run.name,
        "--quantization", "4bit",
        "--resume"
    )
    if (-not $NoWandb) {
        $args += @("--wandb", "--wandb-project", $WandbProject)
    }
    Write-Host "Running $($run.name)"
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Run failed with exit code $LASTEXITCODE`: $($run.name)"
    }
}
