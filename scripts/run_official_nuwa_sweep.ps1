param(
  [int]$Limit = 500,
  [int]$Seed = 0,
  [string]$CacheDir = "D:\piptemp\hf_cache_codex"
)

$ErrorActionPreference = "Stop"

$env:HF_HOME = $CacheDir
$env:HF_HUB_CACHE = Join-Path $CacheDir "hub"
$env:TRANSFORMERS_CACHE = Join-Path $CacheDir "hub"
$env:HF_HUB_DISABLE_XET = "1"

$Python = ".\.venv313\Scripts\python.exe"
$Runner = "scripts\run_official_nuwa.py"

$jobs = @(
  @{ Task = "vqa"; Dataset = "pope"; Manifest = "data\manifests\pope_val.jsonl"; Method = "official_full"; Name = "pope_llava15_vqa_official_full_tokens576_seed${Seed}_limit${Limit}" },
  @{ Task = "grounding"; Dataset = "refcoco"; Manifest = "data\manifests\refcoco_val.jsonl"; Method = "official_full"; Name = "refcoco_llava15_grounding_official_full_tokens576_seed${Seed}_limit${Limit}" }
)

foreach ($mt in @(64, 128, 192)) {
  $tokens = @{ 64 = 16; 128 = 32; 192 = 48 }[$mt]
  $jobs += @{ Task = "vqa"; Dataset = "pope"; Manifest = "data\manifests\pope_val.jsonl"; Method = "official_nuwa"; Matain = $mt; Tokens = $tokens; Name = "pope_llava15_vqa_official_nuwa_mt${mt}_d280_tokens${tokens}_seed${Seed}_limit${Limit}" }
  $jobs += @{ Task = "grounding"; Dataset = "refcoco"; Manifest = "data\manifests\refcoco_val.jsonl"; Method = "official_nuwa"; Matain = $mt; Tokens = $tokens; Name = "refcoco_llava15_grounding_official_nuwa_mt${mt}_d280_tokens${tokens}_seed${Seed}_limit${Limit}" }
}

foreach ($job in $jobs) {
  $args = @(
    $Runner,
    "--task", $job.Task,
    "--dataset", $job.Dataset,
    "--manifest", $job.Manifest,
    "--method", $job.Method,
    "--limit", "$Limit",
    "--seed", "$Seed",
    "--cache-dir", $CacheDir,
    "--run-name", $job.Name,
    "--no-wandb"
  )
  if ($job.Method -eq "official_nuwa") {
    $args += @("--matain-token", "$($job.Matain)", "--distance", "280")
  }
  & $Python @args
}
