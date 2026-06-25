param(
  [int]$Limit = 500,
  [int]$Seed = 0
)

$ErrorActionPreference = "Stop"

$Python = ".\.venv313\Scripts\python.exe"
$OutputRoot = "experiments\runs_official_nuwa_matched_baselines"
$FiguresRoot = "experiments\figures_official_nuwa_matched_baselines"
$TablesRoot = "experiments\tables\official_nuwa_matched_source"

function Test-RunComplete {
  param(
    [string]$RunName,
    [int]$ExpectedRows
  )
  $runDir = Join-Path $OutputRoot $RunName
  $metrics = Join-Path $runDir "metrics.json"
  $samples = Join-Path $runDir "per_sample_results.jsonl"
  if (-not (Test-Path -LiteralPath $metrics) -or -not (Test-Path -LiteralPath $samples)) {
    return $false
  }
  $lineCount = (Get-Content -LiteralPath $samples | Measure-Object -Line).Lines
  return $lineCount -eq $ExpectedRows
}

function Get-ResumeArgs {
  param([string]$RunName)
  $samples = Join-Path (Join-Path $OutputRoot $RunName) "per_sample_results.jsonl"
  if (Test-Path -LiteralPath $samples) {
    return @("--resume")
  }
  return @()
}

function Assert-LastCommandSucceeded {
  param([string]$RunName)
  if ($LASTEXITCODE -ne 0) {
    throw "Matched baseline run failed with exit code ${LASTEXITCODE}: $RunName"
  }
}

$budgets = @(
  @{ Tokens = 16; Ratio = "0.027777777777777776"; Label = "tokens016" },
  @{ Tokens = 32; Ratio = "0.05555555555555555"; Label = "tokens032" },
  @{ Tokens = 48; Ratio = "0.08333333333333333"; Label = "tokens048" }
)

$tasks = @(
  @{ Task = "vqa"; Dataset = "pope"; Manifest = "data\manifests\pope_val.jsonl" },
  @{ Task = "grounding"; Dataset = "refcoco"; Manifest = "data\manifests\refcoco_val.jsonl" }
)

foreach ($task in $tasks) {
  foreach ($budget in $budgets) {
    foreach ($method in @("random", "uniform")) {
      $runName = "$($task.Dataset)_llava15_$($task.Task)_${method}_matched_$($budget.Label)_seed${Seed}_limit${Limit}"
      if (Test-RunComplete -RunName $runName -ExpectedRows $Limit) {
        Write-Host "Skipping completed run $runName"
        continue
      }
      $resumeArgs = Get-ResumeArgs -RunName $runName
      & $Python scripts\run_baseline.py `
        --task $task.Task `
        --dataset $task.Dataset `
        --manifest $task.Manifest `
        --method $method `
        --keep-ratio $budget.Ratio `
        --token-budget $budget.Tokens `
        --seed $Seed `
        --limit $Limit `
        --run-name $runName `
        --output-root $OutputRoot `
        --figures-root $FiguresRoot `
        --tables-root $TablesRoot `
        --quantization 4bit `
        --pruning-implementation true_token_removal `
        --pruning-location after_projector_before_llm_prefill `
        @resumeArgs
      Assert-LastCommandSucceeded -RunName $runName
    }

    foreach ($method in @("visionzip", "fastv")) {
      $runName = "$($task.Dataset)_llava15_$($task.Task)_${method}_matched_$($budget.Label)_seed${Seed}_limit${Limit}"
      if (Test-RunComplete -RunName $runName -ExpectedRows $Limit) {
        Write-Host "Skipping completed run $runName"
        continue
      }
      $resumeArgs = Get-ResumeArgs -RunName $runName
      & $Python scripts\run_prior_methods.py `
        --task $task.Task `
        --dataset $task.Dataset `
        --manifest $task.Manifest `
        --method $method `
        --keep-ratio $budget.Ratio `
        --token-budget $budget.Tokens `
        --seed $Seed `
        --limit $Limit `
        --run-name $runName `
        --output-root $OutputRoot `
        --figures-root $FiguresRoot `
        --tables-root $TablesRoot `
        --quantization 4bit `
        @resumeArgs
      Assert-LastCommandSucceeded -RunName $runName
    }
  }
}
