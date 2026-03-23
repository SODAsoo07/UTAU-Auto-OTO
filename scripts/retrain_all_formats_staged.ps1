param(
    [string]$DatasetRoot = ".\dataset_staged",
    [string]$WorkspaceRoot = "",
    [string]$PythonExe = "python",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",
    [double]$BlankRiskWeight = 0.55,
    [switch]$SkipCache,
    [switch]$SkipTrain,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = Join-Path $RepoRoot "ml_workspace"
}

$datasetRootRaw = $DatasetRoot
if (-not [System.IO.Path]::IsPathRooted($datasetRootRaw)) {
    $datasetRootRaw = Join-Path $RepoRoot $datasetRootRaw
}

$DatasetRoot = [System.IO.Path]::GetFullPath($datasetRootRaw)
$WorkspaceRoot = [System.IO.Path]::GetFullPath($WorkspaceRoot)

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    $rendered = @($Exe) + $Args
    Write-Host ""
    Write-Host ">> $($rendered -join ' ')" -ForegroundColor Cyan
    if ($DryRun) {
        return
    }
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $DatasetRoot -PathType Container)) {
    throw "Dataset root not found: $DatasetRoot"
}

if (-not (Test-Path -LiteralPath $WorkspaceRoot)) {
    New-Item -ItemType Directory -Path $WorkspaceRoot | Out-Null
}

$AutoClassifyScript = Join-Path $RepoRoot "scripts\auto_classify_training_data.py"
$MelCacheScript = Join-Path $RepoRoot "ml\scripts\coupled\build_mel_patch_cache.py"
$TrainEnsembleScript = Join-Path $RepoRoot "scripts\train_ensemble_bundle.py"

foreach ($path in @($AutoClassifyScript, $MelCacheScript, $TrainEnsembleScript)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required script not found: $path"
    }
}

Write-Host "[INFO] RepoRoot     : $RepoRoot"
Write-Host "[INFO] DatasetRoot  : $DatasetRoot"
Write-Host "[INFO] WorkspaceRoot: $WorkspaceRoot"
Write-Host "[INFO] Auto matrix : KR(cv,cvc,cvvc,vcv), JA(cv,cvvc,vcv)"
Write-Host "[INFO] Outputs are written under workspace only."

if ($BlankRiskWeight -lt 0.0 -or $BlankRiskWeight -gt 0.90) {
    throw "BlankRiskWeight must be in [0.0, 0.90]."
}
$env:UTOA_ML_BLANK_RISK_WEIGHT = ("{0:0.##}" -f $BlankRiskWeight)
Write-Host "[INFO] UTOA_ML_BLANK_RISK_WEIGHT=$($env:UTOA_ML_BLANK_RISK_WEIGHT)"

# 1) Candidate classification + dataset CSV build
Invoke-Step -Exe $PythonExe -Args @(
    "-X", "utf8",
    $AutoClassifyScript,
    "--dataset-root", $DatasetRoot,
    "--workspace-root", $WorkspaceRoot,
    "--auto-oto-policy", "require",
    "--build-datasets"
)

$jobs = @(
    @{ Lang = "korean";   Fmt = "cv";   MinMapConf = "0.54"; Epochs = "90";  Batch = "128"; LR = "0.00045" },
    @{ Lang = "korean";   Fmt = "cvc";  MinMapConf = "0.55"; Epochs = "95";  Batch = "112"; LR = "0.00045" },
    @{ Lang = "korean";   Fmt = "cvvc"; MinMapConf = "0.56"; Epochs = "95";  Batch = "112"; LR = "0.00050" },
    @{ Lang = "korean";   Fmt = "vcv";  MinMapConf = "0.55"; Epochs = "105"; Batch = "96";  LR = "0.00035" },
    @{ Lang = "japanese"; Fmt = "cv";   MinMapConf = "0.50"; Epochs = "90";  Batch = "128"; LR = "0.00045" },
    @{ Lang = "japanese"; Fmt = "cvvc"; MinMapConf = "0.52"; Epochs = "95";  Batch = "112"; LR = "0.00050" },
    @{ Lang = "japanese"; Fmt = "vcv";  MinMapConf = "0.50"; Epochs = "105"; Batch = "96";  LR = "0.00035" }
)

foreach ($job in $jobs) {
    $lang = [string]$job.Lang
    $fmt = [string]$job.Fmt
    $csv = Join-Path $WorkspaceRoot "datasets\$lang\dataset_${lang}_${fmt}.csv"
    if (-not (Test-Path -LiteralPath $csv -PathType Leaf)) {
        Write-Warning "[SKIP] dataset csv missing: $csv"
        continue
    }

    $cacheDir = Join-Path $WorkspaceRoot "cache\mel_patches\$lang\$fmt\v2"
    if (-not $SkipCache) {
        Invoke-Step -Exe $PythonExe -Args @(
            "-X", "utf8",
            $MelCacheScript,
            "--dataset", $csv,
            "--dataset-root", $DatasetRoot,
            "--lang", $lang,
            "--format", $fmt,
            "--out-dir", $cacheDir,
            "--progress-every", "500",
            "--allow-missing"
        )
    }

    $outDir = Join-Path $WorkspaceRoot "models\$lang\$fmt\v1_ensemble"
    if (-not $SkipTrain) {
        Invoke-Step -Exe $PythonExe -Args @(
            "-X", "utf8",
            $TrainEnsembleScript,
            "--language", $lang,
            "--format", $fmt,
            "--dataset", $csv,
            "--rawmel-cache-dir", $cacheDir,
            "--out-dir", $outDir,
            "--coupled-backend", "coupled_nn_v2_rawmel",
            "--coupled-device", $Device,
            "--num-folds", "5",
            "--lightgbm-num-boost-round", "700",
            "--lightgbm-early-stopping-rounds", "80",
            "--coupled-epochs", ([string]$job.Epochs),
            "--coupled-batch-size", ([string]$job.Batch),
            "--coupled-learning-rate", ([string]$job.LR),
            "--min-mapping-confidence", ([string]$job.MinMapConf)
        )
    }

    Write-Host "[DONE] $lang/$fmt -> $outDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "[COMPLETE] Full matrix retraining pipeline finished."
Write-Host "[COMPLETE] Dataset folder remained untouched: $DatasetRoot"
