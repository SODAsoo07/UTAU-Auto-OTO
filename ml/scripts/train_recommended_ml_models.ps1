param(
    [string]$WorkspaceRoot = "",
    [string]$PythonExe = "python",
    [string]$SelectorObjective = "pointwise",
    [switch]$NoSelector,
    [switch]$RunBatchEvaluation,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = Join-Path $RepoRoot "logs\ml_workspace"
}

$TrainScript = Join-Path $RepoRoot "ml\scripts\train_oto_ml_model.py"
$EvalBatchScript = Join-Path $RepoRoot "ml\scripts\evaluate_oto_ml_batch.py"
$Failures = New-Object System.Collections.Generic.List[string]

$Targets = @(
    @{
        Name = "korean-cv"
        Lang = "korean"
        Format = "cv"
        AliasFamily = ""
        Dataset = Join-Path $WorkspaceRoot "datasets\korean\dataset_korean_cv.csv"
        OutDir = Join-Path $WorkspaceRoot "models\korean_cv_bundle"
    },
    @{
        Name = "japanese-cv"
        Lang = "japanese"
        Format = "cv"
        AliasFamily = ""
        Dataset = Join-Path $WorkspaceRoot "datasets\japanese\dataset_japanese_cv.csv"
        OutDir = Join-Path $WorkspaceRoot "models\japanese_cv_bundle"
    },
    @{
        Name = "korean-cvc-cv"
        Lang = "korean"
        Format = "cvc"
        AliasFamily = "cv"
        Dataset = Join-Path $WorkspaceRoot "datasets\korean\dataset_korean_cvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\korean\cvc\families\cv\v1"
    },
    @{
        Name = "korean-cvc-bridge"
        Lang = "korean"
        Format = "cvc"
        AliasFamily = "bridge"
        Dataset = Join-Path $WorkspaceRoot "datasets\korean\dataset_korean_cvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\korean\cvc\families\bridge\v1"
    },
    @{
        Name = "korean-cvvc-cv"
        Lang = "korean"
        Format = "cvvc"
        AliasFamily = "cv"
        Dataset = Join-Path $WorkspaceRoot "datasets\korean\dataset_korean_cvvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\korean\cvvc\families\cv\v1"
    },
    @{
        Name = "korean-cvvc-bridge"
        Lang = "korean"
        Format = "cvvc"
        AliasFamily = "bridge"
        Dataset = Join-Path $WorkspaceRoot "datasets\korean\dataset_korean_cvvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\korean\cvvc\families\bridge\v1"
    },
    @{
        Name = "japanese-cvvc-cv"
        Lang = "japanese"
        Format = "cvvc"
        AliasFamily = "cv"
        Dataset = Join-Path $WorkspaceRoot "datasets\japanese\dataset_japanese_cvvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\japanese\cvvc\families\cv\v1"
    },
    @{
        Name = "japanese-cvvc-bridge"
        Lang = "japanese"
        Format = "cvvc"
        AliasFamily = "bridge"
        Dataset = Join-Path $WorkspaceRoot "datasets\japanese\dataset_japanese_cvvc.csv"
        OutDir = Join-Path $WorkspaceRoot "models\japanese\cvvc\families\bridge\v1"
    }
)

function Invoke-TrainTarget {
    param(
        [hashtable]$Target
    )

    if (-not (Test-Path $Target.Dataset)) {
        Write-Host "[skip] missing dataset: $($Target.Name) -> $($Target.Dataset)"
        return
    }

    $args = @(
        "-X", "utf8",
        $TrainScript,
        "--lang", $Target.Lang,
        "--format", $Target.Format,
        "--dataset", $Target.Dataset,
        "--out-dir", $Target.OutDir
    )

    if ($Target.AliasFamily) {
        $args += @("--alias-family", $Target.AliasFamily)
    }
    if (-not $NoSelector) {
        $args += @("--include-selector", "--selector-objective", $SelectorObjective)
    }

    Write-Host "[train] $($Target.Name)"
    Write-Host "        dataset=$($Target.Dataset)"
    Write-Host "        out_dir=$($Target.OutDir)"

    if ($DryRun) {
        Write-Host ("        cmd={0} {1}" -f $PythonExe, ($args -join " "))
        return
    }

    try {
        & $PythonExe @args
        if ($LASTEXITCODE -ne 0) {
            throw "python exited with code $LASTEXITCODE"
        }
    }
    catch {
        $Failures.Add($Target.Name) | Out-Null
        Write-Warning "[failed] $($Target.Name): $_"
    }
}

foreach ($target in $Targets) {
    Invoke-TrainTarget -Target $target
}

if ($RunBatchEvaluation) {
    $evalArgs = @(
        "-X", "utf8",
        $EvalBatchScript,
        "--workspace-root", $WorkspaceRoot
    )
    Write-Host "[eval] batch evaluation"
    if ($DryRun) {
        Write-Host ("        cmd={0} {1}" -f $PythonExe, ($evalArgs -join " "))
    } else {
        & $PythonExe @evalArgs
    }
}

if ($Failures.Count -gt 0) {
    Write-Host ""
    Write-Warning ("Training finished with failures: " + ($Failures -join ", "))
    exit 1
}
