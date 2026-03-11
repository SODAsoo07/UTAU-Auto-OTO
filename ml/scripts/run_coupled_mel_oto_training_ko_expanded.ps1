param(
    [string]$Formats = "cv,cvc,vcv",
    [string]$WorkspaceRoot = "",
    [string]$DatasetRoot = "",
    [string]$PreparedReport = "",
    [string]$PythonExe = "python",
    [string]$Device = "cpu",
    [int]$Epochs = 70,
    [int]$BatchSize = 192,
    [double]$LearningRate = 0.001,
    [double]$MinConfidence = 0.55,
    [double]$MinMappingConfidence = -1.0,
    [switch]$StageSources,
    [switch]$PrepareAuto,
    [switch]$BuildDatasetOnly,
    [switch]$TrainOnly,
    [switch]$EvaluateOnly,
    [switch]$ExportBundle,
    [switch]$InstallBundle,
    [switch]$RequireAllFormats,
    [string]$ExportRoot = "",
    [string]$InstallRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$SingleScript = Join-Path $PSScriptRoot "run_coupled_mel_oto_training.ps1"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
try {
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
} catch {
}

if (-not (Test-Path $SingleScript)) {
    throw "missing script: $SingleScript"
}
if (-not $WorkspaceRoot) { $WorkspaceRoot = Join-Path $RepoRoot "ml_workspace" }
if ($DatasetRoot) { $DatasetRoot = (Resolve-Path $DatasetRoot).Path }
if (-not $PreparedReport -and $DatasetRoot) { $PreparedReport = Join-Path $DatasetRoot "_manifest\prepared_auto_pairs.json" }
if (-not $ExportRoot) { $ExportRoot = Join-Path $WorkspaceRoot "exports\model_bundles" }
if (-not $InstallRoot) { $InstallRoot = Join-Path $RepoRoot "models_installed\oto_ml" }

function Get-DatasetRootCandidates {
    if ($DatasetRoot) {
        return @($DatasetRoot)
    }
    $candidates = @(
        (Join-Path $RepoRoot "dataset_stage")
        (Join-Path $RepoRoot "dataset")
    ) | Where-Object { Test-Path $_ }
    if (-not $candidates -or $candidates.Count -eq 0) {
        return @(Join-Path $RepoRoot "dataset")
    }
    return @($candidates)
}

function Resolve-DatasetRootForFormat {
    param([string]$Fmt)
    $candidates = Get-DatasetRootCandidates
    $bestRoot = $candidates[0]
    $bestCount = -1
    foreach ($root in $candidates) {
        $scanRoot = Join-Path (Join-Path $root "korean") $Fmt
        $cnt = 0
        if (Test-Path $scanRoot) {
            $cnt = @(Get-ChildItem -Path $scanRoot -Recurse -File -Filter "oto_auto_ml.ini" -ErrorAction SilentlyContinue).Count
        }
        if ($cnt -gt $bestCount) {
            $bestCount = $cnt
            $bestRoot = $root
        }
    }
    return $bestRoot
}

function Get-AvailableItemCount {
    param([string]$Fmt)
    $rootForFmt = Resolve-DatasetRootForFormat -Fmt $Fmt
    $preparedForFmt = $PreparedReport
    if (-not $preparedForFmt) {
        $preparedForFmt = Join-Path $rootForFmt "_manifest\prepared_auto_pairs.json"
    }

    $count = 0
    if (Test-Path $preparedForFmt) {
        try {
            $payload = Get-Content -Path $preparedForFmt -Encoding UTF8 | ConvertFrom-Json
            $items = @($payload.items)
            $count += @($items | Where-Object {
                ($_ -ne $null) -and
                (($_.language -as [string]).ToLower() -eq "korean") -and
                (($_.format_type -as [string]).ToLower() -eq $Fmt) -and
                (($_.status -as [string]) -in @("prepared", "prepared_existing"))
            }).Count
        } catch {
            $count += 0
        }
    }
    $scanRoot = Join-Path (Join-Path $rootForFmt "korean") $Fmt
    if (Test-Path $scanRoot) {
        $autoCount = @(Get-ChildItem -Path $scanRoot -Recurse -File -Filter "oto_auto_ml.ini" -ErrorAction SilentlyContinue).Count
        if ($autoCount -gt $count) {
            $count = $autoCount
        }
    }
    return [int]$count
}

$formatList = @($Formats -split "," | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
if (-not $formatList -or $formatList.Count -eq 0) {
    throw "no formats provided"
}

$valid = @("cv", "cvc", "cvvc", "vcv", "general")
foreach ($fmt in $formatList) {
    if ($fmt -notin $valid) {
        throw "unsupported format: $fmt"
    }
}

# conservative defaults from policy (korean)
$minConfByFormat = @{
    "cv" = 0.55
    "cvc" = 0.75
    "cvvc" = 0.78
    "vcv" = 0.72
    "general" = 0.50
}

$first = $true
$runBuild = $true
if ($TrainOnly -or $EvaluateOnly) { $runBuild = $false }

foreach ($fmt in $formatList) {
    $effectiveMinConf = $MinMappingConfidence
    if ($effectiveMinConf -lt 0.0) {
        $effectiveMinConf = [double]$minConfByFormat[$fmt]
    }
    $rootForFmt = Resolve-DatasetRootForFormat -Fmt $fmt
    $preparedForFmt = $PreparedReport
    if (-not $preparedForFmt) {
        $preparedForFmt = Join-Path $rootForFmt "_manifest\prepared_auto_pairs.json"
    }

    Write-Host "[run] format=$fmt min_mapping_confidence=$effectiveMinConf"
    Write-Host "[run] dataset_root=$rootForFmt"

    if ($runBuild -and -not $DryRun) {
        $availableCount = Get-AvailableItemCount -Fmt $fmt
        if ($availableCount -le 0) {
            $msg = "no available coupled dataset items for korean/$fmt"
            if ($RequireAllFormats) {
                throw $msg
            }
            Write-Warning "[skip] $msg"
            continue
        }
    }

    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $SingleScript,
        "-Lang", "korean",
        "-Format", $fmt,
        "-WorkspaceRoot", $WorkspaceRoot,
        "-DatasetRoot", $rootForFmt,
        "-PreparedReport", $preparedForFmt,
        "-PythonExe", $PythonExe,
        "-Device", $Device,
        "-Epochs", "$Epochs",
        "-BatchSize", "$BatchSize",
        "-LearningRate", "$LearningRate",
        "-MinConfidence", "$MinConfidence",
        "-MinMappingConfidence", "$effectiveMinConf",
        "-ExportRoot", $ExportRoot,
        "-InstallRoot", $InstallRoot
    )

    if ($first -and $StageSources) { $args += "-StageSources" }
    if ($first -and $PrepareAuto) { $args += "-PrepareAuto" }
    if ($BuildDatasetOnly) { $args += "-BuildDatasetOnly" }
    if ($TrainOnly) { $args += "-TrainOnly" }
    if ($EvaluateOnly) { $args += "-EvaluateOnly" }
    if ($ExportBundle) { $args += "-ExportBundle" }
    if ($InstallBundle) { $args += "-InstallBundle" }
    if ($DryRun) { $args += "-DryRun" }

    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        if ($RequireAllFormats) {
            throw "failed format: $fmt"
        }
        Write-Warning "[skip] failed format: $fmt"
        continue
    }
    $first = $false
}

Write-Host "[done] korean expanded coupled training complete"
