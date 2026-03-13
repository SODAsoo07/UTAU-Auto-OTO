param(
    [string]$WorkspaceRoot = "",
    [string]$DatasetRoot = "",
    [ValidateSet("korean", "japanese")]
    [string]$Lang = "korean",
    [ValidateSet("cv", "cvc", "cvvc", "vcv", "general")]
    [string]$Format = "cvvc",
    [string]$DatasetCsv = "",
    [string]$OutDir = "",
    [string]$EvalReport = "",
    [string]$PreparedReport = "",
    [string]$SingleWorkDir = "",
    [string]$SingleManualOto = "",
    [string]$SingleVoicebankId = "",
    [string]$AliasTypes = "",
    [string]$AliasFamily = "",
    [double]$MinMappingConfidence = 0.0,
    [string]$PythonExe = "python",
    [string]$Device = "cpu",
    [int]$Epochs = 70,
    [int]$BatchSize = 192,
    [double]$LearningRate = 0.001,
    [double]$MinConfidence = 0.55,
    [switch]$StageSources,
    [switch]$PrepareAuto,
    [switch]$ResumePrepare,
    [switch]$RetryFailedPrepare,
    [switch]$BuildDatasetOnly,
    [switch]$TrainOnly,
    [switch]$EvaluateOnly,
    [switch]$KeepDatasetCsv,
    [switch]$ExportBundle,
    [switch]$InstallBundle,
    [bool]$AutoPrepareOnMissing = $true,
    [string]$ExportRoot = "",
    [string]$BundlePath = "",
    [string]$InstallRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $WorkspaceRoot) { $WorkspaceRoot = Join-Path $RepoRoot "ml_workspace" }
if (-not $DatasetCsv) { $DatasetCsv = Join-Path $WorkspaceRoot ("datasets\{0}\dataset_{0}_{1}_coupled.csv" -f $Lang, $Format) }
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot ("ML_models\{0}\{1}\v1_coupled" -f $Lang, $Format) }
if (-not $EvalReport) { $EvalReport = Join-Path $RepoRoot ("logs\eval_{0}_{1}_coupled.json" -f $Lang, $Format) }
if (-not $ExportRoot) { $ExportRoot = Join-Path $WorkspaceRoot "exports\model_bundles" }
if (-not $InstallRoot) { $InstallRoot = Join-Path $RepoRoot "models_installed\oto_ml" }

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
try {
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $OutputEncoding = $utf8NoBom
} catch {
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$StageScript = Join-Path $RepoRoot "ml\scripts\stage_training_sources.py"
$PrepareScript = Join-Path $RepoRoot "ml\scripts\prepare_staged_auto_pairs.py"
$BuildScript = Join-Path $RepoRoot "ml\scripts\build_oto_mel_coupled_dataset.py"
$TrainScript = Join-Path $RepoRoot "ml\scripts\train_oto_mel_coupled_model.py"
$EvalScript = Join-Path $RepoRoot "ml\scripts\evaluate_oto_mel_coupled.py"
$ExportScript = Join-Path $RepoRoot "ml\scripts\export_oto_ml_bundle.py"
$InstallScript = Join-Path $RepoRoot "ml\scripts\install_oto_ml_bundle.py"
$script:AutoPrepareAttempted = $false

function Resolve-DefaultDatasetRoot {
    $mainRoot = Join-Path $RepoRoot "dataset"
    return $mainRoot
}

if (-not $DatasetRoot) { $DatasetRoot = Resolve-DefaultDatasetRoot }
if (-not $PreparedReport) { $PreparedReport = Join-Path $DatasetRoot "_manifest\prepared_auto_pairs.json" }

function Ensure-ParentDir {
    param([string]$Path)
    $parent = Split-Path -Parent $Path
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
}

function Resolve-ManualOtoFromWorkDir {
    param([string]$WorkDir)
    $candidates = @()
    $otoIni = Join-Path $WorkDir "oto.ini"
    if (Test-Path $otoIni) { $candidates += $otoIni }
    $extra = Get-ChildItem -Path $WorkDir -File -Filter "*.ini" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "oto|base" } |
        Sort-Object Name |
        Select-Object -ExpandProperty FullName
    $candidates += $extra
    foreach ($path in $candidates) {
        if (-not (Test-Path $path)) { continue }
        try {
            $line = Get-Content -Path $path -Encoding UTF8 -ErrorAction Stop | Select-Object -First 64 |
                Where-Object { $_ -match "=" -and $_ -match "," } | Select-Object -First 1
            if ($line) { return $path }
        } catch {
            continue
        }
    }
    return ""
}

function Discover-WorkItemsByScan {
    param(
        [string]$LangValue,
        [string]$FormatValue
    )
    $fmtRoot = Join-Path (Join-Path $DatasetRoot $LangValue) $FormatValue
    if (-not (Test-Path $fmtRoot)) {
        return @()
    }

    $items = @()
    $autoFiles = Get-ChildItem -Path $fmtRoot -Recurse -File -Filter "oto_auto_ml.ini" -ErrorAction SilentlyContinue
    foreach ($autoFile in $autoFiles) {
        $work = $autoFile.DirectoryName
        $manual = Resolve-ManualOtoFromWorkDir -WorkDir $work
        if (-not $manual) { continue }
        $rel = ""
        try {
            $rel = [System.IO.Path]::GetRelativePath($fmtRoot, $work)
        } catch {
            $rel = ""
        }
        $segments = @()
        if ($rel) {
            $segments = @($rel -split "[\\/]" | Where-Object { $_ })
        }
        $voicebank = if ($segments.Count -gt 0) { $segments[0] } else { Split-Path -Leaf (Split-Path -Parent $work) }
        $items += [pscustomobject]@{
            language = $LangValue
            format_type = $FormatValue
            status = "scan_discovered"
            work_dir = $work
            auto_oto = $autoFile.FullName
            manual_oto = $manual
            stage_root = (Join-Path $fmtRoot $voicebank)
        }
    }
    return @($items)
}

function Get-TargetItemsFromPreparedReport {
    if (-not (Test-Path $PreparedReport)) {
        return @()
    }
    try {
        $payload = Get-Content -Path $PreparedReport -Encoding UTF8 | ConvertFrom-Json
        $items = @($payload.items)
    } catch {
        return @()
    }
    return @($items | Where-Object {
        ($_ -ne $null) -and
        (($_.language -as [string]).ToLower() -eq $Lang) -and
        (($_.format_type -as [string]).ToLower() -eq $Format) -and
        (($_.status -as [string]) -in @("prepared", "prepared_existing"))
    })
}

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string[]]$ArgsList
    )
    $cmd = @("-X", "utf8", $ScriptPath) + $ArgsList
    if ($DryRun) {
        Write-Host ("[dry-run] {0} {1}" -f $PythonExe, ($cmd -join " "))
        return
    }
    & $PythonExe @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "python exited with code $LASTEXITCODE for $ScriptPath"
    }
}

function Get-PrepareScriptArgs {
    $args = @("--dataset-root", $DatasetRoot)
    $useResume = $ResumePrepare.IsPresent -or (-not $StageSources)
    if ($RetryFailedPrepare) {
        $useResume = $true
    }
    if ($useResume) {
        $args += "--resume"
    }
    if ($RetryFailedPrepare) {
        $args += "--retry-failed"
    }
    return ,$args
}

function Build-DatasetFromSingleWorkDir {
    $work = (Resolve-Path $SingleWorkDir).Path
    $auto = Join-Path $work "oto_auto_ml.ini"
    $manual = $SingleManualOto
    if (-not $manual) { $manual = Resolve-ManualOtoFromWorkDir -WorkDir $work }
    if (-not $manual) { throw "manual oto not found in $work" }
    if (-not (Test-Path $auto)) { throw "auto oto not found: $auto" }
    if (-not (Test-Path $manual)) { throw "manual oto not found: $manual" }
    $voicebank = $SingleVoicebankId
    if (-not $voicebank) {
        $voicebank = Split-Path -Leaf (Split-Path -Parent $work)
    }
    Ensure-ParentDir -Path $DatasetCsv
    if ((Test-Path $DatasetCsv) -and -not $KeepDatasetCsv -and -not $DryRun) {
        Remove-Item -Path $DatasetCsv -Force
    }
    $args = @(
        "--lang", $Lang,
        "--auto", $auto,
        "--manual", $manual,
        "--tg-dir", $work,
        "--wav-dir", $work,
        "--out", $DatasetCsv,
        "--voicebank-id", $voicebank,
        "--format-override", $Format
    )
    Invoke-PythonScript -ScriptPath $BuildScript -ArgsList $args
}

function Build-DatasetFromPreparedReport {
    $target = @()

    if (Test-Path $PreparedReport) {
        $target = Get-TargetItemsFromPreparedReport
    } elseif ($DryRun) {
        Write-Host "[dry-run] prepared report missing: $PreparedReport (skip dataset build preview)"
        return
    } elseif ($AutoPrepareOnMissing -and -not $script:AutoPrepareAttempted) {
        $script:AutoPrepareAttempted = $true
        Write-Host "[build] prepared report missing -> running auto prepare once"
        try {
            Invoke-PythonScript -ScriptPath $PrepareScript -ArgsList (Get-PrepareScriptArgs)
        } catch {
            Write-Warning ("[build] auto prepare failed: {0}" -f $_.Exception.Message)
        }
        if (Test-Path $PreparedReport) {
            $target = Get-TargetItemsFromPreparedReport
        }
    }
    if (-not $target -or $target.Count -eq 0) {
        $scanItems = Discover-WorkItemsByScan -LangValue $Lang -FormatValue $Format
        if ($scanItems.Count -gt 0) {
            Write-Host "[build] prepared report miss -> scanned dataset root and found $($scanItems.Count) items"
            $target = @($scanItems)
        }
    }
    if ((-not $target -or $target.Count -eq 0) -and (-not $DryRun) -and $AutoPrepareOnMissing -and -not $script:AutoPrepareAttempted) {
        $script:AutoPrepareAttempted = $true
        Write-Host "[build] no prepared/scan items -> running auto prepare once"
        try {
            Invoke-PythonScript -ScriptPath $PrepareScript -ArgsList (Get-PrepareScriptArgs)
        } catch {
            Write-Warning ("[build] auto prepare failed: {0}" -f $_.Exception.Message)
        }
        $target = Get-TargetItemsFromPreparedReport
        if (-not $target -or $target.Count -eq 0) {
            $scanItems = Discover-WorkItemsByScan -LangValue $Lang -FormatValue $Format
            if ($scanItems.Count -gt 0) {
                Write-Host "[build] post-prepare scan found $($scanItems.Count) items"
                $target = @($scanItems)
            }
        }
    }
    if (-not $target -or $target.Count -eq 0) {
        if ($DryRun) {
            Write-Host "[dry-run] no prepared items for $Lang/$Format in $PreparedReport (skip dataset build preview)"
            return
        }
        throw "no prepared items for $Lang/$Format in $PreparedReport (hint: run with -PrepareAuto or verify dataset\$Lang\$Format contains oto_auto_ml.ini + manual oto.ini)"
    }

    Ensure-ParentDir -Path $DatasetCsv
    if ((Test-Path $DatasetCsv) -and -not $KeepDatasetCsv -and -not $DryRun) {
        Remove-Item -Path $DatasetCsv -Force
    }

    $built = 0
    foreach ($it in $target) {
        $work = [string]$it.work_dir
        if (-not $work) { continue }
        $auto = [string]$it.auto_oto
        if (-not $auto) { $auto = Join-Path $work "oto_auto_ml.ini" }
        $manual = [string]$it.manual_oto
        if (-not $manual) { $manual = Resolve-ManualOtoFromWorkDir -WorkDir $work }
        if (-not $auto -or -not $manual) { continue }
        if ((-not $DryRun) -and ((-not (Test-Path $auto)) -or (-not (Test-Path $manual)))) { continue }

        $voicebank = Split-Path -Leaf ([string]$it.stage_root)
        if (-not $voicebank) { $voicebank = Split-Path -Leaf (Split-Path -Parent $work) }

        $args = @(
            "--lang", $Lang,
            "--auto", $auto,
            "--manual", $manual,
            "--tg-dir", $work,
            "--wav-dir", $work,
            "--out", $DatasetCsv,
            "--voicebank-id", $voicebank,
            "--format-override", $Format
        )
        if ($built -gt 0 -or $KeepDatasetCsv) {
            $args += "--append"
        }
        Invoke-PythonScript -ScriptPath $BuildScript -ArgsList $args
        $built += 1
    }

    if ($built -le 0) {
        throw "dataset build failed: no valid items for $Lang/$Format"
    }
    Write-Host "[build] dataset rows appended from items: $built"
}

function Resolve-InstallBundlePath {
    if ($BundlePath) {
        return (Resolve-Path $BundlePath).Path
    }
    $manifestPath = Join-Path $ExportRoot "export_manifest.json"
    if (Test-Path $manifestPath) {
        $manifest = Get-Content -Path $manifestPath -Encoding UTF8 | ConvertFrom-Json
        foreach ($bundle in @($manifest.bundles)) {
            if (($bundle.source_dir -as [string]) -eq (Resolve-Path $OutDir).Path) {
                if ($bundle.zip_path -and (Test-Path $bundle.zip_path)) {
                    return (Resolve-Path $bundle.zip_path).Path
                }
                if ($bundle.export_dir -and (Test-Path $bundle.export_dir)) {
                    return (Resolve-Path $bundle.export_dir).Path
                }
            }
        }
    }
    $zip = Get-ChildItem -Path $ExportRoot -File -Filter "*.zip" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($zip) { return $zip.FullName }
    return ""
}

$runBuild = $true
$runTrain = $true
$runEval = $true
if ($BuildDatasetOnly) { $runTrain = $false; $runEval = $false }
if ($TrainOnly) { $runBuild = $false; $runEval = $false }
if ($EvaluateOnly) { $runBuild = $false; $runTrain = $false }

Write-Host "[info] repo=$RepoRoot"
Write-Host "[info] lang=$Lang format=$Format"
Write-Host "[info] dataset_root=$DatasetRoot"
Write-Host "[info] dataset_csv=$DatasetCsv"
Write-Host "[info] out_dir=$OutDir"

if ($StageSources) {
    Invoke-PythonScript -ScriptPath $StageScript -ArgsList @("--dataset-root", $DatasetRoot)
}
if ($PrepareAuto) {
    Invoke-PythonScript -ScriptPath $PrepareScript -ArgsList (Get-PrepareScriptArgs)
}

if ($runBuild) {
    if ($SingleWorkDir) {
        Build-DatasetFromSingleWorkDir
    } else {
        Build-DatasetFromPreparedReport
    }
}

if ($runTrain) {
    if ((-not $DryRun) -and (-not (Test-Path $DatasetCsv))) {
        throw "dataset csv not found: $DatasetCsv"
    }
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    $trainArgs = @(
        "--lang", $Lang,
        "--format", $Format,
        "--dataset", $DatasetCsv,
        "--out-dir", $OutDir,
        "--device", $Device,
        "--epochs", "$Epochs",
        "--batch-size", "$BatchSize",
        "--learning-rate", "$LearningRate",
        "--min-confidence", "$MinConfidence",
        "--min-mapping-confidence", "$MinMappingConfidence"
    )
    if ($AliasTypes) {
        $trainArgs += @("--alias-types", $AliasTypes)
    }
    if ($AliasFamily) {
        $trainArgs += @("--alias-family", $AliasFamily)
    }
    Invoke-PythonScript -ScriptPath $TrainScript -ArgsList $trainArgs
}

if ($runEval) {
    Ensure-ParentDir -Path $EvalReport
    $evalArgs = @(
        "--model-dir", $OutDir,
        "--dataset", $DatasetCsv,
        "--lang", $Lang,
        "--format", $Format,
        "--device", $Device,
        "--report", $EvalReport
    )
    Invoke-PythonScript -ScriptPath $EvalScript -ArgsList $evalArgs
}

if ($ExportBundle) {
    New-Item -ItemType Directory -Path $ExportRoot -Force | Out-Null
    $exportArgs = @(
        "--model-dir", $OutDir,
        "--export-root", $ExportRoot,
        "--zip"
    )
    Invoke-PythonScript -ScriptPath $ExportScript -ArgsList $exportArgs
}

if ($InstallBundle) {
    $resolvedBundle = Resolve-InstallBundlePath
    if (-not $resolvedBundle) {
        throw "bundle path not resolved. Use -BundlePath or run -ExportBundle first."
    }
    $installArgs = @(
        "--bundle", $resolvedBundle,
        "--install-root", $InstallRoot
    )
    Invoke-PythonScript -ScriptPath $InstallScript -ArgsList $installArgs
}

Write-Host "[done] coupled mel+oto pipeline complete"
