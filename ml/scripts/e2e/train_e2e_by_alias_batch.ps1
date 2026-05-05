param(
    [string]$Root = "",
    [string]$Language = "korean",
    [string]$StageRoot = "",
    [string]$Formats = "cvvc,vcv,cvc",
    [int]$MaxBanksPerFormat = 3,
    [string]$RunTag = "",
    [string]$Python = "python",
    [string]$CVSubgroupBinary = "1",
    [double]$VVCutoffLossWeight = 1.0,
    [double]$VVPositiveCutoffLossWeight = 2.0,
    [string]$CutoffModeAdjust = "1",
    [int]$CutoffModeMinKeyCount = 32,
    [double]$CutoffModePositiveGate = 0.35,
    [string]$CutoffModePositiveOnly = "1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Root([string]$maybeRoot) {
    if ($maybeRoot) {
        return (Resolve-Path -LiteralPath $maybeRoot).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\\..\\..")).Path
}

function Normalize-Name([string]$text) {
    if (-not $text) { return "unknown" }
    $tmp = [Regex]::Replace($text, "[^0-9A-Za-z._-]+", "_")
    $tmp = $tmp.Trim("_")
    if (-not $tmp) { return "unknown" }
    return $tmp
}

function Get-AliasBucket([string]$aliasType) {
    $a = ("" + $aliasType).Trim().ToLowerInvariant()
    if ($a -in @("cv", "cv_head", "mono")) { return "cv" }
    if ($a -eq "vc") { return "vc" }
    if ($a -in @("vv", "vcv")) { return "vv" }
    return "other"
}

function Get-BankCandidates([string]$formatRoot, [int]$maxCount) {
    if (-not (Test-Path -LiteralPath $formatRoot)) { return @() }
    $otoDirs = Get-ChildItem -Path $formatRoot -Recurse -File -Filter "oto.ini" |
        ForEach-Object { $_.Directory.FullName } |
        Sort-Object -Unique

    $ok = @()
    foreach ($dir in $otoDirs) {
        $hasWav = (Get-ChildItem -Path $dir -Recurse -File -Filter "*.wav" -ErrorAction SilentlyContinue | Select-Object -First 1)
        $hasTg = (Get-ChildItem -Path $dir -Recurse -File -Filter "*.TextGrid" -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($hasWav -and $hasTg) {
            $ok += $dir
        }
    }

    if ($maxCount -gt 0 -and $ok.Count -gt $maxCount) {
        return @($ok | Select-Object -First $maxCount)
    }
    return @($ok)
}

$rootAbs = Resolve-Root -maybeRoot $Root
if (-not $StageRoot) {
    $StageRoot = Join-Path $rootAbs ("dataset_staged\\" + $Language)
}
$stageAbs = (Resolve-Path -LiteralPath $StageRoot).Path

if (-not $RunTag) {
    $RunTag = "multi_alias_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$workDir = Join-Path $rootAbs ("ml_workspace\\e2e\\" + $Language + "\\general\\" + $RunTag)
$partsDir = Join-Path $workDir "parts"
$mergedCsv = Join-Path $workDir "dataset_merged.csv"
$summaryJson = Join-Path $workDir "run_summary.json"
$bundleRoot = Join-Path $workDir "model_bundle"
$modelsRoot = Join-Path $rootAbs ("assets\\models\\oto_e2e\\" + $Language + "\\by_alias")

New-Item -ItemType Directory -Force -Path $workDir | Out-Null
New-Item -ItemType Directory -Force -Path $partsDir | Out-Null
New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
New-Item -ItemType Directory -Force -Path $modelsRoot | Out-Null

$parsedFormatList = New-Object System.Collections.ArrayList
$fmtTokens = [Regex]::Split([string]$Formats, "[,\s]+")
foreach ($tok in $fmtTokens) {
    $t = ("" + $tok).Trim().ToLowerInvariant()
    if ($t) {
        [void]$parsedFormatList.Add($t)
    }
}
$parsedFormats = @($parsedFormatList.ToArray())
$buildResults = @()

foreach ($fmt in $parsedFormats) {
    $fmtRoot = Join-Path $stageAbs $fmt
    $candidates = @(Get-BankCandidates -formatRoot $fmtRoot -maxCount $MaxBanksPerFormat)
    if ($candidates.Count -eq 0) {
        $buildResults += [PSCustomObject]@{
            format = $fmt
            bank = ""
            status = "skip"
            reason = "no_candidate"
            part_csv = ""
        }
        continue
    }

    $i = 0
    foreach ($bankDir in $candidates) {
        $i += 1
        $bankName = Split-Path -Path $bankDir -Leaf
        $safeName = Normalize-Name -text $bankName
        $partCsv = Join-Path $partsDir ("part_{0}_{1:000}_{2}.csv" -f $fmt, $i, $safeName)
        $summaryPath = [IO.Path]::ChangeExtension($partCsv, ".summary.json")
        $manualOto = Join-Path $bankDir "oto.ini"
        $autoOto = Join-Path $bankDir "oto_auto_ml.ini"

        if (-not (Test-Path -LiteralPath $manualOto)) {
            $buildResults += [PSCustomObject]@{
                format = $fmt
                bank = $bankDir
                status = "skip"
                reason = "manual_oto_missing"
                part_csv = $partCsv
            }
            continue
        }

        $args = @(
            (Join-Path $rootAbs "ml\\scripts\\e2e\\build_e2e_dataset.py"),
            "--language", $Language,
            "--manual-oto", $manualOto,
            "--tg-dir", $bankDir,
            "--wav-dir", $bankDir,
            "--out-csv", $partCsv,
            "--out-summary", $summaryPath,
            "--format-type", $fmt,
            "--auto-oto-policy", "generate-temp",
            "--cv-only", "0"
        )
        if (Test-Path -LiteralPath $autoOto) {
            $args += @("--auto-oto", $autoOto)
        }

        try {
            Write-Host "[Build] fmt=$fmt bank=$bankName"
            & $Python @args | Out-Host
            if ($LASTEXITCODE -ne 0) {
                throw "build_e2e_dataset exit code $LASTEXITCODE"
            }
            $rowsCount = 0
            if (Test-Path -LiteralPath $partCsv) {
                $rowsCount = @(Import-Csv -Path $partCsv -Encoding UTF8).Count
            }
            $buildResults += [PSCustomObject]@{
                format = $fmt
                bank = $bankDir
                status = "ok"
                reason = ""
                rows = $rowsCount
                part_csv = $partCsv
            }
        }
        catch {
            Write-Warning ("[Build-Error] fmt={0} bank={1} reason={2}" -f $fmt, $bankDir, $_.Exception.Message)
            $buildResults += [PSCustomObject]@{
                format = $fmt
                bank = $bankDir
                status = "error"
                reason = $_.Exception.Message
                part_csv = $partCsv
            }
        }
    }
}

$partFiles = @(Get-ChildItem -Path $partsDir -File -Filter "*.csv" | Sort-Object Name)
if ($partFiles.Count -eq 0) {
    Write-Host "[Build-Summary] no parts generated. details:"
    $buildResults | Format-Table -AutoSize | Out-Host
    throw "No part CSV produced. Check dataset_staged paths or bank assets."
}

$allRows = foreach ($f in $partFiles) { Import-Csv -Path $f.FullName -Encoding UTF8 }
$allRows | Export-Csv -Path $mergedCsv -NoTypeInformation -Encoding UTF8

$aliasTypeCounts = $allRows | Group-Object alias_type | Sort-Object Count -Descending | ForEach-Object {
    [PSCustomObject]@{
        alias_type = $_.Name
        count = [int]$_.Count
    }
}
$aliasBucketCounts = $allRows | Group-Object { Get-AliasBucket $_.alias_type } | Sort-Object Count -Descending | ForEach-Object {
    [PSCustomObject]@{
        alias_bucket = $_.Name
        count = [int]$_.Count
    }
}

$trainResults = @()
$evalResults = @()
$groups = @("cv", "vc", "vv")

foreach ($g in $groups) {
    $modelDir = Join-Path $modelsRoot ($g + "\\" + $RunTag)
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    try {
        & $Python (Join-Path $rootAbs "ml\\scripts\\e2e\\train_e2e_model.py") `
            --dataset $mergedCsv `
            --out-dir $modelDir `
            --language $Language `
            --format-type general `
            --cv-only 0 `
            --alias-group $g `
            --ridge 0.01 `
            --delta-cutoff-transform log_clip `
            --cv-subgroup-binary $CVSubgroupBinary `
            --vv-cutoff-loss-weight $VVCutoffLossWeight `
            --vv-positive-cutoff-loss-weight $VVPositiveCutoffLossWeight `
            --cutoff-mode-adjust $CutoffModeAdjust `
            --cutoff-mode-min-key-count $CutoffModeMinKeyCount `
            --cutoff-mode-positive-gate $CutoffModePositiveGate `
            --cutoff-mode-positive-only $CutoffModePositiveOnly | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "train_e2e_model exit code $LASTEXITCODE"
        }
        $trainResults += [PSCustomObject]@{
            alias_group = $g
            status = "ok"
            model_dir = $modelDir
        }

        $bundleGroupDir = Join-Path $bundleRoot $g
        New-Item -ItemType Directory -Force -Path $bundleGroupDir | Out-Null
        Copy-Item -Path (Join-Path $modelDir "*") -Destination $bundleGroupDir -Recurse -Force
    }
    catch {
        $trainResults += [PSCustomObject]@{
            alias_group = $g
            status = "error"
            model_dir = $modelDir
            reason = $_.Exception.Message
        }
        continue
    }

    $outEval = Join-Path $workDir ("eval_e2e_only_" + $g + ".json")
    try {
        & $Python (Join-Path $rootAbs "ml\\scripts\\e2e\\eval_e2e_model.py") `
            --dataset $mergedCsv `
            --model-dir $modelDir `
            --language $Language `
            --format-type general `
            --mode e2e_only `
            --cv-only 0 `
            --alias-group $g `
            --out-json $outEval | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "eval_e2e_model exit code $LASTEXITCODE"
        }
        $metric = Get-Content -Path $outEval -Raw | ConvertFrom-Json
        $evalResults += [PSCustomObject]@{
            alias_group = $g
            mode = "e2e_only"
            rows_eval = [int]$metric.rows_eval
            mae_offset = [double]$metric.mae.offset
            mae_cons = [double]$metric.mae.cons
            mae_cutoff = [double]$metric.mae.cutoff
            mae_pre = [double]$metric.mae.pre
            mae_ovl = [double]$metric.mae.ovl
            out_json = $outEval
        }
    }
    catch {
        $evalResults += [PSCustomObject]@{
            alias_group = $g
            mode = "e2e_only"
            status = "error"
            reason = $_.Exception.Message
            out_json = $outEval
        }
    }
}

$overallE2E = Join-Path $workDir "eval_e2e_only_all_by_alias.json"
$overallHybrid = Join-Path $workDir "eval_hybrid_all_by_alias.json"

& $Python (Join-Path $rootAbs "ml\\scripts\\e2e\\eval_e2e_model.py") `
    --dataset $mergedCsv `
    --model-dir $bundleRoot `
    --language $Language `
    --format-type general `
    --mode e2e_only `
    --cv-only 0 `
    --alias-group all `
    --out-json $overallE2E | Out-Host

& $Python (Join-Path $rootAbs "ml\\scripts\\e2e\\eval_e2e_model.py") `
    --dataset $mergedCsv `
    --model-dir $bundleRoot `
    --language $Language `
    --format-type general `
    --mode hybrid `
    --t-low 0.52 `
    --t-high 0.72 `
    --alpha 0.60 `
    --cv-only 0 `
    --alias-group all `
    --out-json $overallHybrid | Out-Host

$overallE2EObj = Get-Content -Path $overallE2E -Raw | ConvertFrom-Json
$overallHybridObj = Get-Content -Path $overallHybrid -Raw | ConvertFrom-Json

$summary = [PSCustomObject]@{
    run_tag = $RunTag
    language = $Language
    stage_root = $stageAbs
    formats = $parsedFormats
    max_banks_per_format = $MaxBanksPerFormat
    work_dir = $workDir
    merged_csv = $mergedCsv
    model_bundle_root = $bundleRoot
    models_root = $modelsRoot
    part_count = [int]$partFiles.Count
    rows_total = [int]$allRows.Count
    alias_type_counts = $aliasTypeCounts
    alias_bucket_counts = $aliasBucketCounts
    build_results = $buildResults
    train_results = $trainResults
    per_group_eval = $evalResults
    overall_eval_e2e_only = $overallE2EObj
    overall_eval_hybrid = $overallHybridObj
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryJson -Encoding UTF8

Write-Host "[Alias-Batch] run_tag=$RunTag"
Write-Host "[Alias-Batch] merged_csv=$mergedCsv rows=$($allRows.Count) parts=$($partFiles.Count)"
Write-Host "[Alias-Batch] summary=$summaryJson"
