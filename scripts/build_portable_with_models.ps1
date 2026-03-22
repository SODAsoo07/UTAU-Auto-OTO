param(
    [string]$Channel = "stable",
    [string]$SourceDir = "",
    [string]$ModelDir = "",
    [string]$WorkDir = "portable_output",
    [string]$OutputZip = ""
)

$ErrorActionPreference = "Stop"
$channelNorm = $Channel.Trim().ToLowerInvariant()
$channelNorm = if ($channelNorm -eq "default") { "stable" } else { $channelNorm }
if ($channelNorm -notin @("stable", "preview")) {
    throw "Invalid Channel: $Channel (expected: stable/default or preview)"
}
if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = "UTAU_Auto_OTO_Release_$channelNorm"
}
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $channelModelDir = "ML_models_$channelNorm"
    $channelModelAbs = Join-Path (Split-Path -Parent $PSScriptRoot) $channelModelDir
    if (Test-Path $channelModelAbs) {
        $ModelDir = $channelModelDir
    } else {
        $ModelDir = "ML_models"
    }
}
if ([string]::IsNullOrWhiteSpace($OutputZip)) {
    $OutputZip = "portable_output/UTAU_Auto_OTO_portable_with_models_$channelNorm.zip"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceAbs = Join-Path $repoRoot $SourceDir
$modelsAbs = Join-Path $repoRoot $ModelDir
$workAbs = Join-Path $repoRoot $WorkDir
$outputAbs = Join-Path $repoRoot $OutputZip

if (-not (Test-Path $sourceAbs)) {
    throw "Release folder not found: $sourceAbs"
}

if (-not (Test-Path $modelsAbs)) {
    throw "Model directory not found: $modelsAbs"
}
Write-Host "Using model directory: $modelsAbs"

$pointerFiles = Get-ChildItem -Path $modelsAbs -Recurse -File | Where-Object {
    try {
        (Get-Content -Path $_.FullName -TotalCount 1 -ErrorAction Stop) -eq "version https://git-lfs.github.com/spec/v1"
    } catch {
        $false
    }
}
if ($pointerFiles.Count -gt 0) {
    $firstPointer = $pointerFiles[0].FullName
    throw "Detected Git LFS pointer files in ML_models (example: $firstPointer). Run 'git lfs pull' first."
}

$stageRoot = Join-Path $workAbs "UTAU_Auto_OTO_Release_$channelNorm"
$appDir = Join-Path $stageRoot "UTAU_Auto_OTO"
if (Test-Path $stageRoot) {
    Remove-Item $stageRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $workAbs -Force | Out-Null
Copy-Item -Path $sourceAbs -Destination $stageRoot -Recurse -Force
Copy-Item -Path $modelsAbs -Destination (Join-Path $appDir "ML_models") -Recurse -Force

$outputDir = Split-Path -Parent $outputAbs
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
if (Test-Path $outputAbs) {
    Remove-Item $outputAbs -Force
}

Compress-Archive -Path "$stageRoot\*" -DestinationPath $outputAbs -Force
Write-Host "Portable zip with models created: $outputAbs"
