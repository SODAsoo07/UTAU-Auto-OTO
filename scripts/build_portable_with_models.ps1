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
function Resolve-AbsolutePath {
    param(
        [string]$InputPath,
        [string]$BasePath
    )
    if ([string]::IsNullOrWhiteSpace($InputPath)) {
        return ""
    }
    $candidate = if ([System.IO.Path]::IsPathRooted($InputPath)) {
        $InputPath
    } else {
        Join-Path $BasePath $InputPath
    }
    $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
    if ($resolved) {
        return $resolved.Path
    }
    return [System.IO.Path]::GetFullPath($candidate)
}

$sourceAbs = Resolve-AbsolutePath -InputPath $SourceDir -BasePath $repoRoot
$modelsAbs = Resolve-AbsolutePath -InputPath $ModelDir -BasePath $repoRoot
$workAbs = Resolve-AbsolutePath -InputPath $WorkDir -BasePath $repoRoot
$outputAbs = Resolve-AbsolutePath -InputPath $OutputZip -BasePath $repoRoot

function New-PortableTopShortcut {
    param(
        [Parameter(Mandatory = $true)][string]$RootDir,
        [string]$AppFolder = "UTAU_Auto_OTO",
        [string]$ExeName = "UTAU_Auto_OTO.exe",
        [string]$ShortcutName = "UTAU_Auto_OTO.lnk"
    )

    $targetExe = Join-Path (Join-Path $RootDir $AppFolder) $ExeName
    if (-not (Test-Path -LiteralPath $targetExe)) {
        throw "Shortcut target executable not found: $targetExe"
    }

    $shortcutPath = Join-Path $RootDir $ShortcutName
    if (Test-Path -LiteralPath $shortcutPath) {
        Remove-Item -LiteralPath $shortcutPath -Force
    }

    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $targetExe
    $shortcut.WorkingDirectory = Split-Path -Parent $targetExe
    $shortcut.IconLocation = "$targetExe,0"
    $shortcut.Description = "Launch UTAU Auto OTO"
    $shortcut.Save()

    if (-not (Test-Path -LiteralPath $shortcutPath)) {
        throw "Failed to create top-level shortcut: $shortcutPath"
    }
    return $shortcutPath
}

function Sync-ReleaseFileIfNeeded {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$DestinationPath,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not (Test-Path -LiteralPath $SourcePath)) {
        Write-Host "[WARN] $Label not found; sync skipped:"
        Write-Host "  - $SourcePath"
        return
    }

    $needsSync = $true
    if (Test-Path -LiteralPath $DestinationPath) {
        try {
            $srcHash = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash
            $dstHash = (Get-FileHash -LiteralPath $DestinationPath -Algorithm SHA256).Hash
            if ($srcHash -eq $dstHash) {
                $needsSync = $false
            }
        } catch {
            $needsSync = $true
        }
    }

    if ($needsSync) {
        Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
        Write-Host "[INFO] Synced latest $Label into release payload source:"
        Write-Host "  - $DestinationPath"
    }
}

function Remove-InternalTestScriptsFromPayload {
    param(
        [Parameter(Mandatory = $true)][string]$RootDir
    )

    $blockedNames = @(
        "build_alignment_test_folder.py",
        "compare_alignment_visual.py",
        "export_textgrid_to_sinsy_lab.py",
        "preprocess_oto_cv_for_sequence_training.py",
        "preprocess_sinsy_labels_for_sequence_training.py",
        "train_sequence_aligner_profile_from_sinsy.py",
        "sandbox_smoke_check.ps1"
    ) | ForEach-Object { $_.ToLowerInvariant() }
    $allowedScriptNames = @(
        "runtime_recovery.ps1",
        "startup_diagnose.ps1",
        "startup_diagnose.bat"
    ) | ForEach-Object { $_.ToLowerInvariant() }
    $scriptExts = @(".py", ".ps1", ".bat", ".cmd")

    $roots = @($RootDir, (Join-Path $RootDir "UTAU_Auto_OTO")) | Where-Object { Test-Path -LiteralPath $_ }
    $removed = @()
    foreach ($base in $roots) {
        $files = Get-ChildItem -LiteralPath $base -Recurse -File -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            $name = $f.Name.ToLowerInvariant()
            $rel = $f.FullName.Substring($base.Length).TrimStart('\')
            $ext = [System.IO.Path]::GetExtension($name).ToLowerInvariant()
            $isRootLevel = -not $rel.Contains('\')
            $isScriptsChild = $rel.StartsWith("scripts\", [System.StringComparison]::OrdinalIgnoreCase)
            $remove = $false
            if ($isRootLevel -and ($blockedNames -contains $name)) {
                $remove = $true
            } elseif ($isScriptsChild -and ($scriptExts -contains $ext) -and ($allowedScriptNames -notcontains $name)) {
                $remove = $true
            }
            if (-not $remove) {
                continue
            }
            Remove-Item -LiteralPath $f.FullName -Force -ErrorAction SilentlyContinue
            $removed += $f.FullName
        }
        $scriptsDir = Join-Path $base "scripts"
        if ((Test-Path -LiteralPath $scriptsDir) -and ((Get-ChildItem -LiteralPath $scriptsDir -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)) {
            Remove-Item -LiteralPath $scriptsDir -Force -ErrorAction SilentlyContinue
        }
    }

    if ($removed.Count -gt 0) {
        Write-Host "Pruned internal/test scripts from portable payload:"
        foreach ($path in $removed) {
            Write-Host "  - $path"
        }
    }
}

if (-not (Test-Path $sourceAbs)) {
    throw "Release folder not found: $sourceAbs"
}

if (-not (Test-Path $modelsAbs)) {
    throw "Model directory not found: $modelsAbs"
}
Write-Host "Using model directory: $modelsAbs"

$requirementsSrc = Join-Path $repoRoot "requirements.txt"
$requirementsDst = Join-Path $sourceAbs "requirements.txt"
Sync-ReleaseFileIfNeeded -SourcePath $requirementsSrc -DestinationPath $requirementsDst -Label "requirements.txt"

$requirementsMlSrc = Join-Path $repoRoot "requirements-ml.txt"
$requirementsMlDst = Join-Path $sourceAbs "requirements-ml.txt"
Sync-ReleaseFileIfNeeded -SourcePath $requirementsMlSrc -DestinationPath $requirementsMlDst -Label "requirements-ml.txt"

$requiredReleaseFiles = @(
    "release_channel.json",
    "setup_mfa.bat",
    "requirements.txt",
    "requirements-ml.txt",
    "runtime_recovery.ps1",
    "startup_diagnose.ps1",
    "startup_diagnose.bat",
    "UTAU_Auto_OTO/UTAU_Auto_OTO.exe"
)
$missingReleaseFiles = @()
foreach ($relativePath in $requiredReleaseFiles) {
    $probe = Join-Path $sourceAbs $relativePath
    if (-not (Test-Path $probe)) {
        $missingReleaseFiles += $probe
    }
}
if ($missingReleaseFiles.Count -gt 0) {
    $lines = $missingReleaseFiles -join [Environment]::NewLine
    throw "Release preflight failed. Missing required files:`n$lines"
}

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
Remove-InternalTestScriptsFromPayload -RootDir $stageRoot

$shortcutPath = New-PortableTopShortcut -RootDir $stageRoot
Write-Host "Created top-level shortcut: $shortcutPath"

$modelFileCount = @(Get-ChildItem -Path (Join-Path $appDir "ML_models") -Recurse -File -ErrorAction SilentlyContinue).Count
if ($modelFileCount -le 0) {
    throw "Model copy verification failed: no files found under $appDir\\ML_models"
}
Write-Host "Model copy verification passed: $modelFileCount files"

$outputDir = Split-Path -Parent $outputAbs
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
if (Test-Path $outputAbs) {
    Remove-Item $outputAbs -Force
}

Compress-Archive -Path "$stageRoot\*" -DestinationPath $outputAbs -Force
Write-Host "Portable zip with models created: $outputAbs"
