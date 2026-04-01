param(
    [string]$Channel = "stable",
[string]$SourceDir = "",
[string]$OutputDir = "installer_output",
[ValidateSet("bundled", "online")]
[string]$MfaMode = "online",
[bool]$EmitExternalSetupMfa = $true,
    [switch]$Sign,
    [string]$SignSubject = "SODAsoo",
    [string]$SignToolPath = "",
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
$channelNorm = $Channel.Trim().ToLowerInvariant()
$channelNorm = if ($channelNorm -eq "default") { "stable" } else { $channelNorm }
if ($channelNorm -notin @("stable", "preview")) {
    throw "Invalid Channel: $Channel (expected: stable/default or preview)"
}
if ($MfaMode -eq "online" -and $channelNorm -ne "stable") {
    throw "MfaMode=online is supported only for stable channel."
}
if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = "UTAU_Auto_OTO_Release_$channelNorm"
}

$repoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path

function Sync-FileIfNeeded {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [string]$Label
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

$issPath = Resolve-AbsolutePath -InputPath "installer\UTAU_Auto_OTO.iss" -BasePath $repoRoot
if (-not (Test-Path -LiteralPath $issPath)) {
    throw "Inno Setup script not found: $issPath"
}

$sourceAbs = Resolve-AbsolutePath -InputPath $SourceDir -BasePath $repoRoot
if (-not (Test-Path -LiteralPath $sourceAbs)) {
    throw "Release folder not found: $sourceAbs"
}

$setupMfaSrc = Join-Path $repoRoot "setup_mfa.bat"
$setupMfaDst = Join-Path $sourceAbs "setup_mfa.bat"
Sync-FileIfNeeded -SourcePath $setupMfaSrc -DestinationPath $setupMfaDst -Label "setup_mfa.bat"

$runtimeRecoverySrc = Join-Path $repoRoot "scripts\runtime_recovery.ps1"
$runtimeRecoveryDst = Join-Path $sourceAbs "runtime_recovery.ps1"
Sync-FileIfNeeded -SourcePath $runtimeRecoverySrc -DestinationPath $runtimeRecoveryDst -Label "runtime_recovery.ps1"

$startupDiagnoseSrc = Join-Path $repoRoot "scripts\startup_diagnose.ps1"
$startupDiagnoseDst = Join-Path $sourceAbs "startup_diagnose.ps1"
Sync-FileIfNeeded -SourcePath $startupDiagnoseSrc -DestinationPath $startupDiagnoseDst -Label "startup_diagnose.ps1"

$helperSetupSrc = Join-Path $repoRoot "release_assets\설치_도우미.bat"
$helperSetupDst = Join-Path $sourceAbs "설치_도우미.bat"
Sync-FileIfNeeded -SourcePath $helperSetupSrc -DestinationPath $helperSetupDst -Label "설치_도우미.bat"

$helperDiagnoseSrc = Join-Path $repoRoot "scripts\startup_diagnose.bat"
$helperDiagnoseDst = Join-Path $sourceAbs "startup_diagnose.bat"
Sync-FileIfNeeded -SourcePath $helperDiagnoseSrc -DestinationPath $helperDiagnoseDst -Label "startup_diagnose.bat"

$quickStartSrc = Join-Path $repoRoot "release_assets\먼저 실행.txt"
$quickStartDst = Join-Path $sourceAbs "먼저 실행.txt"
Sync-FileIfNeeded -SourcePath $quickStartSrc -DestinationPath $quickStartDst -Label "먼저 실행.txt"

$requirementsSrc = Join-Path $repoRoot "requirements.txt"
$requirementsDst = Join-Path $sourceAbs "requirements.txt"
Sync-FileIfNeeded -SourcePath $requirementsSrc -DestinationPath $requirementsDst -Label "requirements.txt"

$requirementsMlSrc = Join-Path $repoRoot "requirements-ml.txt"
$requirementsMlDst = Join-Path $sourceAbs "requirements-ml.txt"
Sync-FileIfNeeded -SourcePath $requirementsMlSrc -DestinationPath $requirementsMlDst -Label "requirements-ml.txt"

$requiredPayload = @(
    @{ Name = "Main executable"; Path = (Join-Path $sourceAbs "UTAU_Auto_OTO\\UTAU_Auto_OTO.exe") },
    @{ Name = "MFA setup script"; Path = (Join-Path $sourceAbs "setup_mfa.bat") },
    @{ Name = "Runtime recovery script"; Path = (Join-Path $sourceAbs "runtime_recovery.ps1") },
    @{ Name = "Startup diagnose script"; Path = (Join-Path $sourceAbs "startup_diagnose.ps1") },
    @{ Name = "Startup diagnose launcher"; Path = (Join-Path $sourceAbs "startup_diagnose.bat") },
    @{ Name = "requirements.txt"; Path = (Join-Path $sourceAbs "requirements.txt") },
    @{ Name = "requirements-ml.txt"; Path = (Join-Path $sourceAbs "requirements-ml.txt") },
    @{ Name = "Release channel metadata"; Path = (Join-Path $sourceAbs "release_channel.json") }
)
$missingPayload = @()
foreach ($item in $requiredPayload) {
    if (-not (Test-Path -LiteralPath $item.Path)) {
        $missingPayload += "$($item.Name): $($item.Path)"
    }
}
if ($missingPayload.Count -gt 0) {
    throw "Release payload preflight failed. Missing required files:`n$($missingPayload -join "`n")"
}

$resolvedSignTool = ""
if ($Sign) {
    if (-not [string]::IsNullOrWhiteSpace($SignToolPath)) {
        $candidate = (Resolve-Path -LiteralPath $SignToolPath -ErrorAction SilentlyContinue)
        if (-not $candidate) {
            throw "SignToolPath does not exist: $SignToolPath"
        }
        $resolvedSignTool = $candidate.Path
    } else {
        $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
        if ($cmd) {
            $resolvedSignTool = $cmd.Source
        } else {
            $kitCandidates = Get-ChildItem -Path "${env:ProgramFiles(x86)}\Windows Kits\10\bin" -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending
            foreach ($dir in $kitCandidates) {
                $x64Candidate = Join-Path $dir.FullName "x64\signtool.exe"
                if (Test-Path -LiteralPath $x64Candidate) {
                    $resolvedSignTool = $x64Candidate
                    break
                }
                $x86Candidate = Join-Path $dir.FullName "x86\signtool.exe"
                if (Test-Path -LiteralPath $x86Candidate) {
                    $resolvedSignTool = $x86Candidate
                    break
                }
            }
        }
    }
    if (-not $resolvedSignTool) {
        throw "signtool.exe not found. Install Windows SDK or pass -SignToolPath."
    }
    Write-Host "[INFO] Code signing enabled."
    Write-Host "  signtool: $resolvedSignTool"
    Write-Host "  subject : $SignSubject"
    Write-Host "  ts URL  : $TimestampUrl"
}

function Invoke-CodeSign {
    param(
        [string]$FilePath,
        [string]$StageLabel
    )
    if (-not $Sign) {
        return
    }
    if (-not (Test-Path -LiteralPath $FilePath)) {
        throw "Code sign target not found ($StageLabel): $FilePath"
    }
    Write-Host "[INFO] Signing ($StageLabel): $FilePath"
    $args = @(
        "sign",
        "/fd", "SHA256",
        "/td", "SHA256",
        "/tr", $TimestampUrl,
        "/n", $SignSubject,
        $FilePath
    )
    & $resolvedSignTool @args
    if ($LASTEXITCODE -ne 0) {
        throw "Code signing failed ($StageLabel) with exit code ${LASTEXITCODE}: $FilePath"
    }
}

function Get-FreeSubstDrive {
    $used = @((Get-PSDrive -PSProvider FileSystem).Name)
    $candidates = @("Z", "Y", "X", "W", "V", "U", "T", "S", "R", "Q", "P", "O", "N", "M", "L", "K", "J", "I", "H", "G", "F", "E", "D")
    foreach ($letter in $candidates) {
        if ($used -notcontains $letter) {
            return "${letter}:"
        }
    }
    return ""
}

function New-ShortPathMapping {
    param(
        [string]$TargetPath,
        [string]$Label
    )
    if ([string]::IsNullOrWhiteSpace($TargetPath)) {
        return $null
    }
    if ($TargetPath.StartsWith("\\")) {
        Write-Host "[WARN] Skipping short-path mapping ($Label): UNC path is not supported by SUBST."
        return $null
    }

    $drive = Get-FreeSubstDrive
    if (-not $drive) {
        Write-Host "[WARN] Skipping short-path mapping ($Label): no free drive letter found."
        return $null
    }

    cmd.exe /c "subst $drive `"$TargetPath`"" | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$drive\")) {
        Write-Host "[WARN] Failed to create short-path mapping ($Label): $TargetPath"
        return $null
    }

    Write-Host "[INFO] Using short path mapping ($Label): $drive\ -> $TargetPath"
    return $drive
}

function Remove-ShortPathMapping {
    param(
        [string]$Drive
    )
    if ([string]::IsNullOrWhiteSpace($Drive)) {
        return
    }
    cmd.exe /c "subst $Drive /d" | Out-Null
}

$payloadFiles = @(Get-ChildItem -LiteralPath $sourceAbs -Recurse -File -ErrorAction SilentlyContinue)
if ($MfaMode -eq "online") {
    $bundleRoot = (Join-Path $sourceAbs "mfa_runtime_bundle").ToLowerInvariant()
    $payloadFiles = @($payloadFiles | Where-Object { -not $_.FullName.ToLowerInvariant().StartsWith($bundleRoot) })
}
$payloadSizeBytes = ($payloadFiles | Measure-Object -Property Length -Sum).Sum
if (-not $payloadSizeBytes) {
    $payloadSizeBytes = 0
}
$payloadSizeGiB = [math]::Round(($payloadSizeBytes / 1GB), 2)
Write-Host ("Source payload size: {0} GiB" -f $payloadSizeGiB)
if ($payloadSizeBytes -ge 3900000000) {
    Write-Host "[INFO] Large payload detected. Installer will be emitted as Setup.exe + one or more .bin slice files."
}
$longPathFiles = @($payloadFiles | Where-Object { $_.FullName.Length -gt 260 })
if ($longPathFiles.Count -gt 0) {
    Write-Host ("[WARN] Payload contains {0} files with path length > 260. Enabling short-path fallback for ISCC." -f $longPathFiles.Count)
}

$iconAbs = Join-Path $repoRoot "release_assets\AutoOTO-icon.ico"

$outputAbs = Resolve-AbsolutePath -InputPath $OutputDir -BasePath $repoRoot
New-Item -ItemType Directory -Path $outputAbs -Force | Out-Null

$mainPy = Join-Path $repoRoot "main.py"
$version = "0.0.0"
if (Test-Path $mainPy) {
    $match = Select-String -Path $mainPy -Pattern 'APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($match -and $match.Matches.Count -gt 0) {
        $version = $match.Matches[0].Groups[1].Value
    }
}

$isccCmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
$isccPath = if ($isccCmd) { $isccCmd.Source } else { "" }
if (-not $isccPath) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $isccPath = $candidate
            break
        }
    }
}
if (-not $isccPath) {
    throw "ISCC.exe not found. Install Inno Setup first."
}

Invoke-CodeSign -FilePath (Join-Path $sourceAbs "UTAU_Auto_OTO\\UTAU_Auto_OTO.exe") -StageLabel "payload-exe"

Write-Host "Building installer with version: $version (channel: $channelNorm)"
$outputNameSuffix = ""
$excludeMfaRuntimeBundle = "0"
if ($MfaMode -eq "online") {
    $outputNameSuffix = "_online"
    $excludeMfaRuntimeBundle = "1"
}
Write-Host "MFA packaging mode: $MfaMode"
$sourceForIscc = $sourceAbs
$mappedSourceDrive = $null
$requiresShortSource = ($sourceAbs.Length -gt 120) -or ($longPathFiles.Count -gt 0)
if ($requiresShortSource) {
    $mappedSourceDrive = New-ShortPathMapping -TargetPath $sourceAbs -Label "installer source"
    if ($mappedSourceDrive) {
        $sourceForIscc = $mappedSourceDrive
    }
}

$isccArgs = @(
    "/Qp",
    "/DAppVersion=$version",
    "/DAppChannel=$channelNorm",
    "/DSourceDir=$sourceForIscc",
    "/DOutputDir=$outputAbs",
    "/DOutputNameSuffix=$outputNameSuffix",
    "/DExcludeMfaRuntimeBundle=$excludeMfaRuntimeBundle",
    $issPath
)
if (Test-Path -LiteralPath $iconAbs) {
    Write-Host "Using installer icon: $iconAbs"
    $isccArgs += "/DSetupIconFilePath=$iconAbs"
} else {
    Write-Host "Installer icon not found, setup will use default icon."
}

$isccExit = 0
try {
    & $isccPath $isccArgs
    $isccExit = $LASTEXITCODE
    if ($isccExit -ne 0) {
        Write-Host "[WARN] ISCC quiet compile failed. Re-running once with verbose output for diagnosis..."
        $isccVerboseArgs = @($isccArgs | Where-Object { $_ -ne "/Qp" })
        & $isccPath $isccVerboseArgs
        $isccExit = $LASTEXITCODE
    }
} finally {
    Remove-ShortPathMapping -Drive $mappedSourceDrive
}

if ($isccExit -ne 0) {
    throw "ISCC failed with exit code $isccExit"
}

$baseName = "UTAU_Auto_OTO_Setup_${version}_${channelNorm}${outputNameSuffix}"
$setupExe = Join-Path $outputAbs "$baseName.exe"
$sliceFiles = @(Get-ChildItem -LiteralPath $outputAbs -Filter "$baseName-*.bin" -File -ErrorAction SilentlyContinue | Sort-Object Name)

if (Test-Path -LiteralPath $setupExe) {
    Write-Host "Installer build complete: $setupExe"
    Invoke-CodeSign -FilePath $setupExe -StageLabel "installer-exe"
} else {
    Write-Host "Installer build complete: $outputAbs"
}

if ($sliceFiles.Count -gt 0) {
    Write-Host "[INFO] Disk spanning output detected. Keep all files together when distributing:"
    foreach ($slice in $sliceFiles) {
        $sizeMiB = [math]::Round(($slice.Length / 1MB), 2)
        Write-Host ("  - {0} ({1} MiB)" -f $slice.Name, $sizeMiB)
    }
}

if ($EmitExternalSetupMfa) {
    $setupMfaSrc = Join-Path $repoRoot "setup_mfa.bat"
    if (Test-Path -LiteralPath $setupMfaSrc) {
        $setupMfaOut = Join-Path $outputAbs "setup_mfa.bat"
        $setupMfaVersionedOut = Join-Path $outputAbs ("setup_mfa_{0}_{1}.bat" -f $version, $channelNorm)
        Copy-Item -LiteralPath $setupMfaSrc -Destination $setupMfaOut -Force
        Copy-Item -LiteralPath $setupMfaSrc -Destination $setupMfaVersionedOut -Force
        Write-Host "[INFO] External setup_mfa.bat attached for test override:"
        Write-Host "  - $setupMfaOut"
        Write-Host "  - $setupMfaVersionedOut"
        Write-Host "  (Setup.exe will prefer setup_mfa.bat in the same folder when present.)"
    } else {
        Write-Host "[WARN] setup_mfa.bat not found at repo root; external attachment skipped."
    }

    $runtimeRecoverySrc = Join-Path $repoRoot "scripts\runtime_recovery.ps1"
    if (Test-Path -LiteralPath $runtimeRecoverySrc) {
        $runtimeRecoveryOut = Join-Path $outputAbs "runtime_recovery.ps1"
        $runtimeRecoveryVersionedOut = Join-Path $outputAbs ("runtime_recovery_{0}_{1}.ps1" -f $version, $channelNorm)
        Copy-Item -LiteralPath $runtimeRecoverySrc -Destination $runtimeRecoveryOut -Force
        Copy-Item -LiteralPath $runtimeRecoverySrc -Destination $runtimeRecoveryVersionedOut -Force
        Write-Host "[INFO] External runtime_recovery.ps1 attached:"
        Write-Host "  - $runtimeRecoveryOut"
        Write-Host "  - $runtimeRecoveryVersionedOut"
    } else {
        Write-Host "[WARN] runtime_recovery.ps1 not found at scripts/runtime_recovery.ps1; external attachment skipped."
    }
}
