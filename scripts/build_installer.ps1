param(
    [string]$Channel = "stable",
    [string]$SourceDir = "",
    [string]$OutputDir = "installer_output",
    [ValidateSet("bundled", "online")]
    [string]$MfaMode = "bundled",
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

$repoRoot = Split-Path -Parent $PSScriptRoot
$issPath = Join-Path $repoRoot "installer\UTAU_Auto_OTO.iss"
if (-not (Test-Path $issPath)) {
    throw "Inno Setup script not found: $issPath"
}

$sourceAbs = Join-Path $repoRoot $SourceDir
if (-not (Test-Path $sourceAbs)) {
    throw "Release folder not found: $sourceAbs"
}

$setupMfaSrc = Join-Path $repoRoot "setup_mfa.bat"
$setupMfaDst = Join-Path $sourceAbs "setup_mfa.bat"
if (Test-Path -LiteralPath $setupMfaSrc) {
    $needsSync = $true
    if (Test-Path -LiteralPath $setupMfaDst) {
        try {
            $srcHash = (Get-FileHash -LiteralPath $setupMfaSrc -Algorithm SHA256).Hash
            $dstHash = (Get-FileHash -LiteralPath $setupMfaDst -Algorithm SHA256).Hash
            if ($srcHash -eq $dstHash) {
                $needsSync = $false
            }
        } catch {
            $needsSync = $true
        }
    }
    if ($needsSync) {
        Copy-Item -LiteralPath $setupMfaSrc -Destination $setupMfaDst -Force
        Write-Host "[INFO] Synced latest setup_mfa.bat into release payload source:"
        Write-Host "  - $setupMfaDst"
    }
}

$runtimeRecoverySrc = Join-Path $repoRoot "scripts\runtime_recovery.ps1"
$runtimeRecoveryDst = Join-Path $sourceAbs "runtime_recovery.ps1"
if (Test-Path -LiteralPath $runtimeRecoverySrc) {
    $needsRuntimeRecoverySync = $true
    if (Test-Path -LiteralPath $runtimeRecoveryDst) {
        try {
            $srcHash = (Get-FileHash -LiteralPath $runtimeRecoverySrc -Algorithm SHA256).Hash
            $dstHash = (Get-FileHash -LiteralPath $runtimeRecoveryDst -Algorithm SHA256).Hash
            if ($srcHash -eq $dstHash) {
                $needsRuntimeRecoverySync = $false
            }
        } catch {
            $needsRuntimeRecoverySync = $true
        }
    }
    if ($needsRuntimeRecoverySync) {
        Copy-Item -LiteralPath $runtimeRecoverySrc -Destination $runtimeRecoveryDst -Force
        Write-Host "[INFO] Synced latest runtime_recovery.ps1 into release payload source:"
        Write-Host "  - $runtimeRecoveryDst"
    }
} else {
    Write-Host "[WARN] runtime_recovery.ps1 not found at scripts/runtime_recovery.ps1; sync skipped."
}

$requiredPayload = @(
    @{ Name = "Main executable"; Path = (Join-Path $sourceAbs "UTAU_Auto_OTO\\UTAU_Auto_OTO.exe") },
    @{ Name = "MFA setup script"; Path = (Join-Path $sourceAbs "setup_mfa.bat") },
    @{ Name = "Runtime recovery script"; Path = (Join-Path $sourceAbs "runtime_recovery.ps1") },
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

$iconAbs = Join-Path $repoRoot "release_assets\AutoOTO-icon.ico"

$outputAbs = Join-Path $repoRoot $OutputDir
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

$isccArgs = @(
    "/Qp",
    "/DAppVersion=$version",
    "/DAppChannel=$channelNorm",
    "/DSourceDir=$sourceAbs",
    "/DOutputDir=$outputAbs",
    "/DOutputNameSuffix=$outputNameSuffix",
    "/DExcludeMfaRuntimeBundle=$excludeMfaRuntimeBundle",
    $issPath
)
if (Test-Path $iconAbs) {
    Write-Host "Using installer icon: $iconAbs"
} else {
    Write-Host "Installer icon not found, setup will use default icon."
}
& $isccPath $isccArgs

if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed with exit code $LASTEXITCODE"
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
