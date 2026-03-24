param(
    [ValidateSet("full", "lite")]
    [string]$Profile = "lite",
    [ValidateSet("stable", "preview", "default")]
    [string]$Channel = "stable",
    [ValidateSet("nuitka", "pyinstaller")]
    [string]$Backend = "nuitka",
    [string]$AppName = "UTAU_Auto_OTO",
    [string]$ReleaseDir = "",
    [string]$OutputDir = "installer_output",
    [string]$EnvDir = ".env",
    [string]$MfaRootDir = ".mfa_root_ascii",
    [string]$MlModelsDir = "ML_models",
    [switch]$SkipBuild,
    [switch]$SkipInstaller,
    [switch]$SkipDeps,
    [switch]$NoMlModels,
    [switch]$NoMicromamba,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "== $Text =="
}

function Normalize-Channel {
    param([string]$Raw)
    $value = ($Raw | ForEach-Object { "$_".Trim().ToLowerInvariant() })
    if ($value -eq "default") { return "stable" }
    return $value
}

function Invoke-ProcessChecked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )
    $argText = ($Arguments -join " ")
    if ($DryRun) {
        Write-Host "[DRYRUN] $FilePath $argText"
        return
    }
    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $FilePath $argText (exit=$LASTEXITCODE)"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-RobocopyChecked {
    param(
        [string]$Source,
        [string]$Destination,
        [string[]]$Options
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Source path not found: $Source"
    }
    if ($DryRun) {
        Write-Host "[DRYRUN] robocopy `"$Source`" `"$Destination`" $($Options -join ' ')"
        return
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy $Source $Destination @Options | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -gt 7) {
        throw "robocopy failed: $Source -> $Destination (exit=$exitCode)"
    }
}

function Resolve-FirstExistingPath {
    param([string[]]$Candidates)
    foreach ($candidate in ($Candidates | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return ""
}

function Remove-IfExists {
    param([string]$Path)
    if ($DryRun) {
        if (Test-Path -LiteralPath $Path) {
            Write-Host "[DRYRUN] Remove-Item -Recurse -Force `"$Path`""
        }
        return
    }
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

$channelNorm = Normalize-Channel -Raw $Channel
if ($channelNorm -notin @("stable", "preview")) {
    throw "Invalid channel: $Channel"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ReleaseDir)) {
    $ReleaseDir = "UTAU_Auto_OTO_Release_$channelNorm"
}
$releaseAbs = Join-Path $repoRoot $ReleaseDir
$appAbs = Join-Path $releaseAbs $AppName
$bundleAbs = Join-Path $releaseAbs "mfa_runtime_bundle"
$installerScript = Join-Path $repoRoot "scripts\build_installer.ps1"
$includeMlModels = (-not $NoMlModels)
$includeMicromamba = ($Profile -eq "full" -and -not $NoMicromamba)

Write-Host "Profile               : $Profile"
Write-Host "Channel               : $channelNorm"
Write-Host "Backend               : $Backend"
Write-Host "ReleaseDir            : $ReleaseDir"
Write-Host "Include ML models     : $includeMlModels"
Write-Host "Include micromamba    : $includeMicromamba"
Write-Host "DryRun                : $DryRun"

if (-not $SkipBuild) {
    Write-Step "Build app"
    $buildArgs = @(
        "build.py",
        "--name", $AppName,
        "--backend", $Backend,
        "--channel", $channelNorm
    )
    if ($SkipDeps) {
        $buildArgs += "--skip-deps"
    }
    Invoke-ProcessChecked -FilePath "python" -Arguments $buildArgs -WorkingDirectory $repoRoot
}

if (-not (Test-Path -LiteralPath $releaseAbs)) {
    throw "Release folder not found: $releaseAbs"
}
if (-not (Test-Path -LiteralPath $appAbs)) {
    throw "App folder not found in release: $appAbs"
}

Write-Step "Prepare bundle root"
Remove-IfExists -Path $bundleAbs
if (-not $DryRun) {
    New-Item -ItemType Directory -Path $bundleAbs -Force | Out-Null
}

Write-Step "Copy MFA model payload"
$mfaPretrainedSource = Resolve-FirstExistingPath -Candidates @(
    (Join-Path $repoRoot "$MfaRootDir\pretrained_models"),
    (Join-Path $repoRoot "$EnvDir\.mfa_root_ascii\pretrained_models")
)
$mfaAcousticSource = Resolve-FirstExistingPath -Candidates @(
    (Join-Path $repoRoot "$MfaRootDir\acoustic"),
    (Join-Path $repoRoot "$EnvDir\.mfa_root_ascii\acoustic")
)

if ($mfaPretrainedSource) {
    $target = Join-Path $bundleAbs ".mfa_root_ascii\pretrained_models"
    Invoke-RobocopyChecked -Source $mfaPretrainedSource -Destination $target -Options @("/E", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
} elseif ($mfaAcousticSource) {
    $target = Join-Path $bundleAbs ".mfa_root_ascii\acoustic"
    Invoke-RobocopyChecked -Source $mfaAcousticSource -Destination $target -Options @("/E", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
} else {
    Write-Host "[WARN] No MFA model source found (.mfa_root_ascii / .env\\.mfa_root_ascii)."
}

if ($Profile -eq "full") {
    Write-Step "Copy full MFA runtime (.env)"
    $envSource = Join-Path $repoRoot $EnvDir
    if (-not (Test-Path -LiteralPath $envSource)) {
        throw "Runtime env source not found: $envSource"
    }
    $envTarget = Join-Path $bundleAbs ".env"
    Invoke-RobocopyChecked -Source $envSource -Destination $envTarget -Options @(
        "/E", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
        "/XD", "__pycache__", ".pytest_cache", ".mfa_root_ascii", "logs"
    )
}

if ($includeMicromamba) {
    Write-Step "Copy micromamba runtime"
    $micromambaSource = Join-Path $repoRoot "micromamba"
    if (Test-Path -LiteralPath $micromambaSource) {
        $micromambaTarget = Join-Path $bundleAbs "micromamba"
        Invoke-RobocopyChecked -Source $micromambaSource -Destination $micromambaTarget -Options @(
            "/E", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
            "/XD", "__pycache__", "pkgs\cache"
        )
    } else {
        Write-Host "[WARN] micromamba directory not found, skipping."
    }
}

if ($includeMlModels) {
    Write-Step "Copy ML model bundle"
    $mlSource = Join-Path $repoRoot $MlModelsDir
    if (-not (Test-Path -LiteralPath $mlSource)) {
        throw "ML model directory not found: $mlSource"
    }
    $mlTarget = Join-Path $appAbs "ML_models"
    Remove-IfExists -Path $mlTarget
    Invoke-RobocopyChecked -Source $mlSource -Destination $mlTarget -Options @("/E", "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
}

Write-Step "Write bundle manifest"
$manifestPath = Join-Path $bundleAbs "bundle_manifest.json"
$manifest = [ordered]@{
    bundle_kind = if ($Profile -eq "full") { "runtime_full" } else { "models_minimal" }
    profile = $Profile
    channel = $channelNorm
    app_name = $AppName
    created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    includes = [ordered]@{
        env = ($Profile -eq "full")
        mfa_models = $true
        ml_models = $includeMlModels
        micromamba = $includeMicromamba
    }
}
if ($DryRun) {
    Write-Host "[DRYRUN] Write manifest: $manifestPath"
} else {
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
}

if (-not $SkipInstaller) {
    Write-Step "Build installer"
    if (-not (Test-Path -LiteralPath $installerScript)) {
        throw "Installer script not found: $installerScript"
    }
    $installerArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $installerScript,
        "-Channel", $channelNorm,
        "-SourceDir", $ReleaseDir,
        "-MfaMode", "bundled",
        "-OutputDir", $OutputDir
    )
    if ($DryRun) {
        Write-Host "[DRYRUN] powershell $($installerArgs -join ' ')"
    } else {
        & powershell @installerArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Installer build failed (exit=$LASTEXITCODE)"
        }
    }
}

Write-Step "Done"
Write-Host "Release folder : $releaseAbs"
Write-Host "Bundle folder  : $bundleAbs"
if (-not $SkipInstaller) {
    Write-Host "Installer out  : $(Join-Path $repoRoot $OutputDir)"
}

