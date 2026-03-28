param(
    [string]$PortableRoot = "",
    [string]$WorkDir = "portable_output",
    [int]$LaunchWaitSeconds = 10,
    [string]$ReportPath = "",
    [switch]$SkipLaunch,
    [switch]$RequireLaunchStability
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PortableRoot)) {
    throw "PortableRoot is required. Example: -PortableRoot UTAU_Auto_OTO_Release_stable"
}

$repoRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepoPath {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

$portableAbs = Resolve-RepoPath $PortableRoot
$workAbs = Resolve-RepoPath $WorkDir
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ReportPath = Join-Path $workAbs "portable_smoke_report_$stamp.json"
}
$reportAbs = Resolve-RepoPath $ReportPath

if (-not (Test-Path -LiteralPath $portableAbs)) {
    throw "Portable root not found: $portableAbs"
}

New-Item -ItemType Directory -Path $workAbs -Force | Out-Null
$koreanPathLabel = [string]::Concat([char]0xD55C, [char]0xAE00, " ", [char]0xACF5, [char]0xBC31)
$testRoot = Join-Path $workAbs ("smoke_path_" + $koreanPathLabel)
if (Test-Path -LiteralPath $testRoot) {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
Copy-Item -Path (Join-Path $portableAbs "*") -Destination $testRoot -Recurse -Force

$checks = @()
function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Value = "",
        [string]$Detail = "",
        [bool]$Required = $true
    )
    $script:checks += [pscustomobject]@{
        name = $Name
        passed = $Passed
        required = $Required
        value = $Value
        detail = $Detail
    }
    $state = if ($Passed) { "PASS" } else { if ($Required) { "FAIL" } else { "WARN" } }
    Write-Host "[$state] $Name - $Value"
    if (-not [string]::IsNullOrWhiteSpace($Detail)) {
        Write-Host "       $Detail"
    }
}

$requiredRelativePaths = @(
    "release_channel.json",
    "setup_mfa.bat",
    "setup_ctc.bat",
    "requirements.txt",
    "requirements-ml.txt",
    "runtime_recovery.ps1",
    "startup_diagnose.ps1",
    "startup_diagnose.bat",
    "Launch_UTAU_Auto_OTO.cmd",
    "UTAU_Auto_OTO\UTAU_Auto_OTO.exe"
)
foreach ($rel in $requiredRelativePaths) {
    $probe = Join-Path $testRoot $rel
    Add-Check -Name ("required_file_" + ($rel -replace "[\\/: ]", "_")) -Passed (Test-Path -LiteralPath $probe) -Value $probe -Required $true
}

$legacyShortcut = Join-Path $testRoot "UTAU_Auto_OTO.lnk"
Add-Check -Name "optional_file_UTAU_Auto_OTO_lnk" -Passed (Test-Path -LiteralPath $legacyShortcut) -Value $legacyShortcut -Required $false

$appExe = Join-Path $testRoot "UTAU_Auto_OTO\UTAU_Auto_OTO.exe"
$setupMfa = Join-Path $testRoot "setup_mfa.bat"

if (Test-Path -LiteralPath $appExe) {
    try {
        $length = (Get-Item -LiteralPath $appExe).Length
        Add-Check -Name "app_exe_nonzero_size" -Passed ($length -gt 0) -Value "$length bytes" -Required $true
    } catch {
        Add-Check -Name "app_exe_nonzero_size" -Passed $false -Value $appExe -Detail $_.Exception.Message -Required $true
    }
}

if (Test-Path -LiteralPath $setupMfa) {
    try {
        $setupText = Get-Content -LiteralPath $setupMfa -Raw
        $probeReady = (
            $setupText -match ":check_korean_tokenizer_ready" -and
            $setupText -match "find_spec\('jamo'\)" -and
            $setupText -match "backends=\('mecab','mecab_ko','MeCab'\)"
        )
        Add-Check -Name "setup_mfa_tokenizer_probe_contains_importlib_util" -Passed $probeReady -Value $setupMfa -Required $true
    } catch {
        Add-Check -Name "setup_mfa_tokenizer_probe_contains_importlib_util" -Passed $false -Value $setupMfa -Detail $_.Exception.Message -Required $true
    }

    $setupRc = -1
    try {
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "call `"$setupMfa`" --help") -Wait -PassThru
        $setupRc = $proc.ExitCode
    } catch {
        Add-Check -Name "setup_mfa_help_exit_zero" -Passed $false -Value $setupMfa -Detail $_.Exception.Message -Required $true
    }
    if ($setupRc -ne -1) {
        Add-Check -Name "setup_mfa_help_exit_zero" -Passed ($setupRc -eq 0) -Value "exit=$setupRc" -Required $true
    }
}

if (-not $SkipLaunch -and (Test-Path -LiteralPath $appExe)) {
    $appProc = $null
    $launchRequired = $RequireLaunchStability.IsPresent
    try {
        $appProc = Start-Process -FilePath $appExe -WorkingDirectory (Split-Path -Parent $appExe) -PassThru
        Start-Sleep -Seconds ([Math]::Max(3, $LaunchWaitSeconds))
        $appProc.Refresh()
        if ($appProc.HasExited) {
            Add-Check -Name "app_process_running_after_wait" -Passed $false -Value ("pid={0}" -f $appProc.Id) -Detail ("Exited quickly with code={0}" -f $appProc.ExitCode) -Required $launchRequired
        } else {
            Add-Check -Name "app_process_running_after_wait" -Passed $true -Value ("pid={0}" -f $appProc.Id) -Detail "Process is still running after launch wait." -Required $launchRequired
        }
    } catch {
        Add-Check -Name "app_process_running_after_wait" -Passed $false -Value $appExe -Detail $_.Exception.Message -Required $launchRequired
    } finally {
        if ($appProc) {
            try {
                if (-not $appProc.HasExited) {
                    Stop-Process -Id $appProc.Id -Force -ErrorAction SilentlyContinue
                }
            } catch {
            }
        }
    }
}

$requiredFailures = @($checks | Where-Object { $_.required -and -not $_.passed })
$result = [ordered]@{
    started_at = (Get-Date).ToString("o")
    portable_root = $portableAbs
    smoke_test_root = $testRoot
    checks = $checks
    required_failures = $requiredFailures.Count
    status = if ($requiredFailures.Count -gt 0) { "failed" } else { "passed" }
    finished_at = (Get-Date).ToString("o")
}

New-Item -ItemType Directory -Path (Split-Path -Parent $reportAbs) -Force | Out-Null
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportAbs -Encoding UTF8
Write-Host "[INFO] portable smoke report: $reportAbs"

if ($result.status -ne "passed") {
    exit 1
}
exit 0
