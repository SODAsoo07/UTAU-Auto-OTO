param(
    [string]$SetupExePath = "",
    [string]$InstallRoot = "",
    [string]$RuntimeRoot = "",
    [int]$InstallTimeoutMinutes = 45,
    [int]$SetupMfaTimeoutMinutes = 35,
    [int]$LaunchWaitSeconds = 12,
    [string]$ReportPath = "",
    [switch]$SkipInstall,
    [switch]$SkipSetupMfa,
    [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $InstallRoot = Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO"
}
if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO_v3"
}
if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ReportPath = Join-Path (Get-Location) "sandbox_smoke_report_$stamp.json"
}

$checks = @()
$warnings = @()

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
    $status = if ($Passed) { "PASS" } else { if ($Required) { "FAIL" } else { "WARN" } }
    Write-Host "[$status] $Name - $Value"
    if (-not [string]::IsNullOrWhiteSpace($Detail)) {
        Write-Host "       $Detail"
    }
}

function Add-Warn {
    param([string]$Text)
    $script:warnings += $Text
    Write-Host "[WARN] $Text"
}

function Resolve-SetupExe {
    param([string]$PathHint)
    if (-not [string]::IsNullOrWhiteSpace($PathHint)) {
        $resolved = Resolve-Path -LiteralPath $PathHint -ErrorAction SilentlyContinue
        if ($resolved) {
            return $resolved.Path
        }
        throw "Setup exe not found: $PathHint"
    }
    $candidate = Get-ChildItem -Path (Get-Location) -File -Filter "UTAU_Auto_OTO_Setup_*.exe" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No setup exe found in current directory. Expected: UTAU_Auto_OTO_Setup_*.exe"
    }
    return $candidate.FullName
}

function Wait-ProcessWithTimeout {
    param(
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds
    )
    try {
        Wait-Process -Id $Process.Id -Timeout $TimeoutSeconds -ErrorAction Stop
        $Process.Refresh()
        return $true
    } catch {
        return $false
    }
}

function Safe-StopProcess {
    param([System.Diagnostics.Process]$Process)
    if (-not $Process) { return }
    try {
        if ($Process.HasExited) { return }
    } catch {
        return
    }
    try {
        $null = $Process.CloseMainWindow()
        Start-Sleep -Seconds 2
    } catch {
    }
    try {
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

function Invoke-LauncherHelp {
    param([string]$LauncherPath)
    $result = [pscustomobject]@{
        ExitCode = -1
        Output = ""
    }
    if ([string]::IsNullOrWhiteSpace($LauncherPath) -or -not (Test-Path -LiteralPath $LauncherPath)) {
        $result.Output = "launcher missing"
        return $result
    }
    try {
        $output = & $LauncherPath --help 2>&1 | Out-String
        $result.ExitCode = $LASTEXITCODE
        $result.Output = [string]$output
    } catch {
        $result.ExitCode = -1
        $result.Output = $_.Exception.Message
    }
    return $result
}

function Test-LauncherFailureText {
    param([string]$Text)
    $sample = if ($null -eq $Text) { "" } else { [string]$Text }
    if ([string]::IsNullOrWhiteSpace($sample)) { return $false }
    $lower = $sample.ToLowerInvariant()
    return (
        $lower.Contains("failed to create process") -or
        $lower.Contains("unable to create process") -or
        $lower.Contains("fatal error in launcher") -or
        $lower.Contains("no python at")
    )
}

$setupExeAbs = ""
$installExitCode = $null
$appExe = Join-Path $InstallRoot "UTAU_Auto_OTO\UTAU_Auto_OTO.exe"
$appUiLayout = Join-Path $InstallRoot "UTAU_Auto_OTO\ui\ui_layout.json"
$setupMfaInInstall = Join-Path $InstallRoot "setup_mfa.bat"
$runtimeEnvDir = Join-Path $RuntimeRoot ".env"
$runtimePythonExe = Join-Path $runtimeEnvDir "python.exe"
$runtimePythonExeScripts = Join-Path $runtimeEnvDir "Scripts\python.exe"
$runtimeMfaBat = Join-Path $runtimeEnvDir "Scripts\mfa.bat"
$runtimeMfaExe = Join-Path $runtimeEnvDir "Scripts\mfa.exe"
$runtimeMfaCmd = Join-Path $runtimeEnvDir "Scripts\mfa.cmd"
$runtimeMicromambaDir = Join-Path $RuntimeRoot "micromamba"
$runtimeLogsDir = Join-Path $RuntimeRoot "logs"
$runtimeConfig = Join-Path $RuntimeRoot "config.json"

if (-not $SkipInstall) {
    $setupExeAbs = Resolve-SetupExe -PathHint $SetupExePath
    Add-Check -Name "setup_exe_found" -Passed (Test-Path -LiteralPath $setupExeAbs) -Value $setupExeAbs -Required $true

    $installLog = Join-Path $RuntimeRoot "installer_run.log"
    New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
    $args = @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/CURRENTUSER",
        "/TASKS=runmfa",
        "/LOG=$installLog"
    )
    Write-Host "[INFO] Running installer silently..."
    Write-Host "       $setupExeAbs $($args -join ' ')"

    $proc = Start-Process -FilePath $setupExeAbs -ArgumentList $args -PassThru
    $finished = Wait-ProcessWithTimeout -Process $proc -TimeoutSeconds ($InstallTimeoutMinutes * 60)
    if (-not $finished) {
        Safe-StopProcess -Process $proc
        Add-Check -Name "installer_completed_within_timeout" -Passed $false -Value "$InstallTimeoutMinutes min" -Detail "Installer did not finish in time." -Required $true
    } else {
        $installExitCode = $proc.ExitCode
        Add-Check -Name "installer_exit_code_zero" -Passed ($installExitCode -eq 0) -Value "$installExitCode" -Required $true
    }
} else {
    Add-Warn "SkipInstall enabled. Installer run was skipped."
}

Add-Check -Name "installed_app_exe_exists" -Passed (Test-Path -LiteralPath $appExe) -Value $appExe -Required $true
Add-Check -Name "installed_ui_layout_exists" -Passed (Test-Path -LiteralPath $appUiLayout) -Value $appUiLayout -Required $true
Add-Check -Name "installed_setup_mfa_exists" -Passed (Test-Path -LiteralPath $setupMfaInInstall) -Value $setupMfaInInstall -Required $true

Add-Check -Name "runtime_root_exists" -Passed (Test-Path -LiteralPath $RuntimeRoot) -Value $RuntimeRoot -Required $true
Add-Check -Name "runtime_env_exists" -Passed (Test-Path -LiteralPath $runtimeEnvDir) -Value $runtimeEnvDir -Required $true
Add-Check -Name "runtime_micromamba_exists" -Passed (Test-Path -LiteralPath $runtimeMicromambaDir) -Value $runtimeMicromambaDir -Required $true
Add-Check -Name "runtime_python_exists" -Passed ((Test-Path -LiteralPath $runtimePythonExe) -or (Test-Path -LiteralPath $runtimePythonExeScripts)) -Value $runtimeEnvDir -Required $true

$mfaEntryExists = (Test-Path -LiteralPath $runtimeMfaBat) -or (Test-Path -LiteralPath $runtimeMfaCmd) -or (Test-Path -LiteralPath $runtimeMfaExe)
$mfaEntryValue = if (Test-Path -LiteralPath $runtimeMfaBat) { $runtimeMfaBat } elseif (Test-Path -LiteralPath $runtimeMfaCmd) { $runtimeMfaCmd } elseif (Test-Path -LiteralPath $runtimeMfaExe) { $runtimeMfaExe } else { "" }
Add-Check -Name "runtime_mfa_entry_exists" -Passed $mfaEntryExists -Value $mfaEntryValue -Required $true

if (-not $SkipSetupMfa) {
    if (Test-Path -LiteralPath $setupMfaInInstall) {
        Write-Host "[INFO] Running setup_mfa.bat --non-interactive --install for setup smoke..."
        $setupProc = $null
        try {
            $setupProc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$setupMfaInInstall`" --non-interactive --install") -PassThru
            $setupDone = Wait-ProcessWithTimeout -Process $setupProc -TimeoutSeconds ($SetupMfaTimeoutMinutes * 60)
            if (-not $setupDone) {
                Safe-StopProcess -Process $setupProc
                Add-Check -Name "setup_mfa_non_interactive_completed_within_timeout" -Passed $false -Value "$SetupMfaTimeoutMinutes min" -Detail "setup_mfa.bat did not finish in time." -Required $true
            } else {
                Add-Check -Name "setup_mfa_non_interactive_exit_code_zero" -Passed ($setupProc.ExitCode -eq 0) -Value "$($setupProc.ExitCode)" -Required $true
            }
        } catch {
            Add-Check -Name "setup_mfa_non_interactive_exit_code_zero" -Passed $false -Value $setupMfaInInstall -Detail $_.Exception.Message -Required $true
        } finally {
            Safe-StopProcess -Process $setupProc
        }
    } else {
        Add-Check -Name "setup_mfa_non_interactive_exit_code_zero" -Passed $false -Value $setupMfaInInstall -Detail "setup_mfa.bat missing." -Required $true
    }
} else {
    Add-Warn "SkipSetupMfa enabled. setup_mfa.bat smoke was skipped."
}

$mfaLauncherPath = ""
if (Test-Path -LiteralPath $runtimeMfaBat) {
    $mfaLauncherPath = $runtimeMfaBat
} elseif (Test-Path -LiteralPath $runtimeMfaCmd) {
    $mfaLauncherPath = $runtimeMfaCmd
} elseif (Test-Path -LiteralPath $runtimeMfaExe) {
    $mfaLauncherPath = $runtimeMfaExe
}

if (-not $SkipSetupMfa) {
    if (-not [string]::IsNullOrWhiteSpace($mfaLauncherPath)) {
        $probe = Invoke-LauncherHelp -LauncherPath $mfaLauncherPath
        $launcherFailed = Test-LauncherFailureText -Text $probe.Output
        $probeOk = ($probe.ExitCode -eq 0) -and (-not $launcherFailed)
        $probeDetail = if ($probeOk) { "launcher help ok" } else { ($probe.Output | Out-String).Trim() }
        Add-Check -Name "runtime_mfa_launcher_help_ok" -Passed $probeOk -Value $mfaLauncherPath -Detail $probeDetail -Required $true
    }

    $pythonForCheck = if (Test-Path -LiteralPath $runtimePythonExe) { $runtimePythonExe } elseif (Test-Path -LiteralPath $runtimePythonExeScripts) { $runtimePythonExeScripts } else { "" }
    if (-not [string]::IsNullOrWhiteSpace($pythonForCheck)) {
        try {
            & $pythonForCheck -c "import importlib.util,sys; sys.exit(0 if (importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') or importlib.util.find_spec('montreal_forced_aligner')) else 1)" 2>$null
            $moduleRc = $LASTEXITCODE
            Add-Check -Name "runtime_mfa_python_module_importable" -Passed ($moduleRc -eq 0) -Value $pythonForCheck -Required $true
        } catch {
            Add-Check -Name "runtime_mfa_python_module_importable" -Passed $false -Value $pythonForCheck -Detail $_.Exception.Message -Required $true
        }
    }
} else {
    Add-Warn "SkipSetupMfa enabled. Skipping MFA launcher/module checks."
}

if (-not $SkipLaunch) {
    if (Test-Path -LiteralPath $appExe) {
        Write-Host "[INFO] Launching app for smoke check..."
        $appProc = $null
        try {
            $appProc = Start-Process -FilePath $appExe -PassThru
            Start-Sleep -Seconds ([Math]::Max(3, $LaunchWaitSeconds))
            $stillRunning = $false
            try {
                $appProc.Refresh()
                $stillRunning = -not $appProc.HasExited
            } catch {
                $stillRunning = $true
            }
            $launchDetail = if ($stillRunning) { "Process is running." } else { "Process exited quickly." }
            Add-Check -Name "app_launch_started" -Passed $true -Value "pid=$($appProc.Id)" -Detail $launchDetail -Required $true
        } catch {
            Add-Check -Name "app_launch_started" -Passed $false -Value $appExe -Detail $_.Exception.Message -Required $true
        } finally {
            Safe-StopProcess -Process $appProc
        }
    } else {
        Add-Check -Name "app_launch_started" -Passed $false -Value $appExe -Detail "App executable missing." -Required $true
    }
} else {
    Add-Warn "SkipLaunch enabled. App launch check was skipped."
}

Add-Check -Name "runtime_logs_dir_exists" -Passed (Test-Path -LiteralPath $runtimeLogsDir) -Value $runtimeLogsDir -Required $true
$logFiles = @()
if (Test-Path -LiteralPath $runtimeLogsDir) {
    $logFiles = Get-ChildItem -Path $runtimeLogsDir -File -ErrorAction SilentlyContinue
}
Add-Check -Name "runtime_log_file_created" -Passed ($logFiles.Count -gt 0) -Value "$($logFiles.Count)" -Detail "Expected at least one log file after first launch." -Required $true

$configExists = Test-Path -LiteralPath $runtimeConfig
Add-Check -Name "runtime_config_exists_optional" -Passed $configExists -Value $runtimeConfig -Detail "Config may be created after changing settings; this is optional." -Required $false

$requiredFailures = @($checks | Where-Object { $_.required -and -not $_.passed })
$optionalFailures = @($checks | Where-Object { (-not $_.required) -and (-not $_.passed) })

$report = [pscustomobject]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    machine = $env:COMPUTERNAME
    user = $env:USERNAME
    setup_exe = $setupExeAbs
    skip_install = [bool]$SkipInstall
    skip_launch = [bool]$SkipLaunch
    install_root = $InstallRoot
    runtime_root = $RuntimeRoot
    install_timeout_minutes = $InstallTimeoutMinutes
    launch_wait_seconds = $LaunchWaitSeconds
    installer_exit_code = $installExitCode
    required_failures = $requiredFailures.Count
    optional_failures = $optionalFailures.Count
    warnings = $warnings
    checks = $checks
}

$reportDir = Split-Path -Parent $ReportPath
if (-not [string]::IsNullOrWhiteSpace($reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $ReportPath -Encoding UTF8

Write-Host ""
Write-Host "========== Smoke Check Summary =========="
Write-Host "Required failures: $($requiredFailures.Count)"
Write-Host "Optional failures: $($optionalFailures.Count)"
Write-Host "Report: $ReportPath"

if ($requiredFailures.Count -gt 0) {
    Write-Host "[RESULT] FAIL"
    exit 1
}

if ($optionalFailures.Count -gt 0) {
    Write-Host "[RESULT] PASS (with optional warnings)"
} else {
    Write-Host "[RESULT] PASS"
}
exit 0
