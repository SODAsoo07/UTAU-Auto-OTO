param(
    [ValidateSet("korean", "japanese")]
    [string]$Language = "korean",
    [ValidateSet("mfa", "ctc", "both")]
    [string]$Target = "mfa",
    [string]$SetupScriptPath = "",
    [string]$CtcSetupScriptPath = "",
    [string]$RuntimeRoot = "",
    [switch]$WithMl,
    [switch]$SkipSetup,
    [int]$SetupTimeoutMinutes = 45,
    [ValidateRange(1, 3)]
    [int]$SetupAttempts = 2,
    [string]$ReportPath = "",
    [switch]$InteractiveMenu,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"

$checks = @()
$warnings = @()
$actions = @()
$hints = @()
$recoveryProfile = "cli"
$shouldShowMenu = $false

function Read-MenuChoice {
    param(
        [string]$Prompt,
        [string[]]$Allowed,
        [string]$Default = ""
    )
    $allowedSet = @{}
    foreach ($item in ($Allowed | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
        $allowedSet[$item.Trim().ToLowerInvariant()] = $true
    }

    while ($true) {
        $shownPrompt = if ([string]::IsNullOrWhiteSpace($Default)) { $Prompt } else { "$Prompt [default: $Default]" }
        $raw = Read-Host $shownPrompt
        $value = [string]$raw
        if ([string]::IsNullOrWhiteSpace($value) -and -not [string]::IsNullOrWhiteSpace($Default)) {
            $value = $Default
        }
        $norm = $value.Trim().ToLowerInvariant()
        if ($allowedSet.ContainsKey($norm)) {
            return $norm
        }
        Write-Host "Invalid selection. Allowed values: $($Allowed -join ', ')"
    }
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$Default = $true
    )
    $defaultText = if ($Default) { "y" } else { "n" }
    $choice = Read-MenuChoice -Prompt "$Prompt (y/n)" -Allowed @("y", "n", "yes", "no") -Default $defaultText
    return ($choice -eq "y" -or $choice -eq "yes")
}

function Start-RecoveryMenu {
    Write-Host ""
    Write-Host "==============================================="
    Write-Host " UTAU Auto OTO Runtime Recovery Menu"
    Write-Host "==============================================="
    Write-Host "1) Standard MFA recovery (Korean)"
    Write-Host "2) Full MFA recovery + ML (Korean)"
    Write-Host "3) Standard MFA recovery (Japanese)"
    Write-Host "4) Full MFA recovery + ML (Japanese)"
    Write-Host "5) Diagnose MFA only (Korean)"
    Write-Host "6) Diagnose MFA only (Japanese)"
    Write-Host "7) Custom"
    Write-Host "8) CTC recovery only"
    Write-Host "9) MFA + CTC recovery"
    Write-Host "0) Exit"
    Write-Host ""

    $choice = Read-MenuChoice -Prompt "Choose recovery option number" -Allowed @("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    switch ($choice) {
        "0" {
            Write-Host "Recovery canceled."
            exit 2
        }
        "1" {
            return @{ language = "korean"; target = "mfa"; skip_setup = $false; with_ml = $false; profile = "standard_korean" }
        }
        "2" {
            return @{ language = "korean"; target = "mfa"; skip_setup = $false; with_ml = $true; profile = "full_korean_ml" }
        }
        "3" {
            return @{ language = "japanese"; target = "mfa"; skip_setup = $false; with_ml = $false; profile = "standard_japanese" }
        }
        "4" {
            return @{ language = "japanese"; target = "mfa"; skip_setup = $false; with_ml = $true; profile = "full_japanese_ml" }
        }
        "5" {
            return @{ language = "korean"; target = "mfa"; skip_setup = $true; with_ml = $false; profile = "diagnose_korean" }
        }
        "6" {
            return @{ language = "japanese"; target = "mfa"; skip_setup = $true; with_ml = $false; profile = "diagnose_japanese" }
        }
        "8" {
            return @{ language = "korean"; target = "ctc"; skip_setup = $false; with_ml = $false; profile = "ctc_only" }
        }
        "9" {
            return @{ language = "korean"; target = "both"; skip_setup = $false; with_ml = $false; profile = "mfa_ctc" }
        }
        "7" {
            $langChoice = Read-MenuChoice -Prompt "Language (korean/japanese)" -Allowed @("korean", "japanese") -Default "korean"
            $targetChoice = Read-MenuChoice -Prompt "Target (mfa/ctc/both)" -Allowed @("mfa", "ctc", "both") -Default "mfa"
            $runSetup = Read-YesNo -Prompt "Run setup scripts for selected target?" -Default $true
            $enableMl = $false
            if ($runSetup -and ($targetChoice -eq "mfa" -or $targetChoice -eq "both")) {
                $enableMl = Read-YesNo -Prompt "Include ML package recovery for MFA? (--with-ml)" -Default $false
            }
            return @{ language = $langChoice; target = $targetChoice; skip_setup = (-not $runSetup); with_ml = $enableMl; profile = "custom" }
        }
    }
}

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

function Add-Action {
    param([string]$Text)
    $script:actions += $Text
    Write-Host "[INFO] $Text"
}

function Add-Hint {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) { return }
    if ($script:hints -contains $Text) { return }
    $script:hints += $Text
}

function Resolve-SetupScriptPath {
    param([string]$PathHint)
    if (-not [string]::IsNullOrWhiteSpace($PathHint)) {
        $resolvedHint = Resolve-Path -LiteralPath $PathHint -ErrorAction SilentlyContinue
        if ($resolvedHint -and (Test-Path -LiteralPath $resolvedHint.Path -PathType Leaf)) {
            return $resolvedHint.Path
        }
        return ""
    }

    $candidates = @()
    if ($PSScriptRoot) {
        $candidates += (Join-Path $PSScriptRoot "..\setup_mfa.bat")
        $candidates += (Join-Path $PSScriptRoot "setup_mfa.bat")
    }
    $candidates += (Join-Path (Get-Location) "setup_mfa.bat")

    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidates += (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO_v3\setup_mfa.bat")
        $candidates += (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO\setup_mfa.bat")
    }

    foreach ($candidate in $candidates) {
        try {
            $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
            if ($resolved -and (Test-Path -LiteralPath $resolved.Path -PathType Leaf)) {
                return $resolved.Path
            }
        } catch {
        }
    }
    return ""
}

function Resolve-CtcSetupScriptPath {
    param([string]$PathHint)
    if (-not [string]::IsNullOrWhiteSpace($PathHint)) {
        $resolvedHint = Resolve-Path -LiteralPath $PathHint -ErrorAction SilentlyContinue
        if ($resolvedHint -and (Test-Path -LiteralPath $resolvedHint.Path -PathType Leaf)) {
            return $resolvedHint.Path
        }
        return ""
    }

    $candidates = @()
    if ($PSScriptRoot) {
        $candidates += (Join-Path $PSScriptRoot "..\setup_ctc.bat")
        $candidates += (Join-Path $PSScriptRoot "setup_ctc.bat")
    }
    $candidates += (Join-Path (Get-Location) "setup_ctc.bat")

    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidates += (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO_v3\setup_ctc.bat")
        $candidates += (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO\setup_ctc.bat")
    }

    foreach ($candidate in $candidates) {
        try {
            $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
            if ($resolved -and (Test-Path -LiteralPath $resolved.Path -PathType Leaf)) {
                return $resolved.Path
            }
        } catch {
        }
    }
    return ""
}

function Resolve-RuntimeRootPath {
    param(
        [string]$RuntimeRootHint,
        [string]$ResolvedSetupScript
    )
    if (-not [string]::IsNullOrWhiteSpace($RuntimeRootHint)) {
        $resolved = Resolve-Path -LiteralPath $RuntimeRootHint -ErrorAction SilentlyContinue
        if ($resolved) {
            return $resolved.Path
        }
        return $RuntimeRootHint
    }
    if (-not [string]::IsNullOrWhiteSpace($ResolvedSetupScript)) {
        return Split-Path -Parent $ResolvedSetupScript
    }
    if (Test-Path -LiteralPath (Join-Path (Get-Location) ".env")) {
        return (Get-Location).Path
    }
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        return (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO")
    }
    return (Get-Location).Path
}

function Resolve-PythonInEnv {
    param([string]$EnvDir)
    $pythonCandidates = @(
        (Join-Path $EnvDir "python.exe"),
        (Join-Path $EnvDir "Scripts\python.exe")
    )
    foreach ($candidate in $pythonCandidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return ""
}

function Resolve-CtcEnvDir {
    param([string]$RuntimeRootPath)
    $defaultDir = Join-Path $RuntimeRootPath ".env_ctc"
    if (Test-Path -LiteralPath $defaultDir -PathType Container) {
        return $defaultDir
    }

    $pathFile = Join-Path $RuntimeRootPath ".ctc_env_path"
    if (Test-Path -LiteralPath $pathFile -PathType Leaf) {
        try {
            $candidate = (Get-Content -LiteralPath $pathFile -TotalCount 1 -ErrorAction Stop | Select-Object -First 1)
            $candidate = [string]$candidate
            if (-not [string]::IsNullOrWhiteSpace($candidate)) {
                $candidateAbs = [System.IO.Path]::GetFullPath($candidate.Trim())
                if (Test-Path -LiteralPath $candidateAbs -PathType Container) {
                    return $candidateAbs
                }
            }
        } catch {
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $fallback = Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO_v3\ctc_env"
        if (Test-Path -LiteralPath $fallback -PathType Container) {
            return $fallback
        }
    }
    return $defaultDir
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
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    } catch {
    }
}

function Invoke-SetupScriptRecovery {
    param(
        [string]$Label,
        [string]$ScriptPath,
        [string[]]$SetupArgs,
        [int]$Attempts,
        [int]$TimeoutMinutes
    )
    if ([string]::IsNullOrWhiteSpace($ScriptPath) -or -not (Test-Path -LiteralPath $ScriptPath)) {
        Add-Check -Name "${Label}_setup_script_available" -Passed $false -Value "(missing)" -Detail "Required setup script was not found." -Required $true
        Add-Hint "Missing ${Label} setup script. Check runtime package files."
        return $false
    }

    Add-Check -Name "${Label}_setup_script_available" -Passed $true -Value $ScriptPath -Required $true
    $displayArgs = if ($SetupArgs) { $SetupArgs -join ' ' } else { "" }
    Add-Action "run_${Label}_setup=`"$ScriptPath`" $displayArgs"

    $setupProcessError = ""
    $setupExitCode = -1
    $setupSucceeded = $false
    $setupTimedOut = $false

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        $proc = $null
        try {
            Add-Action "run_${Label}_setup_attempt=$attempt/$Attempts"
            $proc = Start-Process -FilePath $ScriptPath -ArgumentList $SetupArgs -WorkingDirectory (Split-Path -Parent $ScriptPath) -PassThru
            $finished = Wait-ProcessWithTimeout -Process $proc -TimeoutSeconds ($TimeoutMinutes * 60)
            if (-not $finished) {
                $setupTimedOut = $true
                $setupExitCode = 124
                Safe-StopProcess -Process $proc
                Add-Warn "$Label setup attempt $attempt timed out after $TimeoutMinutes minutes."
            } else {
                $setupExitCode = [int]$proc.ExitCode
                if ($setupExitCode -eq 0) {
                    $setupSucceeded = $true
                    break
                }
                Add-Warn "$Label setup attempt $attempt failed with exit code $setupExitCode."
            }
        } catch {
            $setupProcessError = $_.Exception.Message
            break
        } finally {
            Safe-StopProcess -Process $proc
        }
        if (-not $setupSucceeded -and $attempt -lt $Attempts) {
            Start-Sleep -Seconds 2
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($setupProcessError)) {
        Add-Check -Name "${Label}_setup_process_started" -Passed $false -Value $ScriptPath -Detail $setupProcessError -Required $true
        Add-Hint "Failed to launch $Label setup script. Try running it manually from cmd.exe."
        return $false
    }

    if ($setupTimedOut -and -not $setupSucceeded) {
        Add-Check -Name "${Label}_setup_completed_within_timeout" -Passed $false -Value "$TimeoutMinutes min" -Detail "$Label setup did not finish in time." -Required $true
        Add-Hint "$Label setup timed out. Re-run recovery with larger timeout and stable network."
    } else {
        Add-Check -Name "${Label}_setup_completed_within_timeout" -Passed $true -Value "$TimeoutMinutes min" -Required $true
    }
    Add-Check -Name "${Label}_setup_exit_code_zero" -Passed $setupSucceeded -Value "$setupExitCode" -Required $true
    if (-not $setupSucceeded) {
        Add-Hint "$Label setup failed. Check network/proxy/antivirus blocks and rerun."
    }
    return $setupSucceeded
}

function Get-AvailableSpaceGiB {
    param([string]$Path)
    try {
        if ([string]::IsNullOrWhiteSpace($Path)) { return -1 }
        $fullPath = [System.IO.Path]::GetFullPath($Path)
        $root = [System.IO.Path]::GetPathRoot($fullPath)
        if ([string]::IsNullOrWhiteSpace($root)) { return -1 }
        $drive = [System.IO.DriveInfo]::new($root)
        if (-not $drive.IsReady) { return -1 }
        return [math]::Round(($drive.AvailableFreeSpace / 1GB), 2)
    } catch {
        return -1
    }
}

function Test-UrlReachable {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 12
    )
    try {
        $resp = Invoke-WebRequest -Uri $Url -Method Head -TimeoutSec $TimeoutSeconds -UseBasicParsing
        return [pscustomobject]@{
            ok = $true
            detail = "HTTP $([int]$resp.StatusCode)"
        }
    } catch {
        $statusCode = $null
        try {
            if ($_.Exception.Response) {
                $statusCode = [int]$_.Exception.Response.StatusCode
            }
        } catch {
            $statusCode = $null
        }
        if ($statusCode -ge 200 -and $statusCode -lt 500) {
            return [pscustomobject]@{
                ok = $true
                detail = "HTTP $statusCode"
            }
        }
        return [pscustomobject]@{
            ok = $false
            detail = $_.Exception.Message
        }
    }
}

function Invoke-PythonCheck {
    param(
        [string]$PythonExe,
        [string]$Code
    )
    if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe)) {
        return [pscustomobject]@{ ok = $false; rc = -1; output = "python not found" }
    }

    $stdoutPath = ""
    $stderrPath = ""
    $pyScriptPath = ""
    try {
        $pythonDir = Split-Path -Parent $PythonExe
        $pathPrefixParts = @(
            $pythonDir,
            (Join-Path $pythonDir "Scripts"),
            (Join-Path $pythonDir "Library\bin"),
            (Join-Path $pythonDir "Library\usr\bin"),
            (Join-Path $pythonDir "Library\mingw-w64\bin"),
            (Join-Path $pythonDir "bin")
        )
        $pathPrefix = (($pathPrefixParts | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) -join ';')
        $oldPath = [string]$env:PATH
        if (-not [string]::IsNullOrWhiteSpace($pathPrefix)) {
            $env:PATH = "$pathPrefix;$oldPath"
        }

        $tmpBase = [System.IO.Path]::GetTempFileName()
        $pyScriptPath = [System.IO.Path]::ChangeExtension($tmpBase, ".py")
        Move-Item -LiteralPath $tmpBase -Destination $pyScriptPath -Force
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($pyScriptPath, [string]$Code, $utf8NoBom)

        $stdoutPath = [System.IO.Path]::GetTempFileName()
        $stderrPath = [System.IO.Path]::GetTempFileName()

        $proc = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList @($pyScriptPath) `
            -PassThru `
            -Wait `
            -NoNewWindow `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue } else { "" }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue } else { "" }
        $stdoutText = if ($null -eq $stdout) { "" } else { [string]$stdout }
        $stderrText = if ($null -eq $stderr) { "" } else { [string]$stderr }
        $out = "$stdoutText$stderrText"
        $rc = [int]$proc.ExitCode

        return [pscustomobject]@{
            ok = ($rc -eq 0)
            rc = $rc
            output = [string]$out
        }
    } catch {
        return [pscustomobject]@{ ok = $false; rc = -1; output = $_.Exception.Message }
    } finally {
        if ($null -ne $oldPath) {
            $env:PATH = $oldPath
        }
        if ($stdoutPath -and (Test-Path -LiteralPath $stdoutPath)) {
            Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
        }
        if ($stderrPath -and (Test-Path -LiteralPath $stderrPath)) {
            Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue
        }
        if ($pyScriptPath -and (Test-Path -LiteralPath $pyScriptPath)) {
            Remove-Item -LiteralPath $pyScriptPath -Force -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-MfaLauncher {
    param([string]$EnvDir)
    $candidates = @(
        (Join-Path $EnvDir "Scripts\mfa.exe"),
        (Join-Path $EnvDir "Scripts\mfa.bat"),
        (Join-Path $EnvDir "Scripts\mfa.cmd")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    return ""
}

function Test-MfaModelReady {
    param(
        [string]$LauncherPath,
        [string]$RuntimeRootPath,
        [string]$ModelName
    )
    $result = [pscustomobject]@{ ok = $false; source = ""; detail = "" }

    if (-not [string]::IsNullOrWhiteSpace($LauncherPath) -and (Test-Path -LiteralPath $LauncherPath)) {
        try {
            $output = & $LauncherPath model list acoustic 2>&1 | Out-String
            $text = [string]$output
            if ($LASTEXITCODE -eq 0 -and $text.ToLowerInvariant().Contains($ModelName.ToLowerInvariant())) {
                $result.ok = $true
                $result.source = "mfa_cli"
                $result.detail = "model listed by mfa cli"
                return $result
            }
            $result.detail = ($text.Trim())
        } catch {
            $result.detail = $_.Exception.Message
        }
    }

    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($RuntimeRootPath)) {
        $roots += (Join-Path $RuntimeRootPath ".mfa_root_ascii")
    }
    $roots += (Join-Path (Join-Path $env:USERPROFILE "Documents") "MFA")

    foreach ($root in $roots) {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $acousticDir = Join-Path $root "pretrained_models\acoustic"
        $paths = @(
            (Join-Path $acousticDir $ModelName),
            (Join-Path $acousticDir "$ModelName.zip"),
            (Join-Path $acousticDir "$ModelName.yaml"),
            (Join-Path $acousticDir "$ModelName.yml"),
            (Join-Path $acousticDir "$ModelName.meta")
        )
        foreach ($path in $paths) {
            if (Test-Path -LiteralPath $path) {
                $result.ok = $true
                $result.source = "local_artifact"
                $result.detail = $path
                return $result
            }
        }
    }

    if ([string]::IsNullOrWhiteSpace($result.detail)) {
        $result.detail = "model artifact not found"
    }
    return $result
}

$explicitControlParams = @(
    "Language",
    "Target",
    "SetupScriptPath",
    "CtcSetupScriptPath",
    "RuntimeRoot",
    "WithMl",
    "SkipSetup",
    "SetupTimeoutMinutes",
    "ReportPath"
)
$hasExplicitControl = $false
foreach ($key in $explicitControlParams) {
    if ($PSBoundParameters.ContainsKey($key)) {
        $hasExplicitControl = $true
        break
    }
}

$shouldShowMenu = $InteractiveMenu.IsPresent
if (-not $shouldShowMenu -and -not $NonInteractive -and -not $hasExplicitControl) {
    $shouldShowMenu = $true
}

if ($shouldShowMenu) {
    $selection = Start-RecoveryMenu
    $Language = [string]$selection.language
    $Target = [string]$selection.target
    $SkipSetup = [bool]$selection.skip_setup
    $WithMl = [bool]$selection.with_ml
    $recoveryProfile = [string]$selection.profile
    Write-Host ""
    Write-Host "[Selection] profile=$recoveryProfile, target=$Target, language=$Language, skip_setup=$SkipSetup, with_ml=$WithMl"
}

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ReportPath = Join-Path (Get-Location) "runtime_recovery_report_$stamp.json"
}

$targetNorm = [string]$Target
$recoverMfa = ($targetNorm -eq "mfa" -or $targetNorm -eq "both")
$recoverCtc = ($targetNorm -eq "ctc" -or $targetNorm -eq "both")

$setupScriptMfa = Resolve-SetupScriptPath -PathHint $SetupScriptPath
$setupScriptCtc = Resolve-CtcSetupScriptPath -PathHint $CtcSetupScriptPath
$runtimeScriptHint = ""
if ($recoverMfa -and -not [string]::IsNullOrWhiteSpace($setupScriptMfa)) {
    $runtimeScriptHint = $setupScriptMfa
} elseif ($recoverCtc -and -not [string]::IsNullOrWhiteSpace($setupScriptCtc)) {
    $runtimeScriptHint = $setupScriptCtc
}

$runtimeRootAbs = Resolve-RuntimeRootPath -RuntimeRootHint $RuntimeRoot -ResolvedSetupScript $runtimeScriptHint
$runtimeRootAbs = [System.IO.Path]::GetFullPath($runtimeRootAbs)

$mfaEnvDir = Join-Path $runtimeRootAbs ".env"
$ctcEnvDir = Resolve-CtcEnvDir -RuntimeRootPath $runtimeRootAbs
$pythonExeMfa = Resolve-PythonInEnv -EnvDir $mfaEnvDir
$pythonExeCtc = Resolve-PythonInEnv -EnvDir $ctcEnvDir
$launcherPath = Resolve-MfaLauncher -EnvDir $mfaEnvDir
$modelName = if ($Language -eq "japanese") { "japanese_mfa" } else { "korean_mfa" }
$ctcCodeRoot = if (Test-Path -LiteralPath (Join-Path $runtimeRootAbs "UTAU_Auto_OTO\\core\\ctc_runner.py")) {
    Join-Path $runtimeRootAbs "UTAU_Auto_OTO"
} else {
    $runtimeRootAbs
}

Add-Action "target=$targetNorm"
Add-Action "language=$Language"
Add-Action "runtime_root=$runtimeRootAbs"
Add-Action "profile=$recoveryProfile"
Add-Action "ctc_env_dir=$ctcEnvDir"
if ($recoverMfa) {
    if ($setupScriptMfa) { Add-Action "setup_script_mfa=$setupScriptMfa" } else { Add-Warn "setup_mfa.bat was not found." }
}
if ($recoverCtc) {
    if ($setupScriptCtc) { Add-Action "setup_script_ctc=$setupScriptCtc" } else { Add-Warn "setup_ctc.bat was not found." }
}

$availableFreeGiB = Get-AvailableSpaceGiB -Path $runtimeRootAbs
if ($availableFreeGiB -lt 0) {
    Add-Check -Name "disk_space_probe_available" -Passed $false -Value $runtimeRootAbs -Detail "Could not determine free disk space." -Required $false
}

if (-not $SkipSetup) {
    Add-Action "setup_attempts=$SetupAttempts"
    if ($recoverMfa -and $availableFreeGiB -ge 0) {
        $requiredFreeGiBMfa = if ($WithMl) { 12 } else { 7 }
        $mfaDiskOk = ($availableFreeGiB -ge $requiredFreeGiBMfa)
        Add-Check -Name "mfa_disk_space_sufficient" -Passed $mfaDiskOk -Value ("{0} GiB (need >= {1} GiB)" -f $availableFreeGiB, $requiredFreeGiBMfa) -Required $true
        if (-not $mfaDiskOk) { Add-Hint "Not enough free disk space for MFA recovery." }
    }
    if ($recoverCtc -and $availableFreeGiB -ge 0) {
        $requiredFreeGiBCtc = 6
        $ctcDiskOk = ($availableFreeGiB -ge $requiredFreeGiBCtc)
        Add-Check -Name "ctc_disk_space_sufficient" -Passed $ctcDiskOk -Value ("{0} GiB (need >= {1} GiB)" -f $availableFreeGiB, $requiredFreeGiBCtc) -Required $true
        if (-not $ctcDiskOk) { Add-Hint "Not enough free disk space for CTC recovery." }
    }

    if ($recoverMfa) {
        $argsMfa = @("--non-interactive", "--direct-setup", "--install")
        if (-not [string]::IsNullOrWhiteSpace($Language)) {
            $argsMfa += @("--language", $Language)
        }
        if ($WithMl) { $argsMfa += "--with-ml" }
        if (-not [string]::IsNullOrWhiteSpace($runtimeRootAbs)) {
            $argsMfa += @("--runtime-root", $runtimeRootAbs)
        }
        [void](Invoke-SetupScriptRecovery -Label "mfa" -ScriptPath $setupScriptMfa -SetupArgs $argsMfa -Attempts $SetupAttempts -TimeoutMinutes $SetupTimeoutMinutes)
    }
    if ($recoverCtc) {
        $argsCtc = @("--non-interactive")
        if (-not [string]::IsNullOrWhiteSpace($runtimeRootAbs)) {
            $argsCtc += @("--runtime-root", $runtimeRootAbs)
        }
        [void](Invoke-SetupScriptRecovery -Label "ctc" -ScriptPath $setupScriptCtc -SetupArgs $argsCtc -Attempts $SetupAttempts -TimeoutMinutes $SetupTimeoutMinutes)
    }
} else {
    Add-Warn "SkipSetup enabled. Running diagnose-only checks."
}

$pythonExeMfa = Resolve-PythonInEnv -EnvDir $mfaEnvDir
$ctcEnvDir = Resolve-CtcEnvDir -RuntimeRootPath $runtimeRootAbs
$pythonExeCtc = Resolve-PythonInEnv -EnvDir $ctcEnvDir
$launcherPath = Resolve-MfaLauncher -EnvDir $mfaEnvDir

Add-Check -Name "runtime_root_exists" -Passed (Test-Path -LiteralPath $runtimeRootAbs -PathType Container) -Value $runtimeRootAbs -Required $true

if ($recoverMfa) {
    Add-Check -Name "mfa_env_exists" -Passed (Test-Path -LiteralPath $mfaEnvDir -PathType Container) -Value $mfaEnvDir -Required $true
    Add-Check -Name "mfa_python_exists" -Passed (-not [string]::IsNullOrWhiteSpace($pythonExeMfa)) -Value $pythonExeMfa -Required $true
    Add-Check -Name "mfa_launcher_exists" -Passed (-not [string]::IsNullOrWhiteSpace($launcherPath)) -Value $launcherPath -Required $true

    $mfaImportCheck = Invoke-PythonCheck -PythonExe $pythonExeMfa -Code "import importlib.metadata as md,sys; ok=False`ntry:`n md.version('montreal-forced-aligner'); ok=True`nexcept Exception:`n ok=False`nsys.exit(0 if ok else 1)"
    Add-Check -Name "mfa_python_module_importable" -Passed $mfaImportCheck.ok -Value $pythonExeMfa -Detail $mfaImportCheck.output.Trim() -Required $true
    if (-not $mfaImportCheck.ok) {
        Add-Hint "MFA python module import failed. Re-run setup_mfa.bat --non-interactive --install."
    }

    $packagingCheck = Invoke-PythonCheck -PythonExe $pythonExeMfa -Code "import setuptools,pip,wheel"
    Add-Check -Name "mfa_packaging_stack_ready" -Passed $packagingCheck.ok -Value $pythonExeMfa -Detail $packagingCheck.output.Trim() -Required $true

    $audioCheck = Invoke-PythonCheck -PythonExe $pythonExeMfa -Code "import soundfile"
    Add-Check -Name "mfa_audio_dependencies_ready" -Passed $audioCheck.ok -Value $pythonExeMfa -Detail $audioCheck.output.Trim() -Required $true

    if ($Language -eq "japanese") {
        $langCheck = Invoke-PythonCheck -PythonExe $pythonExeMfa -Code "import spacy,sudachipy,sudachidict_core"
    } else {
        $langCheck = Invoke-PythonCheck -PythonExe $pythonExeMfa -Code "import sys,jamo`nok=False`ntry:`n from mecab import MeCab`n ok=True`nexcept Exception:`n try:`n  from mecab import Tagger`n  ok=True`n except Exception:`n  try:`n   import MeCab`n   ok=True`n  except Exception:`n   try:`n    import mecab_ko`n    ok=True`n   except Exception:`n    ok=False`nsys.exit(0 if ok else 1)"
    }
    Add-Check -Name "mfa_${Language}_dependencies_ready" -Passed $langCheck.ok -Value $pythonExeMfa -Detail $langCheck.output.Trim() -Required $true

    $modelCheck = Test-MfaModelReady -LauncherPath $launcherPath -RuntimeRootPath $runtimeRootAbs -ModelName $modelName
    Add-Check -Name "mfa_${Language}_model_ready" -Passed $modelCheck.ok -Value $modelName -Detail $modelCheck.detail -Required $true
}

if ($recoverCtc) {
    Add-Check -Name "ctc_env_exists" -Passed (Test-Path -LiteralPath $ctcEnvDir -PathType Container) -Value $ctcEnvDir -Required $true
    Add-Check -Name "ctc_python_exists" -Passed (-not [string]::IsNullOrWhiteSpace($pythonExeCtc)) -Value $pythonExeCtc -Required $true

    $ctcBaseCheck = Invoke-PythonCheck -PythonExe $pythonExeCtc -Code "import numpy,textgrid,torch,torchaudio"
    Add-Check -Name "ctc_runtime_modules_ready" -Passed $ctcBaseCheck.ok -Value $pythonExeCtc -Detail $ctcBaseCheck.output.Trim() -Required $true
    if (-not $ctcBaseCheck.ok) {
        Add-Hint "CTC runtime modules missing. Re-run setup_ctc.bat --non-interactive."
    }

    $ctcMmsCheck = Invoke-PythonCheck -PythonExe $pythonExeCtc -Code "import torchaudio; _=torchaudio.pipelines.MMS_FA"
    Add-Check -Name "ctc_mms_bundle_ready" -Passed $ctcMmsCheck.ok -Value "MMS_FA" -Detail $ctcMmsCheck.output.Trim() -Required $true
    if (-not $ctcMmsCheck.ok) {
        Add-Hint "CTC MMS bundle is not ready. Re-run setup_ctc.bat with stable network."
    }

    $env:UTOA_RUNTIME_CODE_ROOT = [string]$ctcCodeRoot
    $ctcCoreCheck = Invoke-PythonCheck -PythonExe $pythonExeCtc -Code "import os,sys; root=os.environ.get('UTOA_RUNTIME_CODE_ROOT','').strip(); (root and root not in sys.path) and sys.path.insert(0, root); import core.ctc_runner"
    Remove-Item Env:UTOA_RUNTIME_CODE_ROOT -ErrorAction SilentlyContinue
    Add-Check -Name "ctc_core_runner_importable" -Passed $ctcCoreCheck.ok -Value $ctcCodeRoot -Detail $ctcCoreCheck.output.Trim() -Required $true
    if (-not $ctcCoreCheck.ok) {
        Add-Hint "CTC core import failed. Check runtime root and reinstall CTC runtime."
    }
}

$requiredFailures = @($checks | Where-Object { $_.required -and -not $_.passed })
$optionalFailures = @($checks | Where-Object { (-not $_.required) -and (-not $_.passed) })
$ready = ($requiredFailures.Count -eq 0)

if ($ready) {
    Write-Host ""
    Write-Host "[SUCCESS] Runtime recovery completed and all required checks passed."
} else {
    Write-Host ""
    Write-Host "[FAILED] Runtime recovery still needs attention."
}

if ($hints.Count -gt 0) {
    Write-Host ""
    Write-Host "[NEXT ACTIONS]"
    foreach ($hint in $hints) {
        Write-Host "- $hint"
    }
}

$report = [pscustomobject]@{
    timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
    machine = $env:COMPUTERNAME
    user = $env:USERNAME
    language = $Language
    target = $targetNorm
    recovery_profile = $recoveryProfile
    interactive_mode = [bool]$shouldShowMenu
    setup_script_mfa = $setupScriptMfa
    setup_script_ctc = $setupScriptCtc
    runtime_root = $runtimeRootAbs
    mfa_env_dir = $mfaEnvDir
    ctc_env_dir = $ctcEnvDir
    mfa_python_exe = $pythonExeMfa
    ctc_python_exe = $pythonExeCtc
    mfa_launcher = $launcherPath
    model_name = $modelName
    skip_setup = [bool]$SkipSetup
    with_ml = [bool]$WithMl
    setup_timeout_minutes = $SetupTimeoutMinutes
    setup_attempts = $SetupAttempts
    ready = $ready
    required_failures = $requiredFailures.Count
    optional_failures = $optionalFailures.Count
    checks = $checks
    warnings = $warnings
    actions = $actions
    hints = $hints
}

$reportDir = Split-Path -Parent $ReportPath
if (-not [string]::IsNullOrWhiteSpace($reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}
$report | ConvertTo-Json -Depth 8 | Set-Content -Path $ReportPath -Encoding UTF8
Write-Host "[REPORT] $ReportPath"

if ($ready) {
    exit 0
}
exit 1

