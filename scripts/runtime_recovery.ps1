param(
    [ValidateSet("korean", "japanese")]
    [string]$Language = "korean",
    [string]$SetupScriptPath = "",
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
    Write-Host "1) Standard recovery (Korean)"
    Write-Host "2) Full recovery + ML (Korean)"
    Write-Host "3) Standard recovery (Japanese)"
    Write-Host "4) Full recovery + ML (Japanese)"
    Write-Host "5) Diagnose only (Korean)"
    Write-Host "6) Diagnose only (Japanese)"
    Write-Host "7) Custom"
    Write-Host "0) Exit"
    Write-Host ""

    $choice = Read-MenuChoice -Prompt "Choose recovery option number" -Allowed @("0","1","2","3","4","5","6","7")
    switch ($choice) {
        "0" {
            Write-Host "Recovery canceled."
            exit 2
        }
        "1" {
            return @{ language = "korean"; skip_setup = $false; with_ml = $false; profile = "standard_korean" }
        }
        "2" {
            return @{ language = "korean"; skip_setup = $false; with_ml = $true; profile = "full_korean_ml" }
        }
        "3" {
            return @{ language = "japanese"; skip_setup = $false; with_ml = $false; profile = "standard_japanese" }
        }
        "4" {
            return @{ language = "japanese"; skip_setup = $false; with_ml = $true; profile = "full_japanese_ml" }
        }
        "5" {
            return @{ language = "korean"; skip_setup = $true; with_ml = $false; profile = "diagnose_korean" }
        }
        "6" {
            return @{ language = "japanese"; skip_setup = $true; with_ml = $false; profile = "diagnose_japanese" }
        }
        "7" {
            $langChoice = Read-MenuChoice -Prompt "Language (korean/japanese)" -Allowed @("korean", "japanese") -Default "korean"
            $runSetup = Read-YesNo -Prompt "Run setup_mfa.bat recovery stage?" -Default $true
            $enableMl = $false
            if ($runSetup) {
                $enableMl = Read-YesNo -Prompt "Include ML package recovery? (--with-ml)" -Default $false
            }
            return @{ language = $langChoice; skip_setup = (-not $runSetup); with_ml = $enableMl; profile = "custom" }
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
    "SetupScriptPath",
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
    $SkipSetup = [bool]$selection.skip_setup
    $WithMl = [bool]$selection.with_ml
    $recoveryProfile = [string]$selection.profile
    Write-Host ""
    Write-Host "[Selection] profile=$recoveryProfile, language=$Language, skip_setup=$SkipSetup, with_ml=$WithMl"
}

if ([string]::IsNullOrWhiteSpace($ReportPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ReportPath = Join-Path (Get-Location) "runtime_recovery_report_$stamp.json"
}

$setupScript = Resolve-SetupScriptPath -PathHint $SetupScriptPath
$runtimeRootAbs = Resolve-RuntimeRootPath -RuntimeRootHint $RuntimeRoot -ResolvedSetupScript $setupScript
$runtimeRootAbs = [System.IO.Path]::GetFullPath($runtimeRootAbs)

$envDir = Join-Path $runtimeRootAbs ".env"
$pythonCandidates = @(
    (Join-Path $envDir "python.exe"),
    (Join-Path $envDir "Scripts\python.exe")
)
$pythonExe = ""
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $pythonExe = $candidate
        break
    }
}

$launcherPath = Resolve-MfaLauncher -EnvDir $envDir
$modelName = if ($Language -eq "japanese") { "japanese_mfa" } else { "korean_mfa" }

Add-Action "language=$Language"
Add-Action "runtime_root=$runtimeRootAbs"
Add-Action "profile=$recoveryProfile"
if ($setupScript) {
    Add-Action "setup_script=$setupScript"
} else {
    Add-Warn "setup_mfa.bat was not found."
}

if (-not $SkipSetup) {
    if (-not $setupScript) {
        Add-Check -Name "setup_script_available" -Passed $false -Value "(missing)" -Detail "Provide -SetupScriptPath or place setup_mfa.bat next to the app." -Required $true
        Add-Hint "setup_mfa.bat not found. Run the installer again or pass -SetupScriptPath explicitly."
    } else {
        Add-Check -Name "setup_script_available" -Passed $true -Value $setupScript -Required $true
        Add-Action "setup_attempts=$SetupAttempts"
        $requiredFreeGiB = if ($WithMl) { 12 } else { 7 }
        $availableFreeGiB = Get-AvailableSpaceGiB -Path $runtimeRootAbs
        $diskSufficient = $true
        if ($availableFreeGiB -ge 0) {
            $diskSufficient = ($availableFreeGiB -ge $requiredFreeGiB)
            Add-Check -Name "disk_space_sufficient" -Passed $diskSufficient -Value ("{0} GiB (need >= {1} GiB)" -f $availableFreeGiB, $requiredFreeGiB) -Required $true
        } else {
            Add-Check -Name "disk_space_probe_available" -Passed $false -Value $runtimeRootAbs -Detail "Could not determine free disk space." -Required $false
        }
        if (-not $diskSufficient) {
            Add-Hint "Not enough free disk space for setup. Free up space and rerun recovery."
        }

        $probeTargets = @(
            "https://micro.mamba.pm/api/micromamba/win-64/latest",
            "https://api.anaconda.org/package/conda-forge/micromamba/files"
        )
        $probeSummaries = @()
        $networkReachable = $false
        foreach ($target in $probeTargets) {
            $probe = Test-UrlReachable -Url $target -TimeoutSeconds 12
            $probeSummaries += ("{0} => {1}" -f $target, $probe.detail)
            if ($probe.ok) {
                $networkReachable = $true
            }
        }
        Add-Check -Name "network_probe" -Passed $networkReachable -Value ($probeSummaries -join " | ") -Required $false
        if (-not $networkReachable) {
            Add-Hint "Network probe failed. Check TLS/proxy/firewall settings before rerunning setup."
        }

        $args = @("--non-interactive", "--direct-setup", "--install")
        if ($WithMl) { $args += "--with-ml" }
        $cmdLine = "`"$setupScript`" $($args -join ' ')"
        if (-not $diskSufficient) {
            Add-Warn "Skipping setup run because disk space is insufficient."
        } else {
            Add-Action "run_setup=$cmdLine"
            $setupProcessError = ""
            $setupExitCode = -1
            $setupSucceeded = $false
            $setupTimedOut = $false
            for ($attempt = 1; $attempt -le $SetupAttempts; $attempt++) {
                $proc = $null
                try {
                    Add-Action "run_setup_attempt=$attempt/$SetupAttempts"
                    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmdLine) -WorkingDirectory (Split-Path -Parent $setupScript) -PassThru
                    $finished = Wait-ProcessWithTimeout -Process $proc -TimeoutSeconds ($SetupTimeoutMinutes * 60)
                    if (-not $finished) {
                        $setupTimedOut = $true
                        $setupExitCode = 124
                        Safe-StopProcess -Process $proc
                        Add-Warn "setup_mfa.bat attempt $attempt timed out after $SetupTimeoutMinutes minutes."
                    } else {
                        $setupExitCode = [int]$proc.ExitCode
                        if ($setupExitCode -eq 0) {
                            $setupSucceeded = $true
                            break
                        }
                        Add-Warn "setup_mfa.bat attempt $attempt failed with exit code $setupExitCode."
                    }
                } catch {
                    $setupProcessError = $_.Exception.Message
                    break
                } finally {
                    Safe-StopProcess -Process $proc
                }
                if (-not $setupSucceeded -and $attempt -lt $SetupAttempts) {
                    Start-Sleep -Seconds 2
                }
            }
            if (-not [string]::IsNullOrWhiteSpace($setupProcessError)) {
                Add-Check -Name "setup_process_started" -Passed $false -Value $setupScript -Detail $setupProcessError -Required $true
                Add-Hint "Failed to launch setup_mfa.bat. Try running it manually from cmd.exe with Administrator rights."
            } else {
                if ($setupTimedOut -and -not $setupSucceeded) {
                    Add-Check -Name "setup_completed_within_timeout" -Passed $false -Value "$SetupTimeoutMinutes min" -Detail "setup_mfa.bat did not finish in time." -Required $true
                    Add-Hint "setup_mfa.bat timed out. Re-run with a larger timeout and stable network."
                } else {
                    Add-Check -Name "setup_completed_within_timeout" -Passed $true -Value "$SetupTimeoutMinutes min" -Required $true
                }
                Add-Check -Name "setup_exit_code_zero" -Passed $setupSucceeded -Value "$setupExitCode" -Required $true
                if (-not $setupSucceeded) {
                    Add-Hint "setup_mfa.bat failed. Check network/proxy/antivirus blocks and rerun setup_mfa.bat --non-interactive --install."
                }
            }
        }
    }
} else {
    Add-Warn "SkipSetup enabled. Running diagnose-only checks."
}

$pythonExe = ""
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $pythonExe = $candidate
        break
    }
}
$launcherPath = Resolve-MfaLauncher -EnvDir $envDir

Add-Check -Name "runtime_root_exists" -Passed (Test-Path -LiteralPath $runtimeRootAbs -PathType Container) -Value $runtimeRootAbs -Required $true
Add-Check -Name "runtime_env_exists" -Passed (Test-Path -LiteralPath $envDir -PathType Container) -Value $envDir -Required $true
Add-Check -Name "runtime_python_exists" -Passed (-not [string]::IsNullOrWhiteSpace($pythonExe)) -Value $pythonExe -Required $true
Add-Check -Name "runtime_mfa_launcher_exists" -Passed (-not [string]::IsNullOrWhiteSpace($launcherPath)) -Value $launcherPath -Required $true

$mfaImportCheck = Invoke-PythonCheck -PythonExe $pythonExe -Code "import importlib.util,sys; sys.exit(0 if (importlib.util.find_spec('montreal_forced_aligner.command_line.mfa') or importlib.util.find_spec('montreal_forced_aligner')) else 1)"
Add-Check -Name "mfa_python_module_importable" -Passed $mfaImportCheck.ok -Value $pythonExe -Detail $mfaImportCheck.output.Trim() -Required $true
if (-not $mfaImportCheck.ok) {
    Add-Hint "MFA python module import failed. Re-run setup_mfa.bat --non-interactive --install to rebuild the environment."
}

    $packagingCheck = Invoke-PythonCheck -PythonExe $pythonExe -Code "import setuptools,pip,wheel"
Add-Check -Name "packaging_stack_ready" -Passed $packagingCheck.ok -Value $pythonExe -Detail $packagingCheck.output.Trim() -Required $true
if (-not $packagingCheck.ok) {
    Add-Hint "pip/setuptools/wheel check failed. Re-run setup_mfa.bat --non-interactive --install."
}

$audioCheck = Invoke-PythonCheck -PythonExe $pythonExe -Code "import soundfile"
Add-Check -Name "audio_dependencies_ready" -Passed $audioCheck.ok -Value $pythonExe -Detail $audioCheck.output.Trim() -Required $true
if (-not $audioCheck.ok) {
    Add-Hint "Audio deps (libsndfile/soundfile) are missing or broken. Re-run setup_mfa.bat --non-interactive --install."
}

if ($Language -eq "japanese") {
    $langCheck = Invoke-PythonCheck -PythonExe $pythonExe -Code "import spacy,sudachipy,sudachidict_core"
} else {
    $langCheck = Invoke-PythonCheck -PythonExe $pythonExe -Code "import sys,jamo`nok=False`ntry:`n from mecab import MeCab`n ok=True`nexcept Exception:`n try:`n  from mecab import Tagger`n  ok=True`n except Exception:`n  try:`n   import MeCab`n   ok=True`n  except Exception:`n   try:`n    import mecab_ko`n    ok=True`n   except Exception:`n    ok=False`nsys.exit(0 if ok else 1)"
}
Add-Check -Name "${Language}_dependencies_ready" -Passed $langCheck.ok -Value $pythonExe -Detail $langCheck.output.Trim() -Required $true
if (-not $langCheck.ok) {
    if ($Language -eq "korean") {
        Add-Hint "Korean tokenizer deps are still missing. Check python-mecab-ko / mecab-python3 wheel install and VC++ runtime."
    } else {
        Add-Hint "Japanese tokenizer deps are still missing. Re-run setup_mfa.bat with stable network and conda-forge access."
    }
}

$modelCheck = Test-MfaModelReady -LauncherPath $launcherPath -RuntimeRootPath $runtimeRootAbs -ModelName $modelName
Add-Check -Name "${Language}_model_ready" -Passed $modelCheck.ok -Value $modelName -Detail $modelCheck.detail -Required $true
if (-not $modelCheck.ok) {
    Add-Hint "MFA model is missing. Check TLS/proxy/firewall and rerun setup_mfa.bat --non-interactive --install."
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
    recovery_profile = $recoveryProfile
    interactive_mode = [bool]$shouldShowMenu
    setup_script_path = $setupScript
    runtime_root = $runtimeRootAbs
    env_dir = $envDir
    python_exe = $pythonExe
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
