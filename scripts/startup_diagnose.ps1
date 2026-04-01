param(
    [string]$AppRoot = "",
    [int]$LaunchWaitSeconds = 8,
    [switch]$OpenFolder
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($AppRoot)) {
    if ($PSScriptRoot) {
        $AppRoot = $PSScriptRoot
    } else {
        $AppRoot = (Get-Location).Path
    }
}

$AppRoot = [System.IO.Path]::GetFullPath($AppRoot)
$logsDir = Join-Path $AppRoot "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$jsonPath = Join-Path $logsDir ("startup_diagnose_{0}.json" -f $stamp)
$txtPath = Join-Path $logsDir ("startup_diagnose_{0}.txt" -f $stamp)

$result = [ordered]@{
    started_at = (Get-Date).ToString("o")
    app_root = $AppRoot
    checks = @()
    notes = @()
    status = "running"
}

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Value = "",
        [string]$Detail = "",
        [bool]$Required = $true
    )

    $entry = [ordered]@{
        name = $Name
        passed = $Passed
        required = $Required
        value = $Value
        detail = $Detail
    }
    $script:result.checks += $entry

    $tag = if ($Passed) { "PASS" } else { if ($Required) { "FAIL" } else { "WARN" } }
    Write-Host "[$tag] $Name - $Value"
    if (-not [string]::IsNullOrWhiteSpace($Detail)) {
        Write-Host "       $Detail"
    }
}

function Add-Note {
    param([string]$Text)
    $script:result.notes += $Text
    Write-Host "[INFO] $Text"
}

$appExe = Join-Path $AppRoot "UTAU_Auto_OTO\UTAU_Auto_OTO.exe"
$vcDllA = "msvcp140.dll"
$vcDllB = "msvcp140_1.dll"
$vcDllC = "vcruntime140.dll"
$vcDllD = "vcruntime140_1.dll"

Add-Check -Name "app_exe_exists" -Passed (Test-Path -LiteralPath $appExe) -Value $appExe -Required $true

$localVcA = Join-Path $AppRoot $vcDllA
$localVcB = Join-Path $AppRoot $vcDllB
$localVcC = Join-Path $AppRoot $vcDllC
$localVcD = Join-Path $AppRoot $vcDllD
$sysVcA = Join-Path $env:WINDIR "System32\$vcDllA"
$sysVcB = Join-Path $env:WINDIR "System32\$vcDllB"
$sysVcC = Join-Path $env:WINDIR "System32\$vcDllC"
$sysVcD = Join-Path $env:WINDIR "System32\$vcDllD"

$vcAOk = (Test-Path -LiteralPath $localVcA) -or (Test-Path -LiteralPath $sysVcA)
$vcBOk = (Test-Path -LiteralPath $localVcB) -or (Test-Path -LiteralPath $sysVcB)
$vcCOk = (Test-Path -LiteralPath $localVcC) -or (Test-Path -LiteralPath $sysVcC)
$vcDOk = (Test-Path -LiteralPath $localVcD) -or (Test-Path -LiteralPath $sysVcD)
Add-Check -Name "vc_runtime_msvcp140_available" -Passed $vcAOk -Value $vcDllA -Detail ("local={0}; system={1}" -f (Test-Path -LiteralPath $localVcA), (Test-Path -LiteralPath $sysVcA)) -Required $true
Add-Check -Name "vc_runtime_msvcp140_1_available" -Passed $vcBOk -Value $vcDllB -Detail ("local={0}; system={1}" -f (Test-Path -LiteralPath $localVcB), (Test-Path -LiteralPath $sysVcB)) -Required $true
Add-Check -Name "vc_runtime_vcruntime140_available" -Passed $vcCOk -Value $vcDllC -Detail ("local={0}; system={1}" -f (Test-Path -LiteralPath $localVcC), (Test-Path -LiteralPath $sysVcC)) -Required $true
Add-Check -Name "vc_runtime_vcruntime140_1_available" -Passed $vcDOk -Value $vcDllD -Detail ("local={0}; system={1}" -f (Test-Path -LiteralPath $localVcD), (Test-Path -LiteralPath $sysVcD)) -Required $true

$envDir = Join-Path $AppRoot ".env"
$mambaDir = Join-Path $AppRoot "micromamba"
Add-Check -Name "mfa_env_exists_optional" -Passed (Test-Path -LiteralPath $envDir) -Value $envDir -Required $false
Add-Check -Name "micromamba_exists_optional" -Passed (Test-Path -LiteralPath $mambaDir) -Value $mambaDir -Required $false

$launchResult = [ordered]@{
    started = $false
    running_after_wait = $false
    exited = $false
    exit_code = $null
    error = ""
}

if (Test-Path -LiteralPath $appExe) {
    $proc = $null
    try {
        $proc = Start-Process -FilePath $appExe -PassThru
        $launchResult.started = $true
        Start-Sleep -Seconds ([Math]::Max(3, $LaunchWaitSeconds))

        try {
            $proc.Refresh()
            if ($proc.HasExited) {
                $launchResult.exited = $true
                $launchResult.exit_code = $proc.ExitCode
                Add-Check -Name "app_process_still_running" -Passed $false -Value ("pid={0}" -f $proc.Id) -Detail ("Exited quickly with code={0}" -f $proc.ExitCode) -Required $true
            } else {
                $launchResult.running_after_wait = $true
                Add-Check -Name "app_process_still_running" -Passed $true -Value ("pid={0}" -f $proc.Id) -Detail "Process is running after launch wait." -Required $true
            }
        } catch {
            Add-Check -Name "app_process_still_running" -Passed $false -Value "unknown" -Detail $_.Exception.Message -Required $true
        }
    } catch {
        $launchResult.error = $_.Exception.Message
        Add-Check -Name "app_start_process" -Passed $false -Value $appExe -Detail $_.Exception.Message -Required $true
    } finally {
        if ($proc) {
            try {
                if (-not $proc.HasExited) {
                    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
                }
            } catch {
            }
        }
    }
}

try {
    $cutoff = (Get-Date).AddHours(-24)
    $events = Get-WinEvent -FilterHashtable @{ LogName = "Application"; StartTime = $cutoff } -ErrorAction Stop |
        Where-Object {
            $_.ProviderName -in @("Application Error", ".NET Runtime", "Windows Error Reporting") -and
            ($_.Message -like "*UTAU_Auto_OTO.exe*" -or $_.Message -like "*UTAU Auto OTO*")
        } |
        Select-Object -First 5 TimeCreated, ProviderName, Id, LevelDisplayName, Message

    $hasEvents = ($events.Count -gt 0)
    Add-Check -Name "recent_crash_events_optional" -Passed (-not $hasEvents) -Value ("count={0}" -f $events.Count) -Detail "Application log scan for last 24h." -Required $false
    if ($hasEvents) {
        $script:result.recent_events = @($events | ForEach-Object {
            [ordered]@{
                time = $_.TimeCreated
                provider = $_.ProviderName
                id = $_.Id
                level = $_.LevelDisplayName
                message = [string]$_.Message
            }
        })
    }
} catch {
    Add-Check -Name "recent_crash_events_optional" -Passed $false -Value "scan_failed" -Detail $_.Exception.Message -Required $false
}

try {
    $defenderBlocked = $false
    $defenderDetail = ""
    $defenderState = "not_detected"
    if (Get-Command -Name Get-MpThreatDetection -ErrorAction SilentlyContinue) {
        $detections = Get-MpThreatDetection -ErrorAction SilentlyContinue | Where-Object {
            $_.Resources -match "UTAU_Auto_OTO"
        }
        if ($detections) {
            $defenderBlocked = $true
            $defenderState = "detected"
            $defenderDetail = ($detections | Select-Object -First 3 | ForEach-Object { $_.ThreatName }) -join ", "
        }
    } else {
        $defenderDetail = "Get-MpThreatDetection not available"
    }
    Add-Check -Name "defender_blocking_optional" -Passed (-not $defenderBlocked) -Value $defenderState -Detail $defenderDetail -Required $false
} catch {
    Add-Check -Name "defender_blocking_optional" -Passed $false -Value "scan_failed" -Detail $_.Exception.Message -Required $false
}

$requiredFailures = @($result.checks | Where-Object { $_.required -and -not $_.passed })
$result.launch = $launchResult
$result.finished_at = (Get-Date).ToString("o")
$result.required_failures = $requiredFailures.Count
$result.status = if ($requiredFailures.Count -gt 0) { "failed" } else { "passed" }

$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

$txt = @()
$txt += "UTAU Auto OTO startup diagnosis"
$txt += "started_at: $($result.started_at)"
$txt += "finished_at: $($result.finished_at)"
$txt += "app_root: $AppRoot"
$txt += "status: $($result.status)"
$txt += "required_failures: $($result.required_failures)"
$txt += ""
$txt += "Checks"
foreach ($check in $result.checks) {
    $state = if ($check.passed) { "PASS" } else { if ($check.required) { "FAIL" } else { "WARN" } }
    $txt += "[$state] $($check.name) | $($check.value)"
    if (-not [string]::IsNullOrWhiteSpace([string]$check.detail)) {
        $txt += "  detail: $($check.detail)"
    }
}
$txt += ""
$txt += "json_report: $jsonPath"
$txt | Set-Content -LiteralPath $txtPath -Encoding UTF8

Write-Host ""
Write-Host "[INFO] Diagnosis report (json): $jsonPath"
Write-Host "[INFO] Diagnosis report (txt) : $txtPath"

if ($OpenFolder) {
    try {
        Start-Process explorer.exe $logsDir | Out-Null
    } catch {
    }
}

if ($result.status -ne "passed") {
    exit 1
}
exit 0
