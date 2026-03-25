param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$WorkDir = "windows_mfa_smoke_output",
    [switch]$SkipSetupMfa,
    [int]$SetupTimeoutMinutes = 30,
    [string]$WavFolder = "",
    [string]$DictionaryPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Add-Check {
    param(
        [string]$Name,
        [bool]$Passed,
        [string]$Detail = "",
        [bool]$Required = $true
    )
    $script:Checks.Add([ordered]@{
        name = $Name
        passed = $Passed
        detail = $Detail
        required = $Required
    }) | Out-Null
}

function Add-Warn {
    param([string]$Message)
    $script:Warnings.Add($Message) | Out-Null
}

function Resolve-ExistingPath {
    param([string[]]$Candidates)
    foreach ($c in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($c)) { continue }
        if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path }
    }
    return ""
}

$Checks = New-Object System.Collections.ArrayList
$Warnings = New-Object System.Collections.ArrayList

$repoAbs = (Resolve-Path -LiteralPath $RepoRoot).Path
$workAbs = Join-Path $repoAbs $WorkDir
if (-not (Test-Path -LiteralPath $workAbs)) {
    New-Item -ItemType Directory -Path $workAbs | Out-Null
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = Join-Path $workAbs "windows_mfa_smoke_report_$stamp.json"
$inputWavFolder = ""
$inputDictPath = ""
if (-not [string]::IsNullOrWhiteSpace($WavFolder)) {
    try { $inputWavFolder = (Resolve-Path -LiteralPath $WavFolder).Path } catch { $inputWavFolder = $WavFolder }
}
if (-not [string]::IsNullOrWhiteSpace($DictionaryPath)) {
    try { $inputDictPath = (Resolve-Path -LiteralPath $DictionaryPath).Path } catch { $inputDictPath = $DictionaryPath }
}

Push-Location $repoAbs
try {
    $pythonPath = Resolve-ExistingPath @(
        (Join-Path $repoAbs ".env\python.exe"),
        (Join-Path $repoAbs ".env\Scripts\python.exe")
    )
    if ([string]::IsNullOrWhiteSpace($pythonPath)) {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $pythonCmd) {
            $pythonPath = $pythonCmd.Source
        }
    }
    Add-Check -Name "python_available" -Passed (-not [string]::IsNullOrWhiteSpace($pythonPath)) -Detail $pythonPath -Required $true

    $mfaPath = Resolve-ExistingPath @(
        (Join-Path $repoAbs ".env\Scripts\mfa.exe"),
        (Join-Path $repoAbs ".env\Scripts\mfa.bat"),
        (Join-Path $repoAbs ".env\Scripts\mfa.cmd")
    )
    if ([string]::IsNullOrWhiteSpace($mfaPath)) {
        $mfaCmd = Get-Command mfa -ErrorAction SilentlyContinue
        if ($null -ne $mfaCmd) {
            $mfaPath = $mfaCmd.Source
        }
    }
    Add-Check -Name "mfa_entry_discovered" -Passed (-not [string]::IsNullOrWhiteSpace($mfaPath)) -Detail $mfaPath -Required $false

    if (-not $SkipSetupMfa) {
        $setupBat = Join-Path $repoAbs "setup_mfa.bat"
        if (Test-Path -LiteralPath $setupBat) {
            Write-Host "[INFO] Running setup_mfa.bat --non-interactive --install ..."
            $setupProc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", "`"$setupBat`" --non-interactive --install") -PassThru
            $done = $setupProc.WaitForExit($SetupTimeoutMinutes * 60 * 1000)
            if (-not $done) {
                try { Stop-Process -Id $setupProc.Id -Force } catch {}
                Add-Check -Name "setup_mfa_completed" -Passed $false -Detail "timeout ${SetupTimeoutMinutes}min" -Required $true
            } else {
                Add-Check -Name "setup_mfa_exit_zero" -Passed ($setupProc.ExitCode -eq 0) -Detail ("exit=" + $setupProc.ExitCode) -Required $true
            }
            if ([string]::IsNullOrWhiteSpace($mfaPath)) {
                $mfaPath = Resolve-ExistingPath @(
                    (Join-Path $repoAbs ".env\Scripts\mfa.exe"),
                    (Join-Path $repoAbs ".env\Scripts\mfa.bat"),
                    (Join-Path $repoAbs ".env\Scripts\mfa.cmd")
                )
            }
        } else {
            Add-Warn "setup_mfa.bat not found. Skipping install stage."
        }
    } else {
        Add-Warn "SkipSetupMfa enabled; setup stage skipped."
    }

    $runnerPy = Join-Path $workAbs "run_windows_mfa_smoke.py"
    @'
import json
import os
import struct
import sys
import wave

repo_root = sys.argv[1]
work_dir = sys.argv[2]
mfa_path = sys.argv[3] if len(sys.argv) > 3 else ""
input_wav_dir = sys.argv[4] if len(sys.argv) > 4 else ""
input_dict_path = sys.argv[5] if len(sys.argv) > 5 else ""

sys.path.insert(0, repo_root)
from core.mfa_runner import run_mfa_align, find_mfa_executable  # noqa: E402

out_dir = os.path.join(work_dir, "textgrids")
os.makedirs(out_dir, exist_ok=True)
generated_sample = False

if input_wav_dir and input_dict_path and os.path.isdir(input_wav_dir) and os.path.isfile(input_dict_path):
    wav_dir = input_wav_dir
    dict_path = input_dict_path
else:
    wav_dir = os.path.join(work_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    dict_path = os.path.join(work_dir, "dictionary.txt")
    wav_path = os.path.join(wav_dir, "sample.wav")
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("가 g a\n")
    with wave.open(wav_path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = [0] * 16000
        w.writeframes(b"".join(struct.pack("<h", x) for x in frames))
    generated_sample = True

resolved_mfa = mfa_path or (find_mfa_executable() or "")
ok, err = run_mfa_align(
    resolved_mfa,
    wav_dir,
    dict_path,
    out_dir,
    language="korean",
)
tg_count = len([n for n in os.listdir(out_dir) if n.lower().endswith(".textgrid")])
low_err = str(err or "").lower()
if ok:
    failure_category = ""
elif "permission" in low_err or "access denied" in low_err:
    failure_category = "permission"
elif "tokenizer" in low_err or "dependency" in low_err:
    failure_category = "dependency"
elif "dictionary" in low_err:
    failure_category = "dictionary"
elif "model" in low_err:
    failure_category = "model"
elif "executable not found" in low_err:
    failure_category = "executable"
else:
    failure_category = "runtime"
payload = {
    "ok": bool(ok),
    "error": str(err or ""),
    "failure_category": failure_category,
    "mfa_path": resolved_mfa,
    "wav_dir": wav_dir,
    "dict_path": dict_path,
    "output_dir": out_dir,
    "textgrid_count": tg_count,
    "generated_sample": generated_sample,
}
print(json.dumps(payload, ensure_ascii=False))
'@ | Set-Content -LiteralPath $runnerPy -Encoding UTF8

    if ([string]::IsNullOrWhiteSpace($pythonPath)) {
        Add-Check -Name "smoke_python_runner" -Passed $false -Detail "python not found" -Required $true
    } else {
        Write-Host "[INFO] Running Korean MFA alignment smoke..."
        $smokeOutput = & $pythonPath $runnerPy $repoAbs $workAbs $mfaPath $inputWavFolder $inputDictPath 2>&1
        $smokeText = ($smokeOutput | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($smokeText)) {
            Add-Check -Name "smoke_output_emitted" -Passed $false -Detail "empty output" -Required $true
        } else {
            $jsonLine = ($smokeText -split "`r?`n" | Select-Object -Last 1)
            try {
                $payload = $jsonLine | ConvertFrom-Json
                Add-Check -Name "alignment_invoked" -Passed $true -Detail ($payload.mfa_path) -Required $true
                Add-Check -Name "input_mode" -Passed $true -Detail ($(if ($payload.generated_sample) { "generated_sample" } else { "external_inputs" })) -Required $false
                Add-Check -Name "alignment_ok" -Passed ([bool]$payload.ok) -Detail ($payload.error + " | category=" + $payload.failure_category) -Required $true
                Add-Check -Name "textgrid_generated" -Passed (($payload.textgrid_count -as [int]) -gt 0) -Detail ("count=" + $payload.textgrid_count) -Required $true
            } catch {
                Add-Check -Name "smoke_output_parseable_json" -Passed $false -Detail $jsonLine -Required $true
            }
        }
    }
}
finally {
    Pop-Location
}

$requiredFailed = $false
foreach ($c in $Checks) {
    if ($c.required -and (-not $c.passed)) {
        $requiredFailed = $true
        break
    }
}

$report = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    repo_root = $repoAbs
    work_dir = $workAbs
    warnings = $Warnings
    checks = $Checks
    passed = (-not $requiredFailed)
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] smoke report: $reportPath"

if ($requiredFailed) {
    Write-Host "[FAILED] Windows MFA alignment smoke failed."
    exit 1
}

Write-Host "[OK] Windows MFA alignment smoke passed."
exit 0
