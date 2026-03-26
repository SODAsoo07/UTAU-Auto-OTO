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

function Get-RuntimeRootCandidates {
    param([string]$RepoRoot)

    $candidates = @(
        $RepoRoot,
        (Join-Path $RepoRoot "UTAU_Auto_OTO"),
        (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO"),
        (Join-Path $env:LOCALAPPDATA "UTAU_Auto_OTO_v3"),
        (Join-Path $env:USERPROFILE "AppData\Local\UTAU_Auto_OTO"),
        (Join-Path $env:USERPROFILE "AppData\Local\UTAU_Auto_OTO_v3")
    )
    $resolved = New-Object System.Collections.ArrayList
    foreach ($c in $candidates) {
        if ([string]::IsNullOrWhiteSpace($c)) { continue }
        if (-not (Test-Path -LiteralPath $c)) { continue }
        try {
            $full = (Resolve-Path -LiteralPath $c).Path
            if (-not ($resolved -contains $full)) {
                $resolved.Add($full) | Out-Null
            }
        } catch {}
    }
    return $resolved
}

function Resolve-RuntimePython {
    param(
        [string]$RepoRoot,
        [string]$MfaPath = ""
    )

    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($MfaPath) -and (Test-Path -LiteralPath $MfaPath)) {
        try {
            $mfaDir = Split-Path -Parent $MfaPath
            $envDir = Split-Path -Parent $mfaDir
            if (-not [string]::IsNullOrWhiteSpace($envDir)) {
                $roots += $envDir
                $envName = (Split-Path -Leaf $envDir).ToLowerInvariant()
                if ($envName -in @(".env", "env")) {
                    $roots += (Split-Path -Parent $envDir)
                }
            }
        } catch {}
    }
    $roots += Get-RuntimeRootCandidates -RepoRoot $RepoRoot

    $candidates = @()
    foreach ($r in $roots) {
        $candidates += @(
            (Join-Path $r ".env\python.exe"),
            (Join-Path $r ".env\Scripts\python.exe"),
            (Join-Path $r "env\python.exe"),
            (Join-Path $r "env\Scripts\python.exe"),
            (Join-Path $r "python.exe"),
            (Join-Path $r "Scripts\python.exe")
        )
    }

    $resolved = Resolve-ExistingPath $candidates
    if (-not [string]::IsNullOrWhiteSpace($resolved)) {
        return $resolved
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCmd) {
        return $pythonCmd.Source
    }
    return ""
}

function Resolve-RuntimeMfa {
    param(
        [string]$RepoRoot,
        [string]$PythonPath = ""
    )

    $roots = @()
    if (-not [string]::IsNullOrWhiteSpace($PythonPath) -and (Test-Path -LiteralPath $PythonPath)) {
        try {
            $pyDir = Split-Path -Parent $PythonPath
            $base = Split-Path -Leaf $pyDir
            if ($base -ieq "Scripts") {
                $roots += (Split-Path -Parent $pyDir)
            } else {
                $roots += $pyDir
            }
        } catch {}
    }
    $roots += Get-RuntimeRootCandidates -RepoRoot $RepoRoot

    $candidates = @()
    foreach ($r in $roots) {
        $candidates += @(
            (Join-Path $r ".env\Scripts\mfa.exe"),
            (Join-Path $r ".env\Scripts\mfa.bat"),
            (Join-Path $r ".env\Scripts\mfa.cmd"),
            (Join-Path $r "env\Scripts\mfa.exe"),
            (Join-Path $r "env\Scripts\mfa.bat"),
            (Join-Path $r "env\Scripts\mfa.cmd"),
            (Join-Path $r "Scripts\mfa.exe"),
            (Join-Path $r "Scripts\mfa.bat"),
            (Join-Path $r "Scripts\mfa.cmd")
        )
    }

    $resolved = Resolve-ExistingPath $candidates
    if (-not [string]::IsNullOrWhiteSpace($resolved)) {
        return $resolved
    }
    $mfaCmd = Get-Command mfa -ErrorAction SilentlyContinue
    if ($null -ne $mfaCmd) {
        return $mfaCmd.Source
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

if ([string]::IsNullOrWhiteSpace($inputWavFolder) -or [string]::IsNullOrWhiteSpace($inputDictPath)) {
    $corpusCandidates = @(
        (Join-Path $repoAbs "SODAsoo KR CVC"),
        (Join-Path $repoAbs "UTAU_Auto_OTO\SODAsoo KR CVC"),
        (Join-Path $repoAbs "Test\SODAsoo KR CVC")
    )
    foreach ($candidate in $corpusCandidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $dictCandidate = Join-Path $candidate "korean_dict.txt"
        if (-not (Test-Path -LiteralPath $dictCandidate)) { continue }
        $wavCount = @(Get-ChildItem -LiteralPath $candidate -Filter "*.wav" -File -ErrorAction SilentlyContinue).Count
        $labCount = @(Get-ChildItem -LiteralPath $candidate -Filter "*.lab" -File -ErrorAction SilentlyContinue).Count
        if ($wavCount -lt 1 -or $labCount -lt 1) { continue }
        $inputWavFolder = (Resolve-Path -LiteralPath $candidate).Path
        $inputDictPath = (Resolve-Path -LiteralPath $dictCandidate).Path
        break
    }
}

Push-Location $repoAbs
try {
    $mfaPath = Resolve-RuntimeMfa -RepoRoot $repoAbs
    $pythonPath = Resolve-RuntimePython -RepoRoot $repoAbs -MfaPath $mfaPath
    Add-Check -Name "python_available_precheck" -Passed (-not [string]::IsNullOrWhiteSpace($pythonPath)) -Detail $pythonPath -Required $false
    Add-Check -Name "mfa_entry_discovered_precheck" -Passed (-not [string]::IsNullOrWhiteSpace($mfaPath)) -Detail $mfaPath -Required $false

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
            $mfaPath = Resolve-RuntimeMfa -RepoRoot $repoAbs -PythonPath $pythonPath
            $pythonPath = Resolve-RuntimePython -RepoRoot $repoAbs -MfaPath $mfaPath
        } else {
            Add-Warn "setup_mfa.bat not found. Skipping install stage."
        }
    } else {
        Add-Warn "SkipSetupMfa enabled; setup stage skipped."
    }

    Add-Check -Name "python_available" -Passed (-not [string]::IsNullOrWhiteSpace($pythonPath)) -Detail $pythonPath -Required $true
    Add-Check -Name "mfa_entry_discovered" -Passed (-not [string]::IsNullOrWhiteSpace($mfaPath)) -Detail $mfaPath -Required $false
    if (-not [string]::IsNullOrWhiteSpace($inputWavFolder) -and -not [string]::IsNullOrWhiteSpace($inputDictPath)) {
        Add-Check -Name "smoke_input_autodetect" -Passed $true -Detail ("wav={0}; dict={1}" -f $inputWavFolder, $inputDictPath) -Required $false
    } else {
        Add-Check -Name "smoke_input_autodetect" -Passed $false -Detail "external sample not found; fallback to generated sample" -Required $false
    }

    $runnerPy = Join-Path $workAbs "run_windows_mfa_smoke.py"
@'
import json
import os
import struct
import subprocess
import sys
import traceback
import wave
import importlib.util
import shutil

repo_root = sys.argv[1]
work_dir = sys.argv[2]
mfa_path = sys.argv[3] if len(sys.argv) > 3 else ""
input_wav_dir = sys.argv[4] if len(sys.argv) > 4 else ""
input_dict_path = sys.argv[5] if len(sys.argv) > 5 else ""

def _resolve_code_root(root: str) -> str:
    candidates = [
        root,
        os.path.join(root, "source_bundle"),
        os.path.join(root, "UTAU_Auto_OTO"),
        os.path.join(root, "Auto_OTO"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "core", "mfa_runner.py")):
            return c
    return ""

code_root = _resolve_code_root(repo_root)
if code_root:
    sys.path.insert(0, code_root)

dep_probe = {
    "jamo": bool(importlib.util.find_spec("jamo")),
    "mecab": bool(importlib.util.find_spec("mecab")),
    "mecab_caps": bool(importlib.util.find_spec("MeCab")),
    "mecab_ko": bool(importlib.util.find_spec("mecab_ko")),
}

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

def _env_root_from_mfa_path(path: str) -> str:
    if not path:
        return ""
    p = os.path.abspath(path)
    parent = os.path.dirname(p)
    if os.path.basename(parent).lower() == "scripts":
        return os.path.dirname(parent)
    return parent

def _ensure_mecab_module_alias(mfa_exe: str) -> bool:
    try:
        if importlib.util.find_spec("mecab"):
            return False
        if not importlib.util.find_spec("MeCab"):
            return False
        env_root = _env_root_from_mfa_path(mfa_exe)
        if not env_root:
            return False
        site_dir = os.path.join(env_root, "Lib", "site-packages")
        if not os.path.isdir(site_dir):
            return False
        shim_path = os.path.join(site_dir, "mecab.py")
        if os.path.isfile(shim_path):
            return False
        shim_code = (
            "from MeCab import Tagger\n"
            "class _Node:\n"
            "    def __init__(self, surface, pos):\n"
            "        self.surface = surface\n"
            "        self.pos = pos\n"
            "class MeCab:\n"
            "    def __init__(self):\n"
            "        self._tagger = Tagger()\n"
            "    def parse(self, text):\n"
            "        node = self._tagger.parseToNode(text)\n"
            "        out = []\n"
            "        while node is not None:\n"
            "            surface = getattr(node, 'surface', '') or ''\n"
            "            if surface:\n"
            "                feature = getattr(node, 'feature', '') or ''\n"
            "                if not isinstance(feature, str):\n"
            "                    feature = str(feature)\n"
            "                pos = feature.split(',', 1)[0] if feature else ''\n"
            "                out.append(_Node(surface, pos))\n"
            "            node = getattr(node, 'next', None)\n"
            "        return out\n"
        )
        with open(shim_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(shim_code)
        return True
    except Exception:
        return False

def _resolve_mfa_root(repo_root_path: str, env_root: str) -> str:
    candidates = []
    for c in (
        os.environ.get("MFA_ROOT_DIR", ""),
        os.path.join(repo_root_path, ".mfa_root_ascii"),
        os.path.join(env_root, ".mfa_root_ascii") if env_root else "",
        os.path.join(os.environ.get("PUBLIC", r"C:\Users\Public"), "UTAU_Auto_OTO_v3", ".mfa_root_ascii"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "UTAU_Auto_OTO_v3", ".mfa_root_ascii"),
    ):
        if c and c not in candidates:
            candidates.append(c)
    for c in candidates:
        if os.path.isdir(os.path.join(c, "pretrained_models", "acoustic")):
            return c
    return ""

def _find_local_model_artifact(model_name: str, mfa_exe: str, mfa_root: str) -> str:
    env_root = _env_root_from_mfa_path(mfa_exe)
    search_roots = []
    for root in (repo_root, code_root, env_root, mfa_root):
        if root and os.path.isdir(root) and root not in search_roots:
            search_roots.append(root)
    for root in search_roots:
        model_dir = os.path.join(root, "pretrained_models", "acoustic")
        if not os.path.isdir(model_dir):
            nested_dir = os.path.join(root, ".mfa_root_ascii", "pretrained_models", "acoustic")
            if os.path.isdir(nested_dir):
                model_dir = nested_dir
        candidates = [
            os.path.join(model_dir, model_name),
            os.path.join(model_dir, model_name + ".zip"),
            os.path.join(model_dir, model_name + ".yaml"),
            os.path.join(model_dir, model_name + ".yml"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    return ""

def _resolve_model_reference(model_name: str, mfa_exe: str) -> str:
    env = os.environ.copy()
    env_root = _env_root_from_mfa_path(mfa_exe)
    mfa_root = _resolve_mfa_root(repo_root, env_root)
    if env_root:
        env_path_parts = [
            env_root,
            os.path.join(env_root, "Library", "mingw-w64", "bin"),
            os.path.join(env_root, "Library", "usr", "bin"),
            os.path.join(env_root, "Library", "bin"),
            os.path.join(env_root, "Scripts"),
            os.path.join(env_root, "bin"),
            env.get("PATH", ""),
        ]
        env["PATH"] = os.pathsep.join([p for p in env_path_parts if p])
    if mfa_root:
        env["MFA_ROOT_DIR"] = mfa_root
    if mfa_exe:
        list_cmd = [mfa_exe, "model", "list", "acoustic"]
        if mfa_exe.lower().endswith((".bat", ".cmd")):
            list_cmd = ["cmd.exe", "/c", mfa_exe, "model", "list", "acoustic"]
        try:
            result = subprocess.run(
                list_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45,
                env=env,
            )
            listed = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
            if result.returncode == 0 and model_name.lower() in listed:
                return model_name
        except Exception:
            pass

    local = _find_local_model_artifact(model_name, mfa_exe, mfa_root)
    if local:
        return local
    return model_name

def _run_align_cli(mfa_exe: str, wav_path: str, dictionary_path: str, output_path: str):
    if not mfa_exe:
        return False, "MFA executable not found", ""

    env = os.environ.copy()
    env_root = _env_root_from_mfa_path(mfa_exe)
    mfa_root = _resolve_mfa_root(repo_root, env_root)
    if env_root:
        env["CONDA_PREFIX"] = env_root
        env_path_parts = [
            env_root,
            os.path.join(env_root, "Library", "mingw-w64", "bin"),
            os.path.join(env_root, "Library", "usr", "bin"),
            os.path.join(env_root, "Library", "bin"),
            os.path.join(env_root, "Scripts"),
            os.path.join(env_root, "bin"),
            env.get("PATH", ""),
        ]
        env["PATH"] = os.pathsep.join([p for p in env_path_parts if p])
    if mfa_root:
        env["MFA_ROOT_DIR"] = mfa_root

    model_reference = _resolve_model_reference("korean_mfa", mfa_exe)
    cmd = [
        mfa_exe,
        "align",
        wav_path,
        dictionary_path,
        model_reference,
        output_path,
        "--clean",
        "--single_speaker",
        "--num_jobs",
        "1",
    ]
    if mfa_exe.lower().endswith((".bat", ".cmd")):
        cmd = ["cmd.exe", "/c", mfa_exe] + cmd[1:]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    merged = (proc.stderr or "").strip()
    if not merged:
        merged = (proc.stdout or "").strip()
    tail = "\n".join(merged.splitlines()[-15:]).strip()
    return proc.returncode == 0, tail, model_reference

try:
    runner_mode = "core_api"
    core_import_error = ""
    model_reference = ""
    resolved_mfa = mfa_path or (shutil.which("mfa") or "")
    mecab_alias_written = _ensure_mecab_module_alias(resolved_mfa)
    ok = False
    err = ""

    if code_root:
        try:
            from core.mfa_runner import run_mfa_align, find_mfa_executable  # noqa: E402
            resolved_mfa = mfa_path or (find_mfa_executable() or resolved_mfa)
            ok, err = run_mfa_align(
                resolved_mfa,
                wav_dir,
                dict_path,
                out_dir,
                language="korean",
            )
        except Exception as exc:
            runner_mode = "cli_fallback"
            core_import_error = f"{type(exc).__name__}: {exc}"
            ok, err, model_reference = _run_align_cli(resolved_mfa, wav_dir, dict_path, out_dir)
    else:
        runner_mode = "cli_fallback"
        core_import_error = "core.mfa_runner not found in build payload"
        ok, err, model_reference = _run_align_cli(resolved_mfa, wav_dir, dict_path, out_dir)

    tg_count = len([n for n in os.listdir(out_dir) if n.lower().endswith(".textgrid")])
    low_err = str(err or "").lower()
    if ok:
        failure_category = ""
    elif "permission" in low_err or "access denied" in low_err:
        failure_category = "permission"
    elif "tokenizer" in low_err or "dependency" in low_err:
        failure_category = "dependency"
    elif "model" in low_err or "pretrainedmodelnotfounderror" in low_err:
        failure_category = "model"
    elif "dictionary" in low_err:
        failure_category = "dictionary"
    elif "executable not found" in low_err:
        failure_category = "executable"
    else:
        failure_category = "runtime"
    payload = {
        "ok": bool(ok),
        "error": str(err or ""),
        "failure_category": failure_category,
        "runner_mode": runner_mode,
        "mecab_alias_written": mecab_alias_written,
        "core_import_error": core_import_error,
        "model_reference": model_reference,
        "code_root": code_root,
        "dep_probe": dep_probe,
        "mfa_path": resolved_mfa,
        "wav_dir": wav_dir,
        "dict_path": dict_path,
        "output_dir": out_dir,
        "textgrid_count": tg_count,
        "generated_sample": generated_sample,
    }
except Exception:
    payload = {
        "ok": False,
        "error": traceback.format_exc(),
        "failure_category": "runtime",
        "mecab_alias_written": False,
        "code_root": code_root,
        "dep_probe": dep_probe,
        "mfa_path": mfa_path,
        "wav_dir": wav_dir,
        "dict_path": dict_path,
        "output_dir": out_dir,
        "textgrid_count": 0,
        "generated_sample": generated_sample,
    }
print(json.dumps(payload, ensure_ascii=True))
'@ | Set-Content -LiteralPath $runnerPy -Encoding UTF8

    if ([string]::IsNullOrWhiteSpace($pythonPath)) {
        Add-Check -Name "smoke_python_runner" -Passed $false -Detail "python not found" -Required $true
    } else {
        Write-Host "[INFO] Running Korean MFA alignment smoke..."
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $smokeOutput = & $pythonPath $runnerPy $repoAbs $workAbs $mfaPath $inputWavFolder $inputDictPath 2>&1
        $pythonExit = $LASTEXITCODE
        $ErrorActionPreference = $prevEap
        $smokeText = ($smokeOutput | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($smokeText)) {
            Add-Check -Name "smoke_output_emitted" -Passed $false -Detail "empty output" -Required $true
        } else {
            $jsonLine = ($smokeText -split "`r?`n" | Select-Object -Last 1)
            try {
                $payload = $jsonLine | ConvertFrom-Json
                Add-Check -Name "smoke_python_exit_zero" -Passed ($pythonExit -eq 0) -Detail ("exit=" + $pythonExit) -Required $false
                $hasMfaPath = -not [string]::IsNullOrWhiteSpace([string]$payload.mfa_path)
                $invokeDetail = "mfa={0}; mode={1}" -f [string]$payload.mfa_path, [string]$payload.runner_mode
                Add-Check -Name "alignment_invoked" -Passed $hasMfaPath -Detail $invokeDetail -Required $true
                Add-Check -Name "input_mode" -Passed $true -Detail ($(if ($payload.generated_sample) { "generated_sample" } else { "external_inputs" })) -Required $false
                if ($null -ne $payload.dep_probe) {
                    $depDetail = "jamo=$($payload.dep_probe.jamo); mecab=$($payload.dep_probe.mecab); mecab_caps=$($payload.dep_probe.mecab_caps); mecab_ko=$($payload.dep_probe.mecab_ko)"
                    Add-Check -Name "korean_dep_probe" -Passed $true -Detail $depDetail -Required $false
                }
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
