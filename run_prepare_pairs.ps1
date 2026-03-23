param(
    [string]$DatasetRoot = ".\\dataset",
    [int]$Workers = 4,
    [switch]$RetryFailed,
    [string]$PythonPath = ""
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvDir = Join-Path $ScriptDir ".env"
$MambaRoot = Join-Path $ScriptDir "micromamba"
$MambaExeCandidates = @(
    (Join-Path $MambaRoot "Library\\bin\\micromamba.exe"),
    (Join-Path $MambaRoot "bin\\micromamba.exe"),
    (Join-Path $MambaRoot "micromamba.exe")
)
$MambaExe = $MambaExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

$PrepareScript = Join-Path $ScriptDir "ml\\scripts\\coupled\\prepare_pairs.py"
if (-not (Test-Path $PrepareScript)) {
    Write-Error "prepare_pairs.py not found: $PrepareScript"
    exit 1
}

if ([System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetArg = $DatasetRoot
} else {
    $DatasetArg = Join-Path $ScriptDir $DatasetRoot
}

$ArgsList = @(
    $PrepareScript,
    "--dataset-root", $DatasetArg,
    "--workers", $Workers.ToString()
)
if ($RetryFailed) {
    $ArgsList += "--retry-failed"
}

if (-not $PythonPath -and $MambaExe) {
    if (-not (Test-Path $EnvDir)) {
        Write-Error "Env not found: $EnvDir`nRun setup_mfa.bat --with-ml first."
        exit 1
    }
    & $MambaExe run -r $MambaRoot -p $EnvDir python @ArgsList
    exit $LASTEXITCODE
}

if (-not $PythonPath) {
    $PythonPath = Join-Path $EnvDir "python.exe"
}
if (-not (Test-Path $PythonPath)) {
    Write-Error "Python not found: $PythonPath`nRun setup_mfa.bat --with-ml first."
    exit 1
}

& $PythonPath @ArgsList
exit $LASTEXITCODE
