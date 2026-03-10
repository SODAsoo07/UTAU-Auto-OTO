[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsFromCaller
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    $OutputEncoding = [Console]::OutputEncoding
} catch {
    # Keep going even if host does not allow console encoding changes.
}

try {
    chcp 65001 | Out-Null
} catch {
    # Ignore if command is unavailable in this host.
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $scriptDir "run_oto_generation_batch.py"

if (-not (Test-Path -LiteralPath $py)) {
    throw "Script not found: $py"
}

python $py @ArgsFromCaller
$exitCode = $LASTEXITCODE
if ($null -eq $exitCode) { $exitCode = 0 }
exit $exitCode
