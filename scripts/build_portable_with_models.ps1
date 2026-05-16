$ErrorActionPreference = "Stop"
$target = Join-Path $PSScriptRoot "build\\build_portable_with_models.ps1"
if (-not (Test-Path -LiteralPath $target)) {
    throw "Target script not found: $target"
}
& $target @args
if ($LASTEXITCODE -is [int]) {
    exit $LASTEXITCODE
}
