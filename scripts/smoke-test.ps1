[CmdletBinding()]
param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv not found. Run scripts\setup.ps1 first."
}

& $Python (Join-Path $RepoRoot "tests\smoke_test.py")
if ($LASTEXITCODE -ne 0) {
    throw "Frozen model smoke tests failed."
}

& $Python -m unittest discover -s (Join-Path $RepoRoot "tests") -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) {
    throw "API tests failed."
}

Write-Host "Smoke test complete: 13 model/API tests passed."
