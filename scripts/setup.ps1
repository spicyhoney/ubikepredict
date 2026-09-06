[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.12 -m venv $VenvDir
    } else {
        $Python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $Python) {
            throw "Python not found. Install 64-bit Python 3.12 and try again."
        }
        & $Python.Source -m venv $VenvDir
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
Write-Host "Setup complete. Next run scripts\smoke-test.ps1."
