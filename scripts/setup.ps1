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
            throw "找不到 Python。請先安裝 64-bit Python 3.12，再重新執行。"
        }
        & $Python.Source -m venv $VenvDir
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt")
Write-Host "安裝完成。下一步可執行 scripts\smoke-test.ps1。"

