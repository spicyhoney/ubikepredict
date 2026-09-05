[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "尚未建立 .venv，請先執行 scripts\setup.ps1。"
}
& $Python (Join-Path $RepoRoot "tests\smoke_test.py")

