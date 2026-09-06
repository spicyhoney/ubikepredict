[CmdletBinding()]
param(
    [ValidateSet("empty", "full_dock")]
    [string]$Mode = "empty",
    [string]$Datetime = "2026-06-23 19:30:00",
    [string]$District = "",
    [ValidateSet("balanced", "strict", "all")]
    [string]$Policy = "balanced",
    [ValidateRange(1, 500)]
    [int]$Top = 10
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv not found. Run scripts\setup.ps1 first."
}
$Arguments = @(
    (Join-Path $RepoRoot "backend\inference.py"),
    "--mode", $Mode,
    "--datetime", $Datetime,
    "--policy", $Policy,
    "--top", $Top,
    "--format", "table"
)
if ($District) {
    $Arguments += @("--district", $District)
}
& $Python @Arguments
