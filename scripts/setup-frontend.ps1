[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $RepoRoot "frontend"
$Node = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $Node) {
    throw "Node.js not found. Install Node.js 22.13 or newer."
}

$NodeVersionText = (& $Node.Source --version).TrimStart("v")
$NodeVersion = [version]$NodeVersionText
if ($NodeVersion -lt [version]"22.13.0") {
    throw "Node.js $NodeVersionText is installed; version 22.13 or newer is required."
}

$Pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
if ($null -eq $Pnpm) {
    $Corepack = Get-Command corepack -ErrorAction SilentlyContinue
    if ($null -eq $Corepack) {
        throw "pnpm or corepack not found. Install pnpm and try again."
    }
    & $Corepack.Source enable
    & $Corepack.Source prepare pnpm@11.19.0 --activate
    $Pnpm = Get-Command pnpm -ErrorAction Stop
}

Push-Location $FrontendRoot
try {
    & $Pnpm.Source install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend dependency installation failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Write-Host "Frontend setup complete. Next run scripts\start-local-app.ps1."
