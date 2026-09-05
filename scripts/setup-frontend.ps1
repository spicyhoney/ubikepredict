[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $RepoRoot "frontend"
$Node = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $Node) {
    throw "找不到 Node.js。請先安裝 Node.js 22.13 以上版本。"
}

$NodeVersionText = (& $Node.Source --version).TrimStart("v")
$NodeVersion = [version]$NodeVersionText
if ($NodeVersion -lt [version]"22.13.0") {
    throw "目前 Node.js 是 $NodeVersionText；前端需要 22.13 以上版本。"
}

$Pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
if ($null -eq $Pnpm) {
    $Corepack = Get-Command corepack -ErrorAction SilentlyContinue
    if ($null -eq $Corepack) {
        throw "找不到 pnpm 或 corepack。請先安裝 pnpm，再重新執行。"
    }
    & $Corepack.Source enable
    & $Corepack.Source prepare pnpm@11.19.0 --activate
    $Pnpm = Get-Command pnpm -ErrorAction Stop
}

Push-Location $FrontendRoot
try {
    & $Pnpm.Source install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) {
        throw "前端套件安裝失敗，結束代碼 $LASTEXITCODE。"
    }
} finally {
    Pop-Location
}

Write-Host "前端安裝完成。下一步執行 scripts\start-local-app.ps1。"
