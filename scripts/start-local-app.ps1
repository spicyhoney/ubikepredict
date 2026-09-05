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
    (Resolve-Path -LiteralPath $PythonPath).Path
}
$FrontendRoot = Join-Path $RepoRoot "frontend"
$FrontendModules = Join-Path $FrontendRoot "node_modules"
$Vinext = Join-Path $FrontendModules ".bin\vinext.CMD"
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$ApiOutput = Join-Path $RuntimeRoot "api.stdout.log"
$ApiError = Join-Path $RuntimeRoot "api.stderr.log"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "尚未建立 Python 環境，請先執行 scripts\setup.ps1。"
}
if (-not (Test-Path -LiteralPath $Vinext)) {
    throw "尚未安裝前端套件，請先執行 scripts\setup-frontend.ps1。"
}

New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

$BusyPorts = @(Get-NetTCPConnection -State Listen -LocalPort 3000, 8000 -ErrorAction SilentlyContinue)
if ($BusyPorts.Count -gt 0) {
    $PortList = ($BusyPorts.LocalPort | Sort-Object -Unique) -join "、"
    throw "連接埠 $PortList 已被使用。請先關閉舊的 Demo 視窗，再重新執行本腳本。"
}

$ApiProcess = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "backend.api") `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $ApiOutput `
    -RedirectStandardError $ApiError `
    -PassThru

try {
    $ApiReady = $false
    $ApiServerPid = $null
    for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
        try {
            $Health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2
            if ($Health.status -eq "ok") {
                $ApiReady = $true
                $ApiServerPid = [int]$Health.process_id
                break
            }
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $ApiReady) {
        if (Test-Path -LiteralPath $ApiError) {
            Get-Content -LiteralPath $ApiError | Write-Host
        }
        throw "模型服務未能在10秒內啟動。詳細錯誤位於 .runtime\api.stderr.log。"
    }

    Write-Host "Demo 已啟動：http://localhost:3000"
    Write-Host "按 Ctrl+C 可同時停止前端與模型服務。"
    Push-Location $FrontendRoot
    try {
        & $Vinext dev
    } finally {
        Pop-Location
    }
} finally {
    $ProcessIds = @($ApiProcess.Id, $ApiServerPid) | Where-Object { $null -ne $_ } | Sort-Object -Unique
    foreach ($ProcessId in $ProcessIds) {
        Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    }
}
