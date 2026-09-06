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
    throw "Python environment not found. Run scripts\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath $Vinext)) {
    throw "Frontend dependencies not found. Run scripts\setup-frontend.ps1 first."
}

New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

$BusyPorts = @(Get-NetTCPConnection -State Listen -LocalPort 3000, 8000 -ErrorAction SilentlyContinue)
if ($BusyPorts.Count -gt 0) {
    $PortList = ($BusyPorts.LocalPort | Sort-Object -Unique) -join ", "
    throw "Port $PortList is already in use. Close the old Demo window and try again."
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
        throw "The model API did not start within 10 seconds. See .runtime\api.stderr.log."
    }

    Write-Host "Demo started: http://localhost:3000"
    Write-Host "Press Ctrl+C to stop both the frontend and model API."
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
