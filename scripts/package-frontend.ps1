[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApiBaseUrl,
    [string]$OutputPath = ".aws-artifacts\frontend.zip",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$FrontendRoot = Join-Path $RepoRoot "frontend"
$StaticRoot = Join-Path $FrontendRoot "dist\client"
$IndexPath = Join-Path $StaticRoot "index.html"
$OutputFullPath = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputPath))
}

$ParsedApiUrl = $null
if (-not [uri]::TryCreate($ApiBaseUrl.TrimEnd("/"), [System.UriKind]::Absolute, [ref]$ParsedApiUrl)) {
    throw "ApiBaseUrl must be an absolute URL."
}
if ($ParsedApiUrl.Scheme -ne "https") {
    throw "A deployable frontend requires an https API URL; localhost is only for the local development command."
}
if ($ParsedApiUrl.IsLoopback -or $ParsedApiUrl.Host -in @("localhost", "127.0.0.1")) {
    throw "Refusing to package a public frontend that points to localhost."
}

$Node = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $Node) {
    throw "Node.js was not found. Install Node.js 22.13 or newer."
}
$NodeVersion = [version]((& $Node.Source --version).TrimStart("v"))
if ($NodeVersion -lt [version]"22.13.0") {
    throw "Node.js $NodeVersion is installed; version 22.13 or newer is required."
}

$Pnpm = $null
if (-not $SkipInstall) {
    $Pnpm = Get-Command pnpm -ErrorAction SilentlyContinue
    if ($null -eq $Pnpm) {
        $Corepack = Get-Command corepack -ErrorAction SilentlyContinue
        if ($null -eq $Corepack) {
            throw "pnpm or corepack was not found."
        }
        & $Corepack.Source enable
        if ($LASTEXITCODE -ne 0) {
            throw "corepack enable failed with exit code $LASTEXITCODE."
        }
        & $Corepack.Source prepare pnpm@11.19.0 --activate
        if ($LASTEXITCODE -ne 0) {
            throw "pnpm activation failed with exit code $LASTEXITCODE."
        }
        $Pnpm = Get-Command pnpm -ErrorAction Stop
    }
}

$VinextName = if ($env:OS -eq "Windows_NT") { "vinext.CMD" } else { "vinext" }
$Vinext = Join-Path $FrontendRoot "node_modules\.bin\$VinextName"
$ProductionEnvCheck = Join-Path $FrontendRoot "scripts\check-production-env.mjs"
if (-not (Test-Path -LiteralPath $ProductionEnvCheck -PathType Leaf)) {
    throw "Frontend production environment check is missing."
}

$PreviousApiBaseUrl = [Environment]::GetEnvironmentVariable(
    "NEXT_PUBLIC_API_BASE_URL",
    [EnvironmentVariableTarget]::Process
)

Push-Location $FrontendRoot
try {
    if (-not $SkipInstall) {
        & $Pnpm.Source install --frozen-lockfile
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend dependency installation failed with exit code $LASTEXITCODE."
        }
    }
    if (-not (Test-Path -LiteralPath $Vinext -PathType Leaf)) {
        throw "Frontend dependencies are absent. Rerun without -SkipInstall."
    }

    [Environment]::SetEnvironmentVariable(
        "NEXT_PUBLIC_API_BASE_URL",
        $ApiBaseUrl.TrimEnd("/"),
        [EnvironmentVariableTarget]::Process
    )
    & $Node.Source $ProductionEnvCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Frontend production environment validation failed."
    }
    $BuildStartedUtc = [DateTime]::UtcNow
    & $Vinext build
    $BuildExitCode = $LASTEXITCODE
    if ($BuildExitCode -ne 0) {
        # Vinext 1.0.0-beta.5 can hit a libuv shutdown assertion on Windows
        # after it has successfully exported every route. Never suppress any
        # other failure, or an assertion that did not produce a fresh index.
        $KnownWindowsShutdownAssertion = -1073740791
        $FreshIndexExists = (Test-Path -LiteralPath $IndexPath -PathType Leaf) -and
            ((Get-Item -LiteralPath $IndexPath).LastWriteTimeUtc -ge $BuildStartedUtc.AddSeconds(-2))
        if ($BuildExitCode -eq $KnownWindowsShutdownAssertion -and $FreshIndexExists) {
            Write-Warning "Vinext completed the static export, then hit its known Windows shutdown assertion; the fresh artifact will still be verified."
        } else {
            throw "Frontend production build failed with exit code $BuildExitCode."
        }
    }
} finally {
    Pop-Location
    [Environment]::SetEnvironmentVariable(
        "NEXT_PUBLIC_API_BASE_URL",
        $PreviousApiBaseUrl,
        [EnvironmentVariableTarget]::Process
    )
}

if (-not (Test-Path -LiteralPath $IndexPath -PathType Leaf)) {
    throw "Static build is incomplete: frontend\dist\client\index.html was not produced."
}

$TextArtifacts = Get-ChildItem -LiteralPath $StaticRoot -Recurse -File |
    Where-Object { $_.Extension -in @(".html", ".js", ".json", ".rsc", ".txt") }
$LocalhostLeak = $TextArtifacts |
    Select-String -Pattern "(?:localhost|127\.0\.0\.1):8000" -List -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $LocalhostLeak) {
    throw "Static artifact still contains a localhost API reference: $($LocalhostLeak.Path)"
}

$OutputDirectory = Split-Path -Parent $OutputFullPath
if (-not (Test-Path -LiteralPath $OutputDirectory)) {
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
}
if (Test-Path -LiteralPath $OutputFullPath) {
    Remove-Item -LiteralPath $OutputFullPath -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $StaticRoot,
    $OutputFullPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

if (-not (Test-Path -LiteralPath $OutputFullPath -PathType Leaf)) {
    throw "Frontend archive was not created."
}

$Archive = Get-Item -LiteralPath $OutputFullPath
Write-Host "Frontend package ready: $($Archive.FullName) ($($Archive.Length) bytes)"
