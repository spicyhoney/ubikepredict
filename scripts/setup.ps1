[CmdletBinding()]
param(
    # Used when creating .venv; an existing .venv is checked independently.
    [string]$PythonPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$PythonCheck = "import struct, sys; print('Python ' + sys.version.split()[0] + ' (' + str(struct.calcsize('P') * 8) + '-bit)'); sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)"

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$Arguments, [string]$Step)

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed (exit code $LASTEXITCODE). Setup did not complete."
    }
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonArguments = @()
    if ($PythonPath) {
        $Python = Get-Command $PythonPath -CommandType Application -ErrorAction SilentlyContinue
        if ($null -eq $Python) {
            throw "Python executable not found: $PythonPath. Supply -PythonPath with a 64-bit Python 3.12 executable."
        }
    } else {
        $Python = Get-Command py -CommandType Application -ErrorAction SilentlyContinue
        if ($null -ne $Python) {
            $PythonArguments = @("-3.12")
        } else {
            $Python = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
        }
        if ($null -eq $Python) {
            throw "Python not found. Install 64-bit Python 3.12 or supply -PythonPath."
        }
    }

    Invoke-CheckedPython -Executable $Python.Source -Arguments ($PythonArguments + @("-c", $PythonCheck)) -Step "Python check (64-bit Python 3.12 required; use -PythonPath to select it)"
    Invoke-CheckedPython -Executable $Python.Source -Arguments ($PythonArguments + @("-m", "venv", $VenvDir)) -Step "Virtual environment creation"
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Virtual environment creation did not produce $VenvPython. Setup did not complete."
    }
}

Invoke-CheckedPython -Executable $VenvPython -Arguments @("-c", $PythonCheck) -Step "Virtual environment Python check (64-bit Python 3.12 required)"
Invoke-CheckedPython -Executable $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip") -Step "pip upgrade"
Invoke-CheckedPython -Executable $VenvPython -Arguments @("-m", "pip", "install", "-r", (Join-Path $RepoRoot "requirements.txt")) -Step "Requirements installation"
Write-Host "Setup complete. Next run scripts\smoke-test.ps1."
