[CmdletBinding()]
param([string]$SetupScript, [string]$CaseName)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not $SetupScript) { $SetupScript = Join-Path (Split-Path -Parent $PSScriptRoot) "setup.ps1" }
$SetupScript = (Resolve-Path -LiteralPath $SetupScript).Path
$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("ubike-setup-test-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TemporaryRoot | Out-Null

function Invoke-SetupCase {
    param([hashtable]$Case)

    $Fixture = Join-Path $TemporaryRoot $Case["Name"]
    $Scripts = Join-Path $Fixture "scripts"
    $Venv = Join-Path $Fixture ".venv\Scripts\python.exe"
    $Runtime = Join-Path $Fixture "runtime.cmd"
    New-Item -ItemType Directory -Path $Scripts, (Split-Path -Parent $Venv) -Force | Out-Null
    Copy-Item -LiteralPath $SetupScript -Destination (Join-Path $Scripts "setup.ps1")
    Set-Content -LiteralPath $Runtime -Value ""
    if ($Case["Existing"]) { Set-Content -LiteralPath $Venv -Value "" }

    $State = @{
        Calls = New-Object 'System.Collections.Generic.List[string]'
        Fail = $Case["Fail"]
        MissingOutput = $Case["MissingOutput"]
        Venv = $Venv
    }
    # Full-path function mocks intercept every Python invocation. No native
    # Python, network access, or files in the real repository's .venv are used.
    $FakePython = {
        param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PassedArgs)
        if ($PassedArgs -contains "-c") {
            $Stage = if ($MyInvocation.MyCommand.Name -eq $State.Venv) { "venv-check" } else { "runtime-check" }
        } elseif ($PassedArgs -contains "venv") {
            $Stage = "venv-create"
        } elseif ($PassedArgs -contains "--upgrade") {
            $Stage = "pip-upgrade"
        } else {
            $Stage = "requirements"
        }
        $State.Calls.Add($Stage)
        if ($Stage -eq $State.Fail) {
            $global:LASTEXITCODE = 23
            return
        }
        if ($Stage -eq "venv-create" -and -not $State.MissingOutput) {
            Set-Content -LiteralPath $State.Venv -Value ""
        }
        $global:LASTEXITCODE = 0
    }.GetNewClosure()
    Set-Item -LiteralPath ("Function:\" + $Runtime) -Value $FakePython
    Set-Item -LiteralPath ("Function:\" + $Venv) -Value $FakePython
    function Get-Command {
        [CmdletBinding()]
        param([string]$Name, [string]$CommandType)
        if ($Name -eq $Runtime -or ($Name -eq "py" -and -not $Case["NoLauncher"]) -or ($Name -eq "python" -and $Case["NoLauncher"])) {
            [pscustomobject]@{ Source = $Runtime }
        }
    }

    $Parameters = @{}
    if ($Case["Explicit"]) { $Parameters.PythonPath = $Runtime }
    if ($Case["MissingPath"]) { $Parameters.PythonPath = "missing-python-for-setup-test.exe" }
    $Output = New-Object 'System.Collections.Generic.List[string]'
    $Failure = $null
    try {
        & (Join-Path $Scripts "setup.ps1") @Parameters *>&1 | ForEach-Object { $Output.Add([string]$_) }
    } catch {
        $Failure = $_.Exception.Message
    } finally {
        Remove-Item -LiteralPath ("Function:\" + $Runtime), ("Function:\" + $Venv)
    }

    $Complete = ($Output -join "`n") -match "Setup complete"
    $ActualCalls = $State.Calls -join ","

    if ($Case["Error"]) {
        if (-not $Failure -or $Failure -notlike $Case["Error"] -or $Complete) {
            throw "$($Case['Name']): expected terminating error '$($Case['Error'])' without success message; got error '$Failure', complete=$Complete."
        }
    } elseif ($Failure -or -not $Complete) {
        throw "$($Case['Name']): expected success; got error '$Failure', complete=$Complete."
    }
    if ($ActualCalls -ne $Case["Calls"]) {
        throw "$($Case['Name']): expected calls '$($Case['Calls'])', got '$ActualCalls'. Error: $Failure"
    }
    Write-Output "PASS $($Case['Name']): $ActualCalls"
}

try {
    $Cases = @(
        @{ Name = "missing-explicit-runtime"; MissingPath = $true; Calls = ""; Error = "Python executable not found:*" }
        @{ Name = "launcher-runtime-unavailable"; Fail = "runtime-check"; Calls = "runtime-check"; Error = "Python check*exit code 23*" }
        @{ Name = "incompatible-existing-venv"; Existing = $true; Fail = "venv-check"; Calls = "venv-check"; Error = "Virtual environment Python check*exit code 23*" }
        @{ Name = "venv-creation-failed"; Fail = "venv-create"; Calls = "runtime-check,venv-create"; Error = "Virtual environment creation failed*exit code 23*" }
        @{ Name = "venv-output-missing"; MissingOutput = $true; Calls = "runtime-check,venv-create"; Error = "Virtual environment creation did not produce*" }
        @{ Name = "pip-upgrade-failed"; Existing = $true; Fail = "pip-upgrade"; Calls = "venv-check,pip-upgrade"; Error = "pip upgrade failed*exit code 23*" }
        @{ Name = "requirements-failed"; Existing = $true; Fail = "requirements"; Calls = "venv-check,pip-upgrade,requirements"; Error = "Requirements installation failed*exit code 23*" }
        @{ Name = "explicit-runtime-success"; Explicit = $true; Calls = "runtime-check,venv-create,venv-check,pip-upgrade,requirements" }
        @{ Name = "python-fallback-success"; NoLauncher = $true; Calls = "runtime-check,venv-create,venv-check,pip-upgrade,requirements" }
        @{ Name = "existing-venv-success"; Existing = $true; Calls = "venv-check,pip-upgrade,requirements" }
    )
    if ($CaseName) {
        $Cases = @($Cases | Where-Object { $_["Name"] -eq $CaseName })
        if ($Cases.Count -eq 0) { throw "Unknown test case: $CaseName" }
    }
    foreach ($Case in $Cases) { Invoke-SetupCase -Case $Case }
    Write-Output "All $($Cases.Count) setup diagnostic tests passed."
} finally {
    $ResolvedRoot = [IO.Path]::GetFullPath($TemporaryRoot)
    $ExpectedParent = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $ResolvedRoot.StartsWith($ExpectedParent, [StringComparison]::OrdinalIgnoreCase) -or (Split-Path -Leaf $ResolvedRoot) -notlike "ubike-setup-test-*") {
        throw "Refusing to remove unexpected fixture path: $ResolvedRoot"
    }
    Remove-Item -LiteralPath $ResolvedRoot -Recurse -Force
}
