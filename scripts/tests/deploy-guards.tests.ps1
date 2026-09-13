$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Tokens = $null
$ParseErrors = $null
$Ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $Repo 'scripts\deploy-aws.ps1'), [ref]$Tokens, [ref]$ParseErrors)
if ($ParseErrors.Count) { throw 'PowerShell parse failed' }
# Execute the real preflight only, stopping before SAM/AWS calls.
$Source = Get-Content -Raw -LiteralPath (Join-Path $Repo 'scripts\deploy-aws.ps1')
$RegionParam = $Ast.ParamBlock.Extent.Text
$CheckParameters = [scriptblock]::Create($RegionParam + "`n'accepted'")
foreach ($Bad in @('ap-northeast-1','us-east-2')) {
    try { & $CheckParameters -Region $Bad | Out-Null; throw 'accepted invalid region' } catch {
        if ($_.Exception.Message -eq 'accepted invalid region') { throw }
    }
    try { & $CheckParameters -BedrockRegion $Bad | Out-Null; throw 'accepted invalid Bedrock region' } catch {
        if ($_.Exception.Message -eq 'accepted invalid Bedrock region') { throw }
    }
}
foreach ($Good in @('us-east-1','us-west-2')) { & $CheckParameters -Region $Good -BedrockRegion $Good | Out-Null }
& $CheckParameters -BedrockRegion '' | Out-Null
$BedrockAst = $Ast.Find({param($Node) $Node -is [System.Management.Automation.Language.IfStatementAst] -and $Node.Clauses[0].Item1.Extent.Text -eq '$EnableBedrock'}, $true)
$CheckBedrock = [scriptblock]::Create($BedrockAst.Extent.Text)
$EnableBedrock = $true
$BedrockModelId = 'amazon.nova-lite-v1:0'
foreach ($BadArn in @('arn:aws:bedrock:us-east-2::foundation-model/test','arn:aws:bedrock:::foundation-model/test','arn:aws:bedrock:us-west-2::foundation-model/*')) {
    $BedrockModelResourceArns = @($BadArn)
    try { & $CheckBedrock; throw 'accepted invalid ARN' } catch {
        if ($_.Exception.Message -notlike 'Every BedrockModelResourceArn must be*') { throw }
    }
}
$BedrockModelResourceArns = @('arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-lite-v1:0')
& $CheckBedrock
$RepoRoot = $Repo
$Manifest = Get-Content -Raw -LiteralPath (Join-Path $Repo 'MANIFEST.json') | ConvertFrom-Json
$FunctionAst = $Ast.Find({param($Node) $Node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $Node.Name -eq 'Assert-FrozenAsset'}, $true)
. ([scriptblock]::Create($FunctionAst.Extent.Text))
Assert-FrozenAsset 'model\lgbm_full.txt'
$Entry = $Manifest.files | Where-Object path -eq 'model/lgbm_full.txt'
$OriginalHash = $Entry.sha256
$Entry.sha256 = '0' * 64
try { Assert-FrozenAsset 'model\lgbm_full.txt'; throw 'accepted invalid hash' } catch {
    if ($_.Exception.Message -notlike 'Frozen deployment asset bytes/SHA256 mismatch:*') { throw }
}
$Entry.sha256 = $OriginalHash
$Entry.bytes = 1
try { Assert-FrozenAsset 'model\lgbm_full.txt'; throw 'accepted invalid bytes' } catch {
    if ($_.Exception.Message -notlike 'Frozen deployment asset bytes/SHA256 mismatch:*') { throw }
}
$Manifest.files = @($Manifest.files | Where-Object path -ne 'model/lgbm_full.txt')
try { Assert-FrozenAsset 'model\lgbm_full.txt'; throw 'accepted missing entry' } catch {
    if ($_.Exception.Message -notlike 'Deployment asset must have exactly one manifest entry:*') { throw }
}
Write-Output 'PASS: valid regions/ARN accepted; invalid regions, out-of-region/global/wildcard ARNs, hash, bytes and missing manifest entry rejected. No AWS calls or asset edits.'
