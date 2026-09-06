[CmdletBinding()]
param(
    [ValidatePattern("^[A-Za-z][A-Za-z0-9-]{0,63}$")]
    [string]$StackName = "ubikepredict-demo",
    [ValidatePattern("^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$")]
    [string]$Region = "ap-northeast-1",
    [string]$Profile = "",
    [ValidatePattern("^[A-Za-z0-9_-]+$")]
    [string]$StageName = "prod",
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
    [string]$ReleaseId = "frozen-v1",
    [ValidatePattern("^[A-Za-z0-9-]+$")]
    [string]$FrontendBranchName = "main",
    [string]$FrontendOriginOverride = "",
    [string]$LocalDevelopmentOrigin = "http://localhost:3000",
    [switch]$EnableBedrock,
    [string]$BedrockModelId = "",
    [string[]]$BedrockModelResourceArns = @(),
    [string]$BedrockRegion = "",
    [ValidateRange(1, 15)]
    [int]$BedrockTimeoutSeconds = 6,
    [ValidateRange(1, 15)]
    [int]$S3TimeoutSeconds = 8,
    [switch]$SkipAssetUpload,
    [switch]$SkipFrontend,
    [switch]$SkipVerify,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$TemplatePath = Join-Path $RepoRoot "infra\template.yaml"
$BuildRoot = Join-Path $RepoRoot ".aws-sam\build"
$FrontendArchive = Join-Path $RepoRoot ".aws-artifacts\frontend.zip"

function Require-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        throw "$Name was not found on PATH."
    }
    return $Command.Source
}

$Aws = Require-Command "aws"
$Sam = Require-Command "sam"
if (-not $DryRun) {
    $null = Require-Command "docker"
}

function Get-AwsBaseArguments {
    $Base = @("--region", $Region, "--no-cli-pager")
    if (-not [string]::IsNullOrWhiteSpace($Profile)) {
        $Base = @("--profile", $Profile) + $Base
    }
    return $Base
}

function Invoke-AwsJson {
    param([Parameter(Mandatory = $true)][string[]]$CliArguments)
    $Output = & $Aws @((Get-AwsBaseArguments) + $CliArguments + @("--output", "json")) 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed:`n$($Output -join [Environment]::NewLine)"
    }
    return (($Output -join [Environment]::NewLine) | ConvertFrom-Json)
}

function Invoke-AwsCommand {
    param([Parameter(Mandatory = $true)][string[]]$CliArguments)
    & $Aws @((Get-AwsBaseArguments) + $CliArguments)
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed with exit code ${LASTEXITCODE}: aws $($CliArguments -join ' ')"
    }
}

function Get-StackOutputs {
    $Stack = Invoke-AwsJson @(
        "cloudformation", "describe-stacks",
        "--stack-name", $StackName,
        "--query", "Stacks[0]"
    )
    $Values = @{}
    foreach ($Output in $Stack.Outputs) {
        $Values[$Output.OutputKey] = [string]$Output.OutputValue
    }
    return $Values
}

if ($EnableBedrock) {
    if ([string]::IsNullOrWhiteSpace($BedrockModelId)) {
        throw "EnableBedrock requires BedrockModelId."
    }
    if ($BedrockModelResourceArns.Count -eq 0) {
        throw "EnableBedrock requires at least one exact BedrockModelResourceArn for least-privilege IAM."
    }
    foreach ($ResourceArn in $BedrockModelResourceArns) {
        if ([string]::IsNullOrWhiteSpace($ResourceArn) -or $ResourceArn -notmatch "^arn:[^:]+:bedrock:") {
            throw "Every BedrockModelResourceArn must be an exact Bedrock ARN."
        }
    }
}
if (-not [string]::IsNullOrWhiteSpace($BedrockRegion) -and $BedrockRegion -notmatch "^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$") {
    throw "BedrockRegion is not a valid AWS region name."
}
if (-not [string]::IsNullOrWhiteSpace($FrontendOriginOverride) -and $FrontendOriginOverride -notmatch "^https://[^/]+$") {
    throw "FrontendOriginOverride must be one exact https origin with no trailing slash."
}
if ($LocalDevelopmentOrigin -notmatch "^https?://[^/]+$") {
    throw "LocalDevelopmentOrigin must be one exact origin with no trailing slash."
}

$RequiredAssets = @(
    "model\lgbm_full.txt",
    "config\final_policy_freeze_before_may.json",
    "config\protocol_frozen_before_june.json",
    "data\source\dynamic_red_empty_2026_06_input.parquet",
    "data\stations\dim_station.csv",
    "data\reference\june_all_eligible_decisions.parquet"
)
foreach ($RelativePath in $RequiredAssets) {
    $FullPath = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "Required deployment asset is missing: $RelativePath"
    }
}

$ModelKey = "releases/$ReleaseId/model/lgbm_full.txt"
$FreezeKey = "releases/$ReleaseId/config/final_policy_freeze_before_may.json"
$ProtocolKey = "releases/$ReleaseId/config/protocol_frozen_before_june.json"
$InputKey = "replays/2026-06/input/dynamic_red_empty_2026_06_input.parquet"
$StationsKey = "stations/dim_station.csv"
$TruthKey = "replays/2026-06/truth/june_all_eligible_decisions.parquet"

$ParameterOverrides = @(
    "ParameterKey=StageName,ParameterValue=$StageName",
    "ParameterKey=ReleaseId,ParameterValue=$ReleaseId",
    "ParameterKey=FrontendBranchName,ParameterValue=$FrontendBranchName",
    "ParameterKey=LocalDevelopmentOrigin,ParameterValue=$LocalDevelopmentOrigin",
    "ParameterKey=BedrockEnabled,ParameterValue=$($EnableBedrock.IsPresent.ToString().ToLowerInvariant())",
    "ParameterKey=BedrockTimeoutSeconds,ParameterValue=$BedrockTimeoutSeconds",
    "ParameterKey=S3TimeoutSeconds,ParameterValue=$S3TimeoutSeconds"
)
if (-not [string]::IsNullOrWhiteSpace($FrontendOriginOverride)) {
    $ParameterOverrides += "ParameterKey=FrontendOriginOverride,ParameterValue=$FrontendOriginOverride"
}
if ($EnableBedrock) {
    $ParameterOverrides += @(
        "ParameterKey=BedrockModelId,ParameterValue=$BedrockModelId",
        "ParameterKey=BedrockModelResourceArns,ParameterValue=$($BedrockModelResourceArns -join ',')"
    )
}
if (-not [string]::IsNullOrWhiteSpace($BedrockRegion)) {
    $ParameterOverrides += "ParameterKey=BedrockRegion,ParameterValue=$BedrockRegion"
}

Write-Host "Validating SAM template..."
& $Sam validate --template-file $TemplatePath
if ($LASTEXITCODE -ne 0) {
    throw "SAM template validation failed."
}

if ($DryRun) {
    Write-Host "DRY RUN - no AWS resources or S3 objects will be changed."
    Write-Host "Stack: $StackName"
    Write-Host "Region: $Region"
    Write-Host "Bedrock enabled: $($EnableBedrock.IsPresent)"
    Write-Host "Planned runtime objects: $ModelKey, $FreezeKey, $ProtocolKey, $InputKey, $StationsKey"
    Write-Host "Planned truth object: $TruthKey"
    Write-Host "Planned flow: SAM build/deploy -> exact S3 uploads -> static frontend package -> Amplify manual deployment -> cloud verification."
    exit 0
}

$Identity = Invoke-AwsJson @("sts", "get-caller-identity")
Write-Host "AWS account confirmed: $($Identity.Account) (credentials are not written to the repository)."

Write-Host "Building Linux Lambda container images through SAM..."
& $Sam build `
    --template-file $TemplatePath `
    --build-dir $BuildRoot `
    --cached `
    --parallel
if ($LASTEXITCODE -ne 0) {
    throw "SAM build failed. Confirm Docker Desktop is running in Linux-container mode."
}

$BuiltTemplate = Join-Path $BuildRoot "template.yaml"
$SamDeployArguments = @(
    "deploy",
    "--template-file", $BuiltTemplate,
    "--stack-name", $StackName,
    "--region", $Region,
    "--capabilities", "CAPABILITY_IAM",
    "--resolve-s3",
    "--resolve-image-repos",
    "--no-confirm-changeset",
    "--no-fail-on-empty-changeset"
)
if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    $SamDeployArguments += @("--profile", $Profile)
}
$SamDeployArguments += @("--parameter-overrides") + $ParameterOverrides

Write-Host "Deploying CloudFormation stack..."
& $Sam @SamDeployArguments
if ($LASTEXITCODE -ne 0) {
    throw "SAM deployment failed."
}

$Outputs = Get-StackOutputs
$RuntimeBucket = $Outputs["RuntimeBucketName"]
$TruthBucket = $Outputs["TruthBucketName"]
$ApiBaseUrl = $Outputs["ApiBaseUrl"].TrimEnd("/")
$AmplifyAppId = $Outputs["AmplifyAppId"]
$AmplifyBranch = $Outputs["AmplifyBranchName"]
$FrontendUrl = $Outputs["FrontendUrl"].TrimEnd("/")

if (-not $SkipAssetUpload) {
    Write-Host "Uploading frozen runtime assets and held-out truth to separate private buckets..."
    $Uploads = @(
        @{ Local = "model\lgbm_full.txt"; Bucket = $RuntimeBucket; Key = $ModelKey },
        @{ Local = "config\final_policy_freeze_before_may.json"; Bucket = $RuntimeBucket; Key = $FreezeKey },
        @{ Local = "config\protocol_frozen_before_june.json"; Bucket = $RuntimeBucket; Key = $ProtocolKey },
        @{ Local = "data\source\dynamic_red_empty_2026_06_input.parquet"; Bucket = $RuntimeBucket; Key = $InputKey },
        @{ Local = "data\stations\dim_station.csv"; Bucket = $RuntimeBucket; Key = $StationsKey },
        @{ Local = "data\reference\june_all_eligible_decisions.parquet"; Bucket = $TruthBucket; Key = $TruthKey }
    )
    foreach ($Upload in $Uploads) {
        Invoke-AwsCommand @(
            "s3", "cp",
            (Join-Path $RepoRoot $Upload.Local),
            "s3://$($Upload.Bucket)/$($Upload.Key)",
            "--only-show-errors"
        )
    }
}

if (-not $SkipFrontend) {
    Write-Host "Building the static frontend against the cloud API..."
    & (Join-Path $PSScriptRoot "package-frontend.ps1") `
        -ApiBaseUrl $ApiBaseUrl `
        -OutputPath $FrontendArchive

    $Deployment = Invoke-AwsJson @(
        "amplify", "create-deployment",
        "--app-id", $AmplifyAppId,
        "--branch-name", $AmplifyBranch
    )
    if ($null -eq $Deployment.PSObject.Properties["zipUploadUrl"] -or
        [string]::IsNullOrWhiteSpace([string]$Deployment.zipUploadUrl)) {
        throw "Amplify did not return a zip upload URL. Confirm the branch is configured for manual deployment."
    }
    Invoke-WebRequest -Uri $Deployment.zipUploadUrl -Method Put -InFile $FrontendArchive | Out-Null
    Invoke-AwsCommand @(
        "amplify", "start-deployment",
        "--app-id", $AmplifyAppId,
        "--branch-name", $AmplifyBranch,
        "--job-id", [string]$Deployment.jobId
    )

    $Deadline = [DateTime]::UtcNow.AddMinutes(10)
    do {
        Start-Sleep -Seconds 5
        $Job = Invoke-AwsJson @(
            "amplify", "get-job",
            "--app-id", $AmplifyAppId,
            "--branch-name", $AmplifyBranch,
            "--job-id", [string]$Deployment.jobId
        )
        $JobStatus = [string]$Job.job.summary.status
        Write-Host "Amplify deployment status: $JobStatus"
        if ($JobStatus -in @("FAILED", "CANCELLED")) {
            throw "Amplify deployment ended with status $JobStatus."
        }
    } while ($JobStatus -ne "SUCCEED" -and [DateTime]::UtcNow -lt $Deadline)
    if ($JobStatus -ne "SUCCEED") {
        throw "Amplify deployment did not finish within 10 minutes."
    }
}

if (-not $SkipVerify) {
    & (Join-Path $PSScriptRoot "verify-cloud.ps1") `
        -BaseUrl $ApiBaseUrl `
        -Origin $FrontendUrl `
        -RequireBedrock:$EnableBedrock.IsPresent
}

Write-Host "Deployment complete."
Write-Host "API: $ApiBaseUrl"
Write-Host "Frontend: $FrontendUrl"
