[CmdletBinding()]
param(
    [ValidatePattern("^[A-Za-z][A-Za-z0-9-]{0,63}$")]
    [string]$StackName = "ubikepredict-demo",
    [ValidatePattern("^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$")]
    [string]$Region = "us-east-1",
    [string]$Profile = "",
    [switch]$Execute,
    [switch]$DeleteBucketContents
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AwsCommand = Get-Command aws -ErrorAction SilentlyContinue
if ($null -eq $AwsCommand) {
    throw "AWS CLI was not found on PATH."
}
$Aws = $AwsCommand.Source
$SamCommand = Get-Command sam -ErrorAction SilentlyContinue
if ($null -eq $SamCommand) {
    throw "AWS SAM CLI was not found on PATH."
}
$Sam = $SamCommand.Source

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

$Stack = Invoke-AwsJson @("cloudformation", "describe-stacks", "--stack-name", $StackName)
$StackId = [string]$Stack.Stacks[0].StackId
if ([string]::IsNullOrWhiteSpace($StackId)) {
    throw "Stack $StackName was not found."
}

function Get-VerifiedStackBucket {
    param([Parameter(Mandatory = $true)][string]$LogicalResourceId)

    $Detail = Invoke-AwsJson @(
        "cloudformation", "describe-stack-resource",
        "--stack-name", $StackName,
        "--logical-resource-id", $LogicalResourceId
    )
    $Resource = $Detail.StackResourceDetail
    if ([string]$Resource.StackId -ne $StackId) {
        throw "$LogicalResourceId does not belong to the resolved stack ID."
    }
    if ([string]$Resource.ResourceType -ne "AWS::S3::Bucket") {
        throw "$LogicalResourceId is not an S3 bucket."
    }
    $BucketName = [string]$Resource.PhysicalResourceId
    if ([string]::IsNullOrWhiteSpace($BucketName)) {
        throw "$LogicalResourceId has no physical bucket name."
    }

    $TagResponse = Invoke-AwsJson @("s3api", "get-bucket-tagging", "--bucket", $BucketName)
    $Tags = @{}
    foreach ($Tag in $TagResponse.TagSet) {
        $Tags[[string]$Tag.Key] = [string]$Tag.Value
    }
    if ($Tags["Project"] -ne "ubikepredict" -or
        $Tags["ManagedBy"] -ne "CloudFormation" -or
        $Tags["StackName"] -ne $StackName) {
        throw "Safety check failed for bucket '$BucketName'; required project/stack tags are absent."
    }
    return $BucketName
}

$RuntimeBucket = Get-VerifiedStackBucket "RuntimeBucket"
$TruthBucket = Get-VerifiedStackBucket "TruthBucket"
$VerifiedBuckets = @($RuntimeBucket, $TruthBucket)

Write-Host "Verified CloudFormation stack: $StackId"
Write-Host "Verified stack-owned buckets: $($VerifiedBuckets -join ', ')"

if (-not $Execute) {
    Write-Host "DRY RUN - nothing was deleted."
    Write-Host "To remove the stack and its versioned objects, rerun with -Execute -DeleteBucketContents."
    exit 0
}
if (-not $DeleteBucketContents) {
    throw "Refusing cleanup without -DeleteBucketContents. The stack owns versioned buckets that CloudFormation cannot delete while non-empty."
}

function Remove-AllBucketVersions {
    param([Parameter(Mandatory = $true)][string]$BucketName)

    while ($true) {
        $Listing = Invoke-AwsJson @("s3api", "list-object-versions", "--bucket", $BucketName)
        $Objects = @()
        $Versions = if ($null -ne $Listing.PSObject.Properties["Versions"]) {
            @($Listing.Versions)
        } else {
            @()
        }
        $DeleteMarkers = if ($null -ne $Listing.PSObject.Properties["DeleteMarkers"]) {
            @($Listing.DeleteMarkers)
        } else {
            @()
        }
        foreach ($Version in $Versions) {
            if ($null -ne $Version) {
                $Objects += @{ Key = [string]$Version.Key; VersionId = [string]$Version.VersionId }
            }
        }
        foreach ($Marker in $DeleteMarkers) {
            if ($null -ne $Marker) {
                $Objects += @{ Key = [string]$Marker.Key; VersionId = [string]$Marker.VersionId }
            }
        }
        if ($Objects.Count -eq 0) {
            break
        }

        foreach ($BatchStart in 0..([math]::Floor(($Objects.Count - 1) / 1000))) {
            $Start = $BatchStart * 1000
            $End = [math]::Min($Start + 999, $Objects.Count - 1)
            $Batch = @($Objects[$Start..$End])
            $Payload = @{ Objects = $Batch; Quiet = $true } | ConvertTo-Json -Depth 5 -Compress
            $TemporaryFile = [System.IO.Path]::GetTempFileName()
            try {
                [System.IO.File]::WriteAllText(
                    $TemporaryFile,
                    $Payload,
                    [System.Text.UTF8Encoding]::new($false)
                )
                Invoke-AwsCommand @(
                    "s3api", "delete-objects",
                    "--bucket", $BucketName,
                    "--delete", "file://$TemporaryFile"
                )
            } finally {
                Remove-Item -LiteralPath $TemporaryFile -Force -ErrorAction SilentlyContinue
            }
        }
    }
    Write-Host "Removed all object versions from verified stack bucket: $BucketName"
}

foreach ($Bucket in $VerifiedBuckets) {
    Remove-AllBucketVersions $Bucket
}

$SamArguments = @(
    "delete",
    "--stack-name", $StackName,
    "--region", $Region,
    "--no-prompts"
)
if (-not [string]::IsNullOrWhiteSpace($Profile)) {
    $SamArguments += @("--profile", $Profile)
}
& $Sam @SamArguments
if ($LASTEXITCODE -ne 0) {
    throw "SAM cleanup failed with exit code $LASTEXITCODE."
}
Write-Host "Deleted SAM/CloudFormation stack $StackName, its managed image repositories, and its two verified data buckets."
