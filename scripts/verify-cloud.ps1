[CmdletBinding()]
param(
    [string]$BaseUrl = "",
    [ValidatePattern("^[A-Za-z][A-Za-z0-9-]{0,63}$")]
    [string]$StackName = "ubikepredict-demo",
    [ValidatePattern("^[a-z]{2}(-gov)?-[a-z]+-[0-9]+$")]
    [string]$Region = "ap-northeast-1",
    [string]$Profile = "",
    [string]$Origin = "",
    [string]$LocalBaseUrl = "",
    [ValidateRange(0.000000001, 0.01)]
    [double]$ProbabilityTolerance = 0.0000002,
    [switch]$RequireBedrock,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-CloudBaseUrl {
    $Aws = Get-Command aws -ErrorAction SilentlyContinue
    if ($null -eq $Aws) {
        throw "BaseUrl was omitted and AWS CLI was not found."
    }
    $Arguments = @("--region", $Region, "--no-cli-pager")
    if (-not [string]::IsNullOrWhiteSpace($Profile)) {
        $Arguments = @("--profile", $Profile) + $Arguments
    }
    $Output = & $Aws.Source @Arguments cloudformation describe-stacks `
        --stack-name $StackName `
        --query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue | [0]" `
        --output text 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read ApiBaseUrl from stack $StackName`: $($Output -join ' ')"
    }
    return [string]($Output -join "").Trim()
}

if ([string]::IsNullOrWhiteSpace($BaseUrl)) {
    $BaseUrl = Get-CloudBaseUrl
}
$BaseUrl = $BaseUrl.TrimEnd("/")
$ParsedBaseUrl = $null
if (-not [uri]::TryCreate($BaseUrl, [System.UriKind]::Absolute, [ref]$ParsedBaseUrl)) {
    throw "BaseUrl is not an absolute URL."
}
if ($ParsedBaseUrl.Scheme -ne "https" -and -not $ParsedBaseUrl.IsLoopback) {
    throw "Cloud verification requires https."
}

$Endpoints = @(
    "$BaseUrl/api/health",
    "$BaseUrl/api/options",
    "$BaseUrl/api/predict",
    "$BaseUrl/api/explain",
    "$BaseUrl/api/reveal"
)
if ($DryRun) {
    Write-Host "DRY RUN - no HTTP requests will be made."
    $Endpoints | ForEach-Object { Write-Host $_ }
    exit 0
}

$Headers = @{}
if (-not [string]::IsNullOrWhiteSpace($Origin)) {
    $Headers["Origin"] = $Origin.TrimEnd("/")
}

$Health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/health" -Headers $Headers -TimeoutSec 30
if ($Health.status -ne "ok") {
    throw "Health endpoint did not return status=ok."
}
$null = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/options" -Headers $Headers -TimeoutSec 30
$FullDockOptions = Invoke-RestMethod `
    -Method Get `
    -Uri "$BaseUrl/api/options?mode=full_dock" `
    -Headers $Headers `
    -TimeoutSec 30
if ([string]$FullDockOptions.mode -ne "full_dock") {
    throw "Full-dock options did not return mode=full_dock."
}

$PredictBody = @{
    mode = "empty"
    decision_time = "2026-06-29T09:00:00+08:00"
    district = "三重區"
    policy = "balanced"
    action_limit = 10
} | ConvertTo-Json
$Prediction = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/predict" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $PredictBody `
    -TimeoutSec 30

$ExpectedSummary = @{
    current_empty = 23
    scored_current_empty = 21
    alerts_before_limit = 12
    action_count = 10
}
foreach ($Name in $ExpectedSummary.Keys) {
    if ([int]$Prediction.summary.$Name -ne $ExpectedSummary[$Name]) {
        throw "Prediction summary mismatch for $Name. Expected $($ExpectedSummary[$Name]), got $($Prediction.summary.$Name)."
    }
}

$StationIds = @($Prediction.actions | ForEach-Object { [int]$_.station_id })
if ($StationIds.Count -ne 10 -or ($StationIds | Sort-Object -Unique).Count -ne 10) {
    throw "Prediction action list must contain 10 unique station IDs."
}
$ExplainBody = @{
    decision_time = [string]$Prediction.decision_time
    district = "三重區"
    station_id = $StationIds[0]
    policy = "balanced"
    action_limit = 10
} | ConvertTo-Json
$Explanation = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/explain" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $ExplainBody `
    -TimeoutSec 30
if ([int]$Explanation.station.station_id -ne $StationIds[0]) {
    throw "Explanation station mismatch."
}
if ([math]::Abs([double]$Explanation.shap.sum_error) -gt 0.000001) {
    throw "SHAP contributions do not reconstruct the LightGBM raw score."
}
$SummaryProvider = [string]$Explanation.operational_summary.provider
if ($SummaryProvider -notin @("template", "amazon_bedrock")) {
    throw "Unexpected explanation provider '$SummaryProvider'."
}
if ($RequireBedrock -and $SummaryProvider -ne "amazon_bedrock") {
    $FallbackReason = [string]$Explanation.operational_summary.fallback_reason
    throw "Bedrock was required, but the explanation used '$SummaryProvider' (fallback: '$FallbackReason')."
}
$RevealBody = @{
    mode = "empty"
    decision_time = [string]$Prediction.decision_time
    station_ids = $StationIds
} | ConvertTo-Json -Depth 4
$Reveal = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/reveal" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $RevealBody `
    -TimeoutSec 30
if ([int]$Reveal.summary.evaluated_actions -ne 10 -or [int]$Reveal.summary.hits -ne 7) {
    throw "Reveal mismatch. Expected 7 hits from 10 evaluated actions."
}
if ([math]::Abs([double]$Reveal.summary.precision - 0.7) -gt 0.000000001) {
    throw "Reveal precision mismatch. Expected 0.7."
}

# Auxiliary full-dock mode uses the same endpoints and deployment.  Its
# representative replay is intentionally modest: two alerts and one hit.
$FullDockPredictBody = @{
    mode = "full_dock"
    decision_time = "2026-06-25T08:00:00+08:00"
    district = "板橋區"
    policy = "balanced"
    action_limit = 10
} | ConvertTo-Json
$FullDockPrediction = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/predict" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $FullDockPredictBody `
    -TimeoutSec 30
if ([string]$FullDockPrediction.mode -ne "full_dock") {
    throw "Full-dock prediction mode mismatch."
}
$FullDockExpectedSummary = @{
    current_full_dock = 6
    scored_current_full_dock = 6
    alerts_before_limit = 2
    action_count = 2
}
foreach ($Name in $FullDockExpectedSummary.Keys) {
    if ([int]$FullDockPrediction.summary.$Name -ne $FullDockExpectedSummary[$Name]) {
        throw "Full-dock summary mismatch for $Name. Expected $($FullDockExpectedSummary[$Name]), got $($FullDockPrediction.summary.$Name)."
    }
}
$FullDockStationIds = @(
    $FullDockPrediction.actions | ForEach-Object { [int]$_.station_id }
)
if ($FullDockStationIds.Count -ne 2 -or
    ($FullDockStationIds | Sort-Object -Unique).Count -ne 2) {
    throw "Full-dock action list must contain 2 unique station IDs."
}
$FullDockExplainBody = @{
    mode = "full_dock"
    decision_time = [string]$FullDockPrediction.decision_time
    district = "板橋區"
    station_id = $FullDockStationIds[0]
    policy = "balanced"
    action_limit = 10
} | ConvertTo-Json
$FullDockExplanation = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/explain" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $FullDockExplainBody `
    -TimeoutSec 30
if ([string]$FullDockExplanation.mode -ne "full_dock" -or
    [int]$FullDockExplanation.station.station_id -ne $FullDockStationIds[0]) {
    throw "Full-dock explanation mode or station mismatch."
}
if ([math]::Abs([double]$FullDockExplanation.shap.sum_error) -gt 0.000001) {
    throw "Full-dock SHAP contributions do not reconstruct the LightGBM raw score."
}
$FullDockProvider = [string]$FullDockExplanation.operational_summary.provider
if ($FullDockProvider -notin @("template", "amazon_bedrock")) {
    throw "Unexpected full-dock explanation provider '$FullDockProvider'."
}
if ($RequireBedrock -and $FullDockProvider -ne "amazon_bedrock") {
    $FullDockFallback = [string]$FullDockExplanation.operational_summary.fallback_reason
    throw "Bedrock was required for full_dock, but provider was '$FullDockProvider' (fallback: '$FullDockFallback')."
}
$FullDockRevealBody = @{
    mode = "full_dock"
    decision_time = [string]$FullDockPrediction.decision_time
    station_ids = $FullDockStationIds
} | ConvertTo-Json -Depth 4
$FullDockReveal = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/reveal" `
    -Headers $Headers `
    -ContentType "application/json; charset=utf-8" `
    -Body $FullDockRevealBody `
    -TimeoutSec 30
if ([string]$FullDockReveal.mode -ne "full_dock" -or
    [int]$FullDockReveal.summary.evaluated_actions -ne 2 -or
    [int]$FullDockReveal.summary.hits -ne 1) {
    throw "Full-dock reveal mismatch. Expected 1 hit from 2 evaluated actions."
}
if ([math]::Abs([double]$FullDockReveal.summary.precision - 0.5) -gt 0.000000001) {
    throw "Full-dock reveal precision mismatch. Expected 0.5."
}

if (-not [string]::IsNullOrWhiteSpace($Origin)) {
    $FrontendResponse = Invoke-WebRequest `
        -Method Get `
        -Uri $Origin.TrimEnd("/") `
        -TimeoutSec 30
    if ([int]$FrontendResponse.StatusCode -ne 200 -or
        [string]$FrontendResponse.Content -notmatch "YouBike") {
        throw "The Amplify frontend did not return the expected YouBike page."
    }

    $CorsResponse = Invoke-WebRequest `
        -Method Options `
        -Uri "$BaseUrl/api/predict" `
        -Headers @{
            Origin = $Origin.TrimEnd("/")
            "Access-Control-Request-Method" = "POST"
            "Access-Control-Request-Headers" = "content-type"
        } `
        -TimeoutSec 30
    $AllowedOrigin = [string]$CorsResponse.Headers["Access-Control-Allow-Origin"]
    if ($AllowedOrigin -ne $Origin.TrimEnd("/")) {
        throw "CORS mismatch. Expected '$($Origin.TrimEnd('/'))', got '$AllowedOrigin'."
    }
}

if (-not [string]::IsNullOrWhiteSpace($LocalBaseUrl)) {
    $LocalBaseUrl = $LocalBaseUrl.TrimEnd("/")
    $LocalPrediction = Invoke-RestMethod `
        -Method Post `
        -Uri "$LocalBaseUrl/api/predict" `
        -ContentType "application/json; charset=utf-8" `
        -Body $PredictBody `
        -TimeoutSec 30
    $LocalByStation = @{}
    foreach ($Action in $LocalPrediction.actions) {
        $LocalByStation[[string]$Action.station_id] = [double]$Action.risk_probability
    }
    foreach ($Action in $Prediction.actions) {
        $Key = [string]$Action.station_id
        if (-not $LocalByStation.ContainsKey($Key)) {
            throw "Cloud/local parity mismatch: station $Key is absent locally."
        }
        $Difference = [math]::Abs([double]$Action.risk_probability - $LocalByStation[$Key])
        if ($Difference -gt $ProbabilityTolerance) {
            throw "Cloud/local probability mismatch for station $Key`: $Difference exceeds $ProbabilityTolerance."
        }
    }

    $LocalFullDockPrediction = Invoke-RestMethod `
        -Method Post `
        -Uri "$LocalBaseUrl/api/predict" `
        -ContentType "application/json; charset=utf-8" `
        -Body $FullDockPredictBody `
        -TimeoutSec 30
    $LocalFullDockByStation = @{}
    foreach ($Action in $LocalFullDockPrediction.actions) {
        $LocalFullDockByStation[[string]$Action.station_id] = [double]$Action.risk_probability
    }
    foreach ($Action in $FullDockPrediction.actions) {
        $Key = [string]$Action.station_id
        if (-not $LocalFullDockByStation.ContainsKey($Key)) {
            throw "Full-dock cloud/local parity mismatch: station $Key is absent locally."
        }
        $Difference = [math]::Abs(
            [double]$Action.risk_probability - $LocalFullDockByStation[$Key]
        )
        if ($Difference -gt $ProbabilityTolerance) {
            throw "Full-dock cloud/local probability mismatch for station $Key`: $Difference exceeds $ProbabilityTolerance."
        }
    }
}

Write-Host "Cloud verification passed: empty 12-to-Top-10 and 7/10 reveal; full_dock 2 alerts and 1/2 reveal; both SHAP/summaries; optional Amplify/CORS and parity checks."
