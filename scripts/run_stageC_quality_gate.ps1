param(
    [switch]$SkipStage0,
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-GateResult {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Details
    )

    return [PSCustomObject]@{
        check = $Name
        ok = $Ok
        details = $Details
    }
}

function Test-RequiredFile {
    param([string]$Path)

    if (Test-Path $Path) {
        return New-GateResult -Name "FileExists:$Path" -Ok $true -Details "found"
    }

    return New-GateResult -Name "FileExists:$Path" -Ok $false -Details "missing"
}

function Test-Stage0RawLoadQuality {
    param([string]$ReportPath)

    $checks = New-Object System.Collections.ArrayList
    if (-not (Test-Path $ReportPath)) {
        $null = $checks.Add((New-GateResult -Name "Stage0:RawLoadReportReadable" -Ok $false -Details "missing"))
        return $checks
    }

    try {
        $report = Get-Content $ReportPath -Raw | ConvertFrom-Json
    } catch {
        $null = $checks.Add((New-GateResult -Name "Stage0:RawLoadReportReadable" -Ok $false -Details "invalid json"))
        return $checks
    }

    $null = $checks.Add((New-GateResult -Name "Stage0:RawLoadReportReadable" -Ok $true -Details "ok"))

    $nodeFile = [string]$report.node_file
    $nodeFileOk = -not [string]::IsNullOrWhiteSpace($nodeFile)
    $nodeFileDetails = if ($nodeFileOk) { "node_file=$nodeFile" } else { "node_file missing" }
    $null = $checks.Add((New-GateResult -Name "Stage0:NodeMetadataDetected" -Ok $nodeFileOk -Details $nodeFileDetails))

    $viewsAvailable = 0
    if ($report.PSObject.Properties.Name -contains "views_available") {
        $viewsAvailable = [int]$report.views_available
    }
    $viewsOk = $viewsAvailable -gt 0
    $null = $checks.Add((New-GateResult -Name "Stage0:ViewsAvailable" -Ok $viewsOk -Details "views_available=$viewsAvailable"))

    if ($report.PSObject.Properties.Name -contains "node_edge_overlap_ratio") {
        $overlapRatio = [double]$report.node_edge_overlap_ratio
        $overlapOk = $overlapRatio -ge 0.95
        $null = $checks.Add((New-GateResult -Name "Stage0:NodeEdgeOverlap" -Ok $overlapOk -Details ("ratio=" + $overlapRatio.ToString("F4"))))
    } else {
        $null = $checks.Add((New-GateResult -Name "Stage0:NodeEdgeOverlap" -Ok $false -Details "missing node_edge_overlap_ratio"))
    }

    if (($report.PSObject.Properties.Name -contains "n_nodes_raw") -and ($report.PSObject.Properties.Name -contains "n_nodes_unique")) {
        $nRaw = [int]$report.n_nodes_raw
        $nUnique = [int]$report.n_nodes_unique
        $dedupOk = ($nRaw -eq $nUnique)
        $null = $checks.Add((New-GateResult -Name "Stage0:NodeIdsDeduplicated" -Ok $dedupOk -Details "raw=$nRaw; unique=$nUnique"))
    }

    return $checks
}

function Test-Stage0PreprocessQuality {
    param([string]$MetricsPath)

    $checks = New-Object System.Collections.ArrayList
    if (-not (Test-Path $MetricsPath)) {
        $null = $checks.Add((New-GateResult -Name "Stage0:PreprocessMetricsReadable" -Ok $false -Details "missing"))
        return $checks
    }

    try {
        $metrics = Get-Content $MetricsPath -Raw | ConvertFrom-Json
    } catch {
        $null = $checks.Add((New-GateResult -Name "Stage0:PreprocessMetricsReadable" -Ok $false -Details "invalid json"))
        return $checks
    }

    $null = $checks.Add((New-GateResult -Name "Stage0:PreprocessMetricsReadable" -Ok $true -Details "ok"))

    $nActive = [int]$metrics.n_nodes_active_graph
    $nFilled = [int]$metrics.n_missing_views_filled
    if ($nActive -gt 0) {
        $filledRatio = [double]$nFilled / [double]$nActive
        $ratioOk = $filledRatio -lt 0.98
        $null = $checks.Add((New-GateResult -Name "Stage0:MissingViewsFillRatio" -Ok $ratioOk -Details ("ratio=" + $filledRatio.ToString("F4"))))
    } else {
        $null = $checks.Add((New-GateResult -Name "Stage0:MissingViewsFillRatio" -Ok $false -Details "n_nodes_active_graph=0"))
    }

    return $checks
}

function Invoke-Stage {
    param(
        [int]$Stage,
        [string]$PythonExePath
    )

    $start = Get-Date
    Write-Host "[Stage $Stage] Starting..." -ForegroundColor Cyan
    & $PythonExePath run_all.py --stage $Stage | Out-Host
    $exitCode = $LASTEXITCODE
    $end = Get-Date

    $durationSec = [math]::Round((New-TimeSpan -Start $start -End $end).TotalSeconds, 2)
    $ok = ($exitCode -eq 0)

    if ($ok) {
        Write-Host "[Stage $Stage] Completed in $durationSec s" -ForegroundColor Green
    } else {
        Write-Host "[Stage $Stage] Failed with exit code $exitCode" -ForegroundColor Red
    }

    return [PSCustomObject]@{
        stage = $Stage
        ok = $ok
        exit_code = $exitCode
        duration_seconds = $durationSec
        started_at = $start.ToString("s")
        ended_at = $end.ToString("s")
    }
}

function Get-LastStageResult {
    param([System.Collections.ArrayList]$Results)

    if ($Results.Count -eq 0) {
        return $null
    }

    return $Results[$Results.Count - 1]
}

$runId = Get-Date -Format "yyyyMMdd_HHmmss"
$runStart = Get-Date
$stageResults = New-Object System.Collections.ArrayList
$gateChecks = New-Object System.Collections.ArrayList

if (-not $SkipStage0) {
    $null = $stageResults.Add((Invoke-Stage -Stage 0 -PythonExePath $PythonExe))
    $last = Get-LastStageResult -Results $stageResults
    if ($null -ne $last -and -not $last.ok) {
        Write-Host "Stopping due to stage 0 failure." -ForegroundColor Red
    }
}

$last = Get-LastStageResult -Results $stageResults
if ($stageResults.Count -eq 0 -or ($null -ne $last -and $last.ok)) {
    $null = $stageResults.Add((Invoke-Stage -Stage 1 -PythonExePath $PythonExe))
}
$last = Get-LastStageResult -Results $stageResults
if ($null -ne $last -and $last.ok) {
    $null = $stageResults.Add((Invoke-Stage -Stage 2 -PythonExePath $PythonExe))
}
$last = Get-LastStageResult -Results $stageResults
if ($null -ne $last -and $last.ok) {
    $null = $stageResults.Add((Invoke-Stage -Stage 3 -PythonExePath $PythonExe))
}

$requiredArtifacts = @(
    "outputs/stage0_data_quality/raw_load_report.json",
    "outputs/stage0_data_quality/metrics.json",
    "data/processed/graph_active.edgelist",
    "data/processed/node_attributes.parquet",
    "data/processed/centrality_table.parquet",
    "outputs/stage1/metrics.json",
    "outputs/stage1/params.json",
    "data/processed/community_labels.parquet",
    "outputs/stage2/metrics.json",
    "outputs/stage2/louvain_stability_report.json",
    "data/processed/sis_table.parquet",
    "data/processed/typology_labels.parquet",
    "outputs/stage3/robustness_summary.json"
)

foreach ($artifact in $requiredArtifacts) {
    $null = $gateChecks.Add((Test-RequiredFile -Path $artifact))
}

foreach ($check in (Test-Stage0RawLoadQuality -ReportPath "outputs/stage0_data_quality/raw_load_report.json")) {
    $null = $gateChecks.Add($check)
}

foreach ($check in (Test-Stage0PreprocessQuality -MetricsPath "outputs/stage0_data_quality/metrics.json")) {
    $null = $gateChecks.Add($check)
}

$failedStages = @($stageResults | Where-Object { -not $_.ok })
$failedChecks = @($gateChecks | Where-Object { -not $_.ok })

$allStagesOk = ($stageResults.Count -gt 0) -and ($failedStages.Count -eq 0)
$allArtifactsOk = ($failedChecks.Count -eq 0)
$gatePassed = $allStagesOk -and $allArtifactsOk

$runEnd = Get-Date
$totalSec = [math]::Round((New-TimeSpan -Start $runStart -End $runEnd).TotalSeconds, 2)

$report = [PSCustomObject]@{
    run_id = $runId
    started_at = $runStart.ToString("s")
    ended_at = $runEnd.ToString("s")
    total_duration_seconds = $totalSec
    stage_results = $stageResults
    quality_gate = [PSCustomObject]@{
        stages_ok = $allStagesOk
        artifacts_ok = $allArtifactsOk
        passed = $gatePassed
        checks = $gateChecks
    }
}

if (-not (Test-Path "logs/run_history")) {
    New-Item -Path "logs/run_history" -ItemType Directory -Force | Out-Null
}

$reportPath = "logs/run_history/stageC_gate_$runId.json"
$report | ConvertTo-Json -Depth 8 | Out-File -FilePath $reportPath -Encoding utf8

Write-Host "" 
Write-Host "Stage C quality gate report: $reportPath" -ForegroundColor Yellow
if ($gatePassed) {
    Write-Host "QUALITY GATE: PASSED" -ForegroundColor Green
    exit 0
}

Write-Host "QUALITY GATE: FAILED" -ForegroundColor Red
exit 1
