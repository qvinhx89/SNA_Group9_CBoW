Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[Stage 1] Running centrality pipeline..." -ForegroundColor Cyan
python run_all.py --stage 1

if ($LASTEXITCODE -ne 0) {
	Write-Host "[Stage 1] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
	exit $LASTEXITCODE
}

Write-Host "[Stage 1] DONE" -ForegroundColor Green
