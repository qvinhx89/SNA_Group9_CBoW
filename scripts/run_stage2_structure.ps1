Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[Stage 2] Running community and k-shell pipeline..." -ForegroundColor Cyan
python run_all.py --stage 2

if ($LASTEXITCODE -ne 0) {
	Write-Host "[Stage 2] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
	exit $LASTEXITCODE
}

Write-Host "[Stage 2] DONE" -ForegroundColor Green
