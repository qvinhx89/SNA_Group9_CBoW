Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[Stage 3] Running SIS, typology, and robustness pipeline..." -ForegroundColor Cyan
python run_all.py --stage 3

if ($LASTEXITCODE -ne 0) {
	Write-Host "[Stage 3] FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
	exit $LASTEXITCODE
}

Write-Host "[Stage 3] DONE" -ForegroundColor Green
