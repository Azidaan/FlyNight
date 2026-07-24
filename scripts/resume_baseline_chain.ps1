# Baseline: seed 42 resume (чекпойнт эпохи 27) -> сиды 123, 2024.
# Все запуски — под супервизором (автоперезапуск при дедлоке DataLoader).
# Запуск: powershell -File scripts\resume_baseline_chain.ps1
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$sup = Join-Path $PSScriptRoot 'train_supervisor.py'
$train = Join-Path $PSScriptRoot '..\training\train_baseline.py'

Write-Host "`n===== BASELINE SEED 42 (RESUME, supervised) =====" -ForegroundColor Cyan
& $python $sup --watch runs/baseline -- python training/train_baseline.py --seed 42 --resume
if ($LASTEXITCODE -ne 0) {
    Write-Host "Сид 42 не добит даже супервизором ($LASTEXITCODE) — остановка." -ForegroundColor Red
    exit 1
}
foreach ($seed in 123, 2024) {
    Write-Host "`n===== BASELINE SEED $seed (supervised) =====" -ForegroundColor Cyan
    & $python $sup --watch runs/baseline -- python training/train_baseline.py --seed $seed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Сид $seed не добит ($LASTEXITCODE) — остановка." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nВсе три сида обучены. Далее: python scripts\evaluate_baseline.py" -ForegroundColor Green
