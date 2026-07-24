# Последовательное обучение трёх сидов baseline (Этап 3).
# Запуск: powershell -File scripts\run_baseline_all_seeds.ps1
# Ожидаемое время: ~10-14 ч на сид, суммарно ~1.5-2 суток GPU.
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$train = Join-Path $PSScriptRoot '..\training\train_baseline.py'
foreach ($seed in 42, 123, 2024) {
    Write-Host "`n===== BASELINE SEED $seed =====" -ForegroundColor Cyan
    & $python $train --seed $seed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Сид $seed завершился с ошибкой ($LASTEXITCODE) — остановка." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nВсе три сида обучены. Далее: python scripts\evaluate_baseline.py" -ForegroundColor Green
