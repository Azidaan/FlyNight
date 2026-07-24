# Последовательное обучение трёх сидов DRB-full (Этап 5) под супервизором
# (автоперезапуск с --resume при дедлоке DataLoader или падении).
# Запуск: powershell -File scripts\run_drb_all_seeds.ps1
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$sup = Join-Path $PSScriptRoot 'train_supervisor.py'
foreach ($seed in 42, 123, 2024) {
    Write-Host "`n===== DRB-FULL SEED $seed (supervised) =====" -ForegroundColor Cyan
    & $python $sup --watch runs/drb_full -- python training/train_drb.py --seed $seed
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Сид $seed не добит супервизором ($LASTEXITCODE) — остановка." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nВсе три сида DRB обучены. Далее: python scripts\evaluate_run.py --run drb_full" -ForegroundColor Green
