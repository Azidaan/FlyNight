# Финальное обучение YOLO + DRB + аугментации, три сида (Этап 6),
# под супервизором. Перед запуском: python scripts\visualize_aug_batches.py
# и просмотреть figs/aug_examples/ (проверка выполнена 05.07.2026 — батчи валидны).
# Запуск: powershell -File scripts\run_drb_aug_all_seeds.ps1
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$sup = Join-Path $PSScriptRoot 'train_supervisor.py'
$config = 'training/configs/drb_aug_full.yaml'

foreach ($seed in 42, 123, 2024) {
    Write-Host "`n===== DRB-AUG-FULL SEED $seed (supervised) =====" -ForegroundColor Cyan
    & $python $sup --watch runs/drb_aug_full -- python training/train_drb.py --seed $seed --config $config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Сид $seed не добит ($LASTEXITCODE) — остановка." -ForegroundColor Red
        exit 1
    }
}
Write-Host "`nФинальные модели обучены. Далее:" -ForegroundColor Green
Write-Host "  python scripts\evaluate_run.py --run drb_aug_full"
Write-Host "  python scripts\compare_runs.py --runs baseline drb_full drb_aug_full --out drb_aug_vs_drb_vs_baseline"
