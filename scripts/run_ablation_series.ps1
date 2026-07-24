# Ablation-обучения Этапа 7 под супервизором: Серия I (LQA-only, FGA-only
# x 2 сида) и Серия II (DRB-at-input x 1 сид). 5 обучений по 50 эпох.
# Запуск: powershell -File scripts\run_ablation_series.ps1
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
$sup = Join-Path $PSScriptRoot 'train_supervisor.py'

foreach ($variant in 'lqa', 'fga') {
    foreach ($seed in 42, 123) {
        Write-Host "`n===== ABLATION $variant SEED $seed (supervised) =====" -ForegroundColor Cyan
        & $python $sup --watch "runs/ablation_${variant}_only" -- python training/train_drb.py --seed $seed --variant $variant --config "training/configs/${variant}_only.yaml"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Обучение $variant/$seed не добито ($LASTEXITCODE) — остановка." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "`n===== ABLATION drb_at_input SEED 42 (supervised) =====" -ForegroundColor Cyan
& $python $sup --watch runs/ablation_drb_input -- python training/train_drb.py --seed 42 --variant input --config training/configs/drb_at_input.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "`nAblation-обучения завершены. Далее:" -ForegroundColor Green
Write-Host "  python scripts\evaluate_run.py --run ablation_lqa_only --seeds 42 123"
Write-Host "  python scripts\evaluate_run.py --run ablation_fga_only --seeds 42 123"
Write-Host "  python scripts\evaluate_run.py --run ablation_drb_input --seeds 42"
Write-Host "  python scripts\build_ablation_tables.py"
