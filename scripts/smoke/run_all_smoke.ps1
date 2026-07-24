# Прогон всех тестов проекта одной командой (smoke-скрипты + юниты).
# CPU-блок можно гонять в любой момент; GPU-блок НЕ запускать параллельно
# с обучением (риск дедлока DataLoader на Windows и замедление эпох).
#
# Запуск:
#   powershell -File scripts\smoke\run_all_smoke.ps1            # CPU + GPU
#   powershell -File scripts\smoke\run_all_smoke.ps1 -CpuOnly   # безопасно при идущем обучении
#   powershell -File scripts\smoke\run_all_smoke.ps1 -GpuOnly
#   powershell -File scripts\smoke\run_all_smoke.ps1 -Full      # + test_yolo_versions (~10+ мин, мини-обучения)
param(
    [switch]$CpuOnly,
    [switch]$GpuOnly,
    [switch]$Full
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$python = Join-Path $root '.venv\Scripts\python.exe'

$cpuTests = @(
    'scripts\smoke\test_bdd_annotations_parse.py',
    'scripts\smoke\test_bdd_eval_tools.py',
    'scripts\smoke\test_launcher_pipeline.py',
    'tests\test_q_aggregation.py',
    'scripts\smoke\test_dataloader_workers.py',
    'scripts\smoke\test_train_supervisor.py'
)
$gpuTests = @(
    'scripts\smoke\test_cuda.py',
    'scripts\smoke\test_fft.py',
    'tests\test_drb.py',
    'scripts\smoke\test_offline_association.py',
    'scripts\smoke\test_inference_with_q.py',
    'scripts\smoke\test_lqa_guided_tracker.py'
)
if ($Full) { $gpuTests += 'scripts\smoke\test_yolo_versions.py' }

if ($CpuOnly -and $GpuOnly) {
    Write-Host 'Флаги -CpuOnly и -GpuOnly несовместимы.' -ForegroundColor Red
    exit 2
}
$tests = @()
if (-not $GpuOnly) { $tests += $cpuTests }
if (-not $CpuOnly) {
    $tests += $gpuTests
    Write-Host 'ВНИМАНИЕ: GPU-блок нельзя гонять параллельно с обучением.' -ForegroundColor Yellow
}

$failed = @()
foreach ($t in $tests) {
    Write-Host "`n========== $t ==========" -ForegroundColor Cyan
    & $python (Join-Path $root $t)
    if ($LASTEXITCODE -ne 0) { $failed += $t }
}
Write-Host "`n========== ИТОГ: $($tests.Count) файлов =========="
if ($failed.Count -eq 0) {
    Write-Host 'Все тесты пройдены.' -ForegroundColor Green
    exit 0
} else {
    Write-Host "Провалены ($($failed.Count)): $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
