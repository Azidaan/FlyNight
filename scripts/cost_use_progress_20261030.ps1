param([switch]$Once)
$ErrorActionPreference = 'Stop'
$taskRoot = 'E:\Dissertation\runs\continuation_20261029'
function Read-ProgressJson([string]$TaskPath) {
    if (-not (Test-Path -LiteralPath $TaskPath)) { return $null }
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        try { return Get-Content -LiteralPath $TaskPath -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { if ($attempt -eq 2) { return $null }; Start-Sleep -Milliseconds 100 }
    }
}
$taskStages = @(
    @('cost_use_preparation','Подготовка и проверки каталога'),
    @('cost_use_dev','Полная проба на учебном видео'),
    @('cost_use_dev_audit','Отдельное повторение пробы'),
    @('cost_use_val_preparation_recovery1','Подготовка двадцати видео'),
    @('cost_use_val','Три новых правила на двадцати видео'),
    @('cost_use_val_audit','Полное отдельное повторение'),
    @('cost_use_val_record','Запись результата и проверка графа')
)
do {
    if (-not $Once) { Clear-Host }
    Write-Host 'Применение признаков: центрирование и общий сдвиг без внешности'
    Write-Host 'Только наблюдение. Закрытие окна не останавливает расчёт.'
    Write-Host ''
    $taskDone = 0
    foreach ($taskStage in $taskStages) {
        $taskFolder = Join-Path $taskRoot $taskStage[0]
        $taskController = Read-ProgressJson (Join-Path $taskFolder 'controller_result.json')
        $taskStatus = Read-ProgressJson (Join-Path $taskFolder 'status.json')
        if ($null -ne $taskController) {
            if ($taskController.exit_code -eq 0 -and $taskController.all_owned_ended) {
                $taskDone++
                Write-Host ($taskStage[1] + ': готово') -ForegroundColor Green
            } else { Write-Host ($taskStage[1] + ': отказ, код ' + $taskController.exit_code) -ForegroundColor Red }
        } elseif ($null -ne $taskStatus) {
            $taskLabel = if ($taskStatus.stage) { $taskStatus.stage } else { $taskStatus.state }
            Write-Host ($taskStage[1] + ': ' + $taskLabel) -ForegroundColor Yellow
            $taskLaunch = Read-ProgressJson (Join-Path $taskFolder 'launch.json')
            if ($null -ne $taskLaunch) {
                $taskElapsed = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [long]$taskLaunch.created
                Write-Host ('  Прошло с запуска этапа: ' + [math]::Round($taskElapsed / 60, 1) + ' мин.')
            }
            if ($null -ne $taskStatus.index) {
                Write-Host ('  Полных видео до текущего: ' + $taskStatus.index + '/20; текущее: ' + $taskStatus.video)
            }
            if ($null -ne $taskStatus.frame -and $null -ne $taskStatus.total_frames) {
                $taskPercent = [math]::Round(100 * $taskStatus.frame / $taskStatus.total_frames, 1)
                Write-Host ('  Кадр ' + $taskStatus.frame + '/' + $taskStatus.total_frames + ' (' + $taskPercent + '% текущего видео)')
            }
            if ($taskStatus.variant) { Write-Host ('  Вариант: ' + $taskStatus.variant) }
            Write-Host ('  Обновление источника: ' + $taskStatus.updated)
        } else { Write-Host ($taskStage[1] + ': ожидает') }
    }
    Write-Host ''
    Write-Host ('Готовых этапов: ' + $taskDone + '/' + $taskStages.Count + '. Это не процент научной приёмки.')
    $taskPrice = Read-ProgressJson (Join-Path $taskRoot 'cost_use_val\price.json')
    if ($null -ne $taskPrice) {
        Write-Host ('Первое полное видео: ' + [math]::Round($taskPrice.first_complete_video_seconds, 2) + ' с.')
        Write-Host 'Прогноз полной цепочки не обновляется этим наблюдателем; готовность определяется отдельным аудитом.'
    }
    $taskReport = Read-ProgressJson (Join-Path $taskRoot 'cost_use_agent_review\verified.json')
    if ($null -ne $taskReport) {
        Write-Host ('Полный итог проверен агентом. Главный критерий: ' + $taskReport.main_acceptance)
    }
    Write-Host 'Сохранение 30.10 07:30 МСК; остановка всех своих процессов 07:39:30; конец 07:40.'
    if (-not $Once) { Start-Sleep -Seconds 5 }
} while (-not $Once)
