$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'Б1 — полный трекер: ход проверки'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$projectRoot = 'E:\Dissertation'
$researchRoot = Join-Path $projectRoot 'runs\recovery_research_20261027'
$sourceFolder = Join-Path $researchRoot 'b1_tracker\recovery1'
$auditFolder = Join-Path $researchRoot 'b1_tracker_audit\recovery1'
function Read-ProgressJson([string]$path) {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            if (Test-Path -LiteralPath $path) { return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json }
            return $null
        } catch { Start-Sleep -Milliseconds 100 }
    }
    return $null
}
while ($true) {
    $activeStage = Read-ProgressJson (Join-Path $researchRoot 'b1_active_stage.json')
    if ($activeStage) {
        $sourceFolder = Join-Path $researchRoot $activeStage.source
        $auditFolder = Join-Path $researchRoot $activeStage.audit
    }
    $queueState = Read-ProgressJson (Join-Path $researchRoot 'queue_status.json')
    $stageFolder = $sourceFolder
    $stageName = 'Полный трекер на GPU'
    if ($queueState.job -like '*b1*' -and $queueState.job -like '*audit*') {
        $stageFolder = $auditFolder
        $stageName = 'Независимый повтор на CPU'
    }
    $progressState = Read-ProgressJson (Join-Path $stageFolder 'status.json')
    $controllerState = Read-ProgressJson (Join-Path $stageFolder 'controller_result.json')
    $executionWindow = Read-ProgressJson (Join-Path $stageFolder 'execution_deadline.json')
    $priceState = Read-ProgressJson (Join-Path $sourceFolder 'price.json')
    $regressionState = Read-ProgressJson (Join-Path $sourceFolder 'incremental_regression.json')
    $completedCount = @(Get-ChildItem -LiteralPath (Join-Path $sourceFolder 'completed') -Filter '*.json' -File -ErrorAction SilentlyContinue).Count
    $moscowNow = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(3))
    Clear-Host
    Write-Host 'Б1 — временная поправка к внешнему виду внутри полного трекера' -ForegroundColor Cyan
    Write-Host ('Время: {0:dd.MM.yyyy HH:mm:ss} МСК' -f $moscowNow)
    Write-Host 'Это окно только читает файлы. Его закрытие не останавливает расчёт.' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host ('Этап: {0}' -f $stageName)
    Write-Host ('Очередь: {0} / {1}' -f $queueState.state, $queueState.job)
    Write-Host ('Состояние: {0}' -f $progressState.state)
    Write-Host ('Сейчас: {0}' -f $progressState.stage)
    Write-Host ('Полных видео: {0}/20' -f $completedCount)
    if ($null -ne $progressState.video) { Write-Host ('Видео: {0}' -f $progressState.video) }
    if ($null -ne $progressState.frame -and $progressState.total_frames -gt 0) {
        $framePercent = 100.0 * [double]$progressState.frame / [double]$progressState.total_frames
        Write-Host ('Текущее видео: {0}/{1} кадров ({2:F1}%)' -f $progressState.frame, $progressState.total_frames, $framePercent)
    }
    if ($null -ne $progressState.variant) { Write-Host ('Вариант: {0}' -f $progressState.variant) }
    if ($null -ne $progressState.queries) {
        Write-Host ('Запросы вариантов: {0}' -f ($progressState.queries | ConvertTo-Json -Compress))
    }
    if ($null -ne $progressState.updated) {
        $lastUpdate = [DateTimeOffset]::Parse($progressState.updated)
        Write-Host ('Последний прогресс: {0:HH:mm:ss} МСК, {1:F0} с назад' -f $lastUpdate.ToOffset([TimeSpan]::FromHours(3)), ($moscowNow-$lastUpdate).TotalSeconds)
    }
    if ($executionWindow) {
        $startedAt = [DateTimeOffset]::Parse($executionWindow.started)
        $endAt = [DateTimeOffset]::Parse($executionWindow.deadline)
        Write-Host ('Прошло этапа: {0:F1} мин; осталось бюджета: {1:F1} мин' -f ($moscowNow-$startedAt).TotalMinutes, [Math]::Max(0, ($endAt-$moscowNow).TotalMinutes))
        Write-Host ('Остановка текущего этапа: {0:dd.MM HH:mm:ss} МСК' -f $endAt.ToOffset([TimeSpan]::FromHours(3)))
    }
    Write-Host 'Общий срок: сохранить 07:30; остановить деревья 07:39:30; завершить работу 07:40 МСК.'
    Write-Host ''
    Write-Host ('Проверка последовательного состояния: {0}' -f $(if ($regressionState.passed) {'пройдена точно для C/D/R1/B'} else {'ещё не завершена'}))
    if ($priceState.author_override) {
        Write-Host 'Продолжение оставшихся 19 видео прямо разрешено автором после прежнего ценового стопа.' -ForegroundColor Yellow
        Write-Host 'Исходная оценка стоимости остаётся отрицательной; общий срок остановки сохранён.'
    } elseif ($priceState) {
        Write-Host ('Первое полное видео: {0:F1} с; консервативная оставшаяся цена: {1:F1} мин; допуск серии: {2}' -f $priceState.first_full_video_seconds, ($priceState.conservative_remaining_seconds/60), $priceState.proceed)
    } else {
        Write-Host 'Стоимость полной серии: пока неизвестна. Остальные видео допускаются после первого полного прохода.'
    }
    if ($controllerState) {
        Write-Host ('Контроллер: код {0}; превышение срока {1}; дерево завершено {2}' -f $controllerState.exit_code, $controllerState.timed_out, $controllerState.all_owned_ended)
        if ($controllerState.exit_code -ne 0) {
            Write-Host 'Сохранён технический отказ. Конец журнала ошибки:' -ForegroundColor Red
            Get-Content -LiteralPath (Join-Path $stageFolder 'stderr.log') -Tail 12 -Encoding utf8 -ErrorAction SilentlyContinue
        }
    }
    $finalAudit = Read-ProgressJson (Join-Path $sourceFolder 'independent_audit.json')
    if ($finalAudit) { Write-Host ('Итоговый независимый аудит: {0}; полная серия: {1}' -f $finalAudit.passed, $finalAudit.full_series) -ForegroundColor Green }
    Write-Host ''
    Write-Host ('Файлы: {0}' -f $sourceFolder) -ForegroundColor DarkGray
    Write-Host 'Обновление каждые 3 секунды; Ctrl+C — закрыть наблюдение.' -ForegroundColor DarkGray
    Start-Sleep -Seconds 3
}
