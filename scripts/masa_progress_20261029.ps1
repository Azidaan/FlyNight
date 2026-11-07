param([switch]$Once)
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'MASA-Detic — полный трекер и независимая проверка'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$researchRoot = 'E:\Dissertation\runs\continuation_20261029'
$gpuFolder = Join-Path $researchRoot 'masa_query_fp32_recovery1'
$auditFolder = Join-Path $researchRoot 'masa_audit_fp32_recovery1'
$recordFolder = Join-Path $researchRoot 'masa_record_fp32_recovery1'
function Read-MasaJson([string]$path) {
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
        try {
            if (Test-Path -LiteralPath $path) {
                return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
            }
            return $null
        } catch { Start-Sleep -Milliseconds 100 }
    }
    return $null
}
do {
    $finalFolder = Join-Path $researchRoot 'masa_final_checked_record_recovery1'
    if (Test-Path -LiteralPath $finalFolder) { $recordFolder = $finalFolder }
    $queue = Read-MasaJson (Join-Path $researchRoot 'queue_status.json')
    $gpuController = Read-MasaJson (Join-Path $gpuFolder 'controller_result.json')
    $audit = Read-MasaJson (Join-Path $auditFolder 'report.json')
    $record = Read-MasaJson (Join-Path $recordFolder 'report.json')
    $stageFolder = $gpuFolder
    $stageName = 'Полный трекер на GPU — точность FP32'
    if ($queue.job -like '*050*' -or ($gpuController.exit_code -eq 0 -and $gpuController -and -not $audit)) {
        $stageFolder = $auditFolder
        $stageName = 'Независимая проверка на CPU'
    }
    if ($queue.job -like '*055*' -or ($audit.passed -and $audit.complete)) {
        $stageFolder = $recordFolder
        $stageName = 'Запись результата и обязательные проверки'
    }
    if ($queue.job -like '*036*') { $stageName = 'Запись причины численного отказа предыдущей версии' }
    $status = Read-MasaJson (Join-Path $stageFolder 'status.json')
    $controller = Read-MasaJson (Join-Path $stageFolder 'controller_result.json')
    $window = Read-MasaJson (Join-Path $stageFolder 'execution_deadline.json')
    $authorization = Read-MasaJson (Join-Path $researchRoot 'authorization.json')
    $now = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(3))
    $finished = @(Get-ChildItem -Path (Join-Path $gpuFolder 'videos\*\report.json') -File -ErrorAction SilentlyContinue)
    $audited = @(Get-ChildItem -LiteralPath (Join-Path $auditFolder 'completed') -Filter '*.json' -File -ErrorAction SilentlyContinue)
    Clear-Host
    Write-Host 'MASA-Detic — сравнение признаков в полном трекере' -ForegroundColor Cyan
    Write-Host ('Время: {0:dd.MM.yyyy HH:mm:ss} МСК' -f $now)
    Write-Host 'Окно только читает файлы. Закрытие окна не останавливает расчёт.' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host ('Этап: {0}' -f $stageName)
    Write-Host ('Очередь: {0} / {1}' -f $queue.state, $queue.job)
    Write-Host 'Варианты: исходный; нулевой; MASA-R50 1024; MASA-R50 1333; MASA-Detic 1333.'
    Write-Host ('Полных видеопроходов GPU с метриками: {0}/20 ({1:F0}%)' -f $finished.Count, (100.0*$finished.Count/20))
    Write-Host ('Полностью повторено видео на CPU: {0}/20' -f $audited.Count)
    Write-Host 'После повторения видео отдельно проверяются признаки и итоговые сравнения.' -ForegroundColor DarkGray
    if ($status.stage) { Write-Host ('Сейчас: {0}' -f $status.stage) }
    if ($status.video) { Write-Host ('Видео: {0}' -f $status.video) }
    if ($null -ne $status.frame) {
        if ($status.total_frames -gt 0) {
            Write-Host ('Текущее видео: {0}/{1} кадров ({2:F1}%)' -f $status.frame, $status.total_frames, (100.0*$status.frame/$status.total_frames))
        } else { Write-Host ('Кадр: {0}' -f $status.frame) }
    }
    if ($status.variant) { Write-Host ('Вариант: {0}' -f $status.variant) }
    if ($status.updated) {
        $updated = [DateTimeOffset]::Parse($status.updated)
        Write-Host ('Последняя запись прогресса: {0:HH:mm:ss} МСК, {1:F0} с назад' -f $updated.ToOffset([TimeSpan]::FromHours(3)), ($now-$updated).TotalSeconds)
    }
    if ($window) {
        $start = [DateTimeOffset]::Parse($window.started)
        Write-Host ('Прошло текущего этапа: {0:F1} мин' -f ($now-$start).TotalMinutes)
    }
    if ($stageFolder -eq $gpuFolder -and $finished.Count -gt 0 -and $finished.Count -lt 20) {
        $seconds = @($finished | ForEach-Object { (Read-MasaJson $_.FullName).seconds } | Where-Object { $null -ne $_ })
        if ($seconds.Count -gt 0) {
            $average = ($seconds | Measure-Object -Average).Average
            $remaining = [Math]::Max(0, (20-$finished.Count)*$average)
            Write-Host ('Оценка оставшейся GPU-серии: около {0:F1} мин, затем аудит' -f ($remaining/60))
        }
    }
    if ($authorization) {
        $save = [DateTimeOffset]::Parse($authorization.save_at).ToOffset([TimeSpan]::FromHours(3))
        $stop = [DateTimeOffset]::Parse($authorization.hard_stop).ToOffset([TimeSpan]::FromHours(3))
        Write-Host ('Сохранение: {0:dd.MM HH:mm} МСК; окончание окна: {1:dd.MM HH:mm} МСК' -f $save, $stop)
    }
    if ($controller) {
        Write-Host ('Контроллер этапа: код {0}; дерево завершено {1}' -f $controller.exit_code, $controller.all_owned_ended)
        if ($controller.exit_code -ne 0) {
            Write-Host 'Этап завершился с отказом; вывод о качестве не принят.' -ForegroundColor Yellow
            Get-Content -LiteralPath (Join-Path $stageFolder 'stderr.log') -Tail 5 -Encoding utf8 -ErrorAction SilentlyContinue
        }
    }
    if ($audit.complete -and $audit.passed) {
        Write-Host 'Полный независимый аудит пройден.' -ForegroundColor Cyan
        Write-Host ('Общий научный критерий выполнен: {0}' -f $audit.main_acceptance)
    } else { Write-Host 'Полного принятого научного итога пока нет.' -ForegroundColor Yellow }
    if ($record.complete -and $record.checks_passed) { Write-Host 'Результат оформлен, обязательные проверки пройдены.' -ForegroundColor Cyan }
    if ($record.complete -and $record.checks_passed) {
        $summary = Read-MasaJson (Join-Path $recordFolder 'scientific_summary.json')
        if ($summary) {
            Write-Host ('HOTA: исходный {0:F6}; MASA-R50 1333 {1:F6}; MASA-Detic 1333 {2:F6}' -f $summary.pooled.native.HOTA, $summary.pooled.r50_1333.HOTA, $summary.pooled.detic_1333.HOTA)
            Write-Host 'Преимущество MASA-Detic не подтверждено: доверительные интервалы включают ноль.' -ForegroundColor Yellow
        }
    }
    Write-Host ''
    Write-Host ('Файлы: {0}' -f $researchRoot) -ForegroundColor DarkGray
    Write-Host 'Обновление каждые 3 секунды. Ctrl+C — закончить наблюдение.' -ForegroundColor DarkGray
    if (-not $Once) { Start-Sleep -Seconds 3 }
} while (-not $Once)
