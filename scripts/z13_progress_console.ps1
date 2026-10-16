param(
    [string]$RunDirectory = 'E:\Dissertation\runs\z13_candidates_20261016',
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
function Read-State([string]$Name) {
    $file = Join-Path $RunDirectory $Name
    if (Test-Path -LiteralPath $file) {
        try { return (Get-Content -LiteralPath $file -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
    }
    return $null
}
function Progress-Bar([double]$Value) {
    $value = [Math]::Max(0.0, [Math]::Min(100.0, $Value))
    $filled = [int][Math]::Floor($value * 40 / 100)
    return '[' + ('#' * $filled) + ('-' * (40 - $filled)) + '] ' + $value.ToString('0.0') + '%'
}
$Host.UI.RawUI.WindowTitle = 'FlyVis З13 — прогресс до 07:40'
if (-not $Once) {
    @{pid=$PID; started=(Get-Date -Format o); output_redirected=[Console]::IsOutputRedirected; run=$RunDirectory} |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDirectory 'progress_console_started.json') -Encoding utf8
}
$modes = @{base='Контрольное продолжение'; foreground='Цель и окружение'; size='Размеры рамки'; fixed='Фиксированное накопление'; adaptive='Адаптивное накопление'}
do {
    $state = Read-State 'status.json'
    if (Test-Path -LiteralPath (Join-Path $RunDirectory 'recovery2_controller.json')) {
        $controller = Read-State 'recovery2_controller.json'
        $result = Read-State 'recovery2_controller_result.json'
    } else {
        $controller = Read-State 'resume_controller.json'
        $result = Read-State 'resume_controller_result.json'
    }
    $protocol = Read-State 'protocol.json'
    if (-not $Once) { Clear-Host }
    Write-Host 'FlyVis З13 | Три отдельных узла | GPU' -ForegroundColor Cyan
    Write-Host ('Время: ' + (Get-Date -Format 'dd.MM.yyyy HH:mm:ss'))
    Write-Host ''
    $done = 0
    if ($protocol) {
        foreach ($job in $protocol.jobs) {
            $marker = Join-Path $RunDirectory ($job.arm + '/' + $job.mode + '/completed.json')
            if (Test-Path -LiteralPath $marker) { $done++ }
        }
        $count = @($protocol.jobs).Count
        Write-Host ('Готовые задания: {0} / {1} (доля заданий, не оценка времени)' -f $done, $count)
        Write-Host (Progress-Bar (100 * $done / [Math]::Max(1, $count))) -ForegroundColor Green
    }
    if ($state) {
        Write-Host ''
        Write-Host ('Текущий вариант: {0} | {1}' -f $state.arm, $modes[$state.mode])
        $percent = $null
        if ($state.stage -eq 'training') {
            Write-Host ('Обучение: {0} / {1} шагов' -f $state.step,$state.total)
            $percent = 100 * $state.step / [Math]::Max(1,$state.total)
            if ($state.seconds_per_step -gt 0) {
                $seconds = [Math]::Max(0,($state.total-$state.step)*$state.seconds_per_step)
                Write-Host ('Оценка до конца обучения этого задания: {0:0.0} мин; затем предсказания' -f ($seconds/60))
            }
        } elseif ($state.stage -eq 'prediction') {
            Write-Host ('Предсказания: {0} / {1} эпизодов' -f $state.episodes_done,$state.episodes_total)
            $percent = 100 * $state.episodes_done / [Math]::Max(1,$state.episodes_total)
        } else {
            Write-Host ('Этап: ' + $state.stage + ' | точный процент этапа неизвестен')
        }
        if ($null -ne $percent) { Write-Host (Progress-Bar $percent) -ForegroundColor Yellow }
        Write-Host ('Сохранение подтверждено: ' + $state.checkpoint_verified)
        $age = ((Get-Date).ToUniversalTime() - ([datetimeoffset]$state.updated).UtcDateTime).TotalSeconds
        Write-Host ('Последнее обновление: {0} ({1:0} с назад)' -f $state.updated, $age)
        if ($age -gt 120) { Write-Host 'Давно нет обновления; это сигнал проверить журнал, а не доказательство зависания.' -ForegroundColor Yellow }
    }
    if ($controller) {
        $elapsed = ((Get-Date).ToUniversalTime()-([datetimeoffset]$controller.started).UtcDateTime).TotalMinutes
        Write-Host ('С начала GPU-очереди: {0:0.0} мин' -f $elapsed)
        Write-Host ('Сохранение к сроку: ' + $controller.save_deadline)
        Write-Host ('Предел: ' + $controller.deadline)
    }
    Write-Host 'Оставшееся время всей очереди пока не оценено.'
    Write-Host ''
    Write-Host 'Окно только наблюдает. Его закрытие НЕ остановит расчёт.' -ForegroundColor DarkGray
    if ($result) {
        Write-Host ('Процесс завершён. Код: {0}; причина: {1}' -f $result.exit_code,$result.reason) -ForegroundColor Cyan
        break
    }
    if ($Once) { break }
    Start-Sleep -Seconds 2
} while ($true)
