param([Parameter(Mandatory=$true)][string]$Run)
$protocol=Get-Content -LiteralPath (Join-Path $Run 'protocol.json') -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String
$experimentName=switch ($protocol.id) {
    'z13_candidate_reserve_v1' {'Запас среди трёх пиков'}
    'z13_candidate_identity_v1' {'Сопоставление кандидатов'}
    'z13_candidate_ranker_v1' {'Обучаемый выбор пиков'}
    'z13_query_memory_v1' {'Чтение моделей памяти'}
    'z13_local_peak_v1' {'Локальная вершина карты'}
    'z13_candidate_data104_v1' {'Данные считывателя: 20 → 104'}
    'z13_map_only104_v1' {'Выбор по карте: фиксированные104'}
    'z13_map_transfer104_v1' {'Перенос выбора по карте на35видео'}
    default {$protocol.id}
}
$Host.UI.RawUI.WindowTitle = 'FlyVis — '+$experimentName
@{pid=$PID;started=[DateTimeOffset]::Now.ToString('o');run=$Run} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'progress_console.json') -Encoding utf8
while ($true) {
    try {
        $s=Get-Content -LiteralPath (Join-Path $Run 'status.json') -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String
        Clear-Host
        Write-Host ('FlyVis: '+$experimentName) -ForegroundColor Cyan
        Write-Host ('Состояние: '+$s.state+' | фаза: '+$s.phase)
        Write-Host ('Этап: '+$s.stage+' | вариант: '+$s.variant)
        if ($s.total -gt 0) {
            $pct=[Math]::Min(100,100*$s.done/$s.total)
            Write-Progress -Activity 'Текущий этап' -Status $s.stage -PercentComplete $pct
            Write-Host ('Этап: '+$s.done+'/'+$s.total+' ('+$pct.ToString('F1')+'%)')
        } else {Write-Progress -Activity 'Текущий этап' -Completed;Write-Host 'Объём этапа неизвестен.'}
        if ($null -ne $s.arms_total) {Write-Host ('Готовых вариантов: '+$s.arms_done+'/'+$s.arms_total)}
        if (Test-Path -LiteralPath (Join-Path $Run 'controller.json')) {
            $c=Get-Content -LiteralPath (Join-Path $Run 'controller.json') -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String
            Write-Host ('Прошло: '+([DateTimeOffset]::Now-[DateTimeOffset]::Parse($c.started_at)).ToString('hh\:mm\:ss'))
            Write-Host ('Сохранение этапа: '+$c.save_stop_at+' | конец дерева: '+$c.hard_stop_at)
        }
        Write-Host ('Обновление: '+$s.updated)
        Write-Host 'Осталось: оценка пока неизвестна. Утром сохранение 07:35, остановка до 07:40 МСК.'
        Write-Host 'Сохранение: каждый пакет и готовый вариант. Закрытие окна не останавливает расчёт.'
        if (Test-Path -LiteralPath (Join-Path $Run 'night_finished.json')) {break}
    } catch {Write-Host ('Ожидание: '+$_.Exception.Message)}
    Start-Sleep -Seconds 2
}
Read-Host 'Ночная работа сохранена. Enter — закрыть'
