param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — память наблюдений до потери'
@{pid=$PID;started=[DateTimeOffset]::Now.ToString('o');run=$Run} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run 'progress_console.json') -Encoding utf8
while ($true) {
    try {
        $s=Get-Content -LiteralPath (Join-Path $Run 'status.json') -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String
        Clear-Host
        Write-Host 'FlyVis: память отдельных наблюдений до потери' -ForegroundColor Cyan
        Write-Host ('Фаза: '+$s.phase+' | состояние: '+$s.state)
        Write-Host ('Этап: '+$s.stage+' | вариант: '+$s.variant)
        if ($s.total -gt 0) {
            Write-Progress -Activity 'Текущий этап; не процент всего опыта' -Status $s.stage -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))
            Write-Host ('Выполнено в этапе: '+$s.done+'/'+$s.total)
        } else {Write-Host 'Объём этапа неизвестен.'}
        $cpath=Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $cpath) {
            $c=Get-Content -LiteralPath $cpath -Raw -Encoding UTF8 | ConvertFrom-Json -DateKind String
            Write-Host ('Прошло: '+([DateTimeOffset]::Now-[DateTimeOffset]::Parse($c.started_at)).ToString('hh\:mm\:ss'))
            Write-Host ('Сохранение/остановка этапа: '+$c.save_stop_at)
            Write-Host ('Полное завершение дерева: '+$c.hard_stop_at)
        }
        Write-Host ('Последнее обновление: '+$s.updated)
        Write-Host 'Осталось: пока неизвестно; бюджет этапа не является оценкой длительности.'
        Write-Host 'Сохранение: после видео восстановления и готового варианта. Закрытие окна не останавливает расчёт.'
        if ($s.state -in @('failed','paused')) {Write-Progress -Activity 'Текущий этап; не процент всего опыта' -Completed;break}
        if ($s.state -eq 'completed' -and $s.phase -eq 'full') {Write-Progress -Activity 'Текущий этап; не процент всего опыта' -Completed;break}
    } catch {Write-Host ('Ожидание состояния: '+$_.Exception.Message)}
    Start-Sleep -Seconds 1
}
Read-Host 'Итог оставлен в окне. Enter — закрыть'
