param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — парное обучение оценки полезности'
while ($true) {
    try {
        $s = Get-Content -LiteralPath (Join-Path $Run 'status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: парная цель оценки; сохранённые признаки, только процессор'
        Write-Host ('Состояние: '+$s.state+'; этап: '+$s.stage+'; вариант: '+$s.variant)
        if ($s.total -gt 0) {
            Write-Progress -Activity 'Текущий этап, не процент всего опыта' -Status ($s.stage+' '+$s.variant) -PercentComplete (100*$s.done/$s.total)
            Write-Host ('Шагов текущего этапа: '+$s.done+'/'+$s.total)
        }
        $done = @(Get-ChildItem -LiteralPath $Run -Directory | Where-Object {Test-Path -LiteralPath (Join-Path $_.FullName 'done.json')}).Count
        Write-Host ('Готово вариантов D/C/R1: '+$done+'/3')
        $cp = Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $cp) {
            $c = Get-Content -LiteralPath $cp -Raw -Encoding UTF8 | ConvertFrom-Json
            $elapsed = [datetimeoffset]::Now - [datetimeoffset]::Parse($c.started_at)
            Write-Host ('Прошло: '+$elapsed.ToString('hh\:mm\:ss'))
            Write-Host ('Сохранение до: '+$c.save_stop_at+'; завершение дерева до: '+$c.hard_stop_at)
            Write-Host ('Расписание автора: '+$c.schedule_hard_stop)
        }
        Write-Host ('Последнее обновление: '+$s.updated)
        Write-Host 'Осталось: неизвестно. Проверка ограничена 10 минутами, это не оценка длительности.'
        Write-Host 'Сохранение после каждого обучения; закрытие этого окна не останавливает расчёт.'
        if ($s.state -in @('completed','failed','paused')) {Write-Progress -Activity 'Текущий этап, не процент всего опыта' -Completed;break}
    } catch {Write-Host ('Ожидание состояния: '+$_.Exception.Message)}
    Start-Sleep -Seconds 1
}
Read-Host 'Итог оставлен в окне. Enter — закрыть'
