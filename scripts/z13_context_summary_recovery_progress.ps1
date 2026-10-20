param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — восстановление итоговой сводки'
while ($true) {
    try {
        $s = Get-Content -LiteralPath (Join-Path $Run 'recovery_summary_status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: сборка сводки из готовых предсказаний C и R1'
        Write-Host ('Состояние: '+$s.state+'; этап: '+$s.stage)
        Write-Host ('Выполнено этапов: '+$s.done+'/'+$s.total)
        if ($s.total -gt 0) { Write-Progress -Activity 'Сборка сохранённой сводки' -Status $s.stage -PercentComplete (100*$s.done/$s.total) }
        Write-Host ('Последнее обновление: '+$s.updated)
        $path = Join-Path $Run 'recovery_summary_controller.json'
        if (Test-Path -LiteralPath $path) {
            $c = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            Write-Host ('Прошло: '+((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
            Write-Host ('Предел сохранения: '+$c.save_stop_at+'; завершение дерева: '+$c.hard_stop_at)
        }
        Write-Host 'Осталось: неизвестно. Отдельные этапы имеют разную стоимость.'
        Write-Host 'Предсказания уже сохранены. Закрытие окна не останавливает сборку.'
        if ($s.state -in @('completed','failed')) { Write-Progress -Activity 'Сборка сохранённой сводки' -Completed; break }
    } catch { Write-Host ('Ожидание состояния: '+$_.Exception.Message) }
    Start-Sleep -Seconds 1
}
Read-Host 'Итог сохранён. Enter — закрыть окно'
