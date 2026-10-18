param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — перенос train26 → dev35'
function Read-SharedJson([string]$Path) {
    $stream=[System.IO.File]::Open($Path,'Open','Read',([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader=[System.IO.StreamReader]::new($stream,[System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
}
while ($true) {
    try {
        $s=Read-SharedJson (Join-Path $Run 'status.json')
        $p=Read-SharedJson (Join-Path $Run 'protocol.json')
        Clear-Host
        Write-Host 'FlyVis: перенос готового отбора с train26 на dev35'
        Write-Host ('Состояние: '+$s.state+'; вариант: '+$s.variant)
        Write-Host ('Этап: '+$s.stage)
        Write-Host ('Готово вариантов: '+$s.arms_done+'/3; видео текущего варианта: '+$s.videos_done+'/35')
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Текущий этап: {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity 'FlyVis dev35' -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Объём текущего этапа неизвестен' }
        Write-Host ('Последнее обновление: '+$s.updated)
        $cpath=Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $cpath) {
            $c=Read-SharedJson $cpath
            Write-Host ('Прошло: '+((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
        }
        Write-Host 'Осталось: пока неизвестно; процент этапа не равен проценту всего опыта.'
        Write-Host ('Сохранение и остановка: '+$p.save_stop_at+'; конец дерева: '+$p.hard_stop_at)
        Write-Host 'Каждое готовое видео сохранено. Закрытие окна не останавливает расчёт.'
        if ($s.state -in @('completed','failed','paused')) { Write-Progress -Activity 'FlyVis dev35' -Completed; break }
    } catch { Write-Host ('Ожидание файла состояния: '+$_.Exception.Message) }
    Start-Sleep -Seconds 3
}
Read-Host 'Итог сохранён. Enter — закрыть окно'
