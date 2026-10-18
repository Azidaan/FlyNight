param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — направленный контекст train26'
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
        Write-Host 'FlyVis: направленный контекст считывания; ядра заморожены'
        Write-Host ('Состояние: '+$s.state+'; вариант: '+$s.variant+'; этап: '+$s.stage)
        Write-Host ('Готово ядер: '+$s.arms_done+'/3; видео текущего ядра: '+$s.videos_done+'/26')
        Write-Host 'На одном проходе ядра сравниваются среднее, направления и перемешанный контроль.'
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Текущий этап: {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity 'FlyVis train26' -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Объём текущего этапа неизвестен' }
        Write-Host ('Последнее обновление: '+$s.updated)
        $cpath=Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $cpath) {
            $c=Read-SharedJson $cpath
            Write-Host ('Прошло: '+((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
        }
        Write-Host 'Осталось: неизвестно; процент этапа не равен проценту всего опыта.'
        Write-Host ('Сохранение: '+$p.save_stop_at+'; конец дерева: '+$p.hard_stop_at)
        Write-Host 'Готовые видео сохранены. Закрытие окна не останавливает расчёт.'
        if ($s.state -in @('completed','failed','paused')) { Write-Progress -Activity 'FlyVis train26' -Completed; break }
    } catch { Write-Host ('Ожидание состояния: '+$_.Exception.Message) }
    Start-Sleep -Seconds 3
}
Read-Host 'Итог сохранён. Enter — закрыть окно'
