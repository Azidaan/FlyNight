param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis — пространственное считывание C, перенос'
function Read-SharedJson([string]$Path) {
    $stream=[System.IO.File]::Open($Path,'Open','Read',([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader=[System.IO.StreamReader]::new($stream,[System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
}
while ($true) {
    try {
        $s=Read-SharedJson (Join-Path $Run 'status.json')
        Clear-Host
        Write-Host 'FlyVis: проверка переноса пространственного считывания C'
        Write-Host ('Состояние: '+$s.state+'; вариант: '+$s.variant)
        Write-Host ('Этап: '+$s.stage)
        if ($s.arms_total) { Write-Host ('Готово вариантов: '+$s.arms_done+'/'+$s.arms_total) }
        if ($s.videos_total) { Write-Host ('Готово видео текущего варианта: '+$s.videos_done+'/'+$s.videos_total) }
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Текущий этап: {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity 'FlyVis CPU' -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Объём текущего этапа неизвестен' }
        Write-Host ('Последнее обновление: '+$s.updated)
        $controller=Join-Path $Run 'controller.json'
        if (!(Test-Path -LiteralPath $controller)) { $controller=Join-Path $Run 'technical_controller.json' }
        if (Test-Path -LiteralPath $controller) {
            $c=Read-SharedJson $controller
            Write-Host ('Прошло: '+((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
            Write-Host ('Сохранение: '+$c.save_stop_at+'; конец дерева: '+$c.hard_stop_at)
        }
        Write-Host 'Осталось: пока неизвестно. Процент этапа не равен проценту всего опыта.'
        Write-Host 'Готовые видео сохраняются. Закрытие окна не останавливает расчёт.'
        $ended=(Test-Path -LiteralPath (Join-Path $Run 'technical_controller_result.json')) -or (Test-Path -LiteralPath (Join-Path $Run 'controller_result.json'))
        if ($s.state -in @('completed','failed','paused') -or ($ended -and $s.state -eq 'ready')) { Write-Progress -Activity 'FlyVis CPU' -Completed; break }
    } catch { Write-Host ('Ожидание файла состояния: '+$_.Exception.Message) }
    Start-Sleep -Seconds 3
}
Read-Host 'Итог сохранён. Enter — закрыть окно'
