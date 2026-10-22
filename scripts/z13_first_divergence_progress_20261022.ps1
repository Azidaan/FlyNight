$Host.UI.RawUI.WindowTitle = 'FlyVis — разбор первого расхождения — 22.10.2026'
$runDirectory = 'E:\Dissertation\runs\z13_first_divergence_20261022'
while ($true) {
    try {
        $traceStatus = Get-Content -LiteralPath "$runDirectory\status.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'Разбор кадра 183: воспроизведение стенда и R1' -ForegroundColor Cyan
        Write-Host ('Этап: ' + $traceStatus.stage)
        Write-Host ('Кадр: ' + $traceStatus.done + ' / ' + $traceStatus.total)
        Write-Host ('Обновлено: ' + $traceStatus.updated)
        Write-Host 'Бюджет: до 360 секунд. Сохранение до 07:35 МСК. Новая HOTA не вычисляется.'
        if ($traceStatus.state -in @('completed','failed','paused')) { break }
    } catch { Write-Host 'Ожидание записи состояния' }
    Start-Sleep -Seconds 2
}
Write-Host 'Окно можно закрыть; результат сохранён отдельно.'
