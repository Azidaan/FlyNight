$run = 'E:\Dissertation\runs\z13_activation_timing_20261025'
$Host.UI.RawUI.WindowTitle = 'FlyVis — причины момента включения, 26 видео'
while ($true) {
    try {
        $s = Get-Content -LiteralPath "$run\status.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: до потери или после — проверка мишени на CPU' -ForegroundColor Cyan
        Write-Host $s.stage
        Write-Host "Видео: $($s.done)/$($s.total); текущий: $($s.variant)"
        if ($s.frames) { Write-Host "Кадры текущего видео: $($s.frame)/$($s.frames)" }
        Write-Host "Последнее обновление: $($s.updated)"
        $p = Get-Content -LiteralPath "$run\protocol.json" -Raw | ConvertFrom-Json
        Write-Host "Сохранение: $($p.save_at); остановка: $($p.hard_stop)"
        Write-Host 'Время до конца: пока неизвестно. Прогресс кадров не равен прогрессу всего опыта.'
        if ($s.state -eq 'completed' -or $s.state -eq 'failed') {
            Write-Host "Расчёт окончен: $($s.state)" -ForegroundColor Yellow
            break
        }
        Write-Progress -Activity 'Готовые видео' -Status "$($s.done)/26" -PercentComplete (100*$s.done/26)
    } catch { Write-Host "Ожидание служебного файла: $($_.Exception.Message)" }
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Готовые видео' -Completed
Read-Host 'Enter — закрыть окно'
