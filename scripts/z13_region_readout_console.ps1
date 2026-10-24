$run='E:\Dissertation\runs\z13_region_readout_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis — считывание отклика по рамкам, CPU'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: карта отклика → оценки реальных рамок' -ForegroundColor Cyan
        Write-Host $s.stage
        Write-Host "Этап: $($s.variant); готовность: $($s.done)/$($s.total)"
        Write-Host "Обновлено: $($s.updated)"
        Write-Host "Сохранение: $($p.save_at); остановка: $($p.hard_stop)"
        Write-Host 'CPU, без обучения и HOTA. Время до завершения пока неизвестно.'
        if ($s.state -eq 'completed' -or $s.state -eq 'failed') {Write-Host "Окончено: $($s.state)";break}
        if ($s.total) {Write-Progress -Activity 'Текущая ступень' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
    } catch {Write-Host "Ожидание: $($_.Exception.Message)"}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Текущая ступень' -Completed
Read-Host 'Enter — закрыть окно'
