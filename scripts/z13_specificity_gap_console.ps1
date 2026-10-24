$run='E:\Dissertation\runs\z13_specificity_gap_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis — контраст уверенности в выбранной точке'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: проверка новой формулы уверенности, CPU' -ForegroundColor Cyan
        Write-Host $s.stage
        Write-Host "Вариант: $($s.variant); блоки: $($s.done)/$($s.total)"
        Write-Host "Обновлено: $($s.updated)"
        Write-Host "Сохранение: $($p.save_at); остановка: $($p.hard_stop)"
        Write-Host 'Оставшееся время неизвестно. Сохранённые карты; центр прежний.'
        if ($s.total) {Write-Progress -Activity 'Контраст уверенности' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','failed')) {break}
    } catch {Write-Host "Ожидание: $($_.Exception.Message)"}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Контраст уверенности' -Completed
Read-Host 'Enter — закрыть окно'
