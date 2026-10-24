$run='E:\Dissertation\runs\z13_region_readout_20261025\recovery1'
$Host.UI.RawUI.WindowTitle='FlyVis — региональное считывание, восстановление1'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: региональный выход, служебное восстановление1' -ForegroundColor Cyan
        Write-Host $s.stage
        Write-Host "Текущий вариант: $($s.variant); ступень: $($s.done)/$($s.total)"
        Write-Host "Обновление: $($s.updated); сохранение: $($p.save_at); конец: $($p.hard_stop)"
        Write-Host 'Исходный бюджет не продлён. CPU, без обучения и HOTA. Остаток времени расчёта пока неизвестен.'
        if ($s.state -eq 'completed' -or $s.state -eq 'failed') {Write-Host "Окончено: $($s.state)";break}
        if ($s.total) {Write-Progress -Activity 'Текущая ступень' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
    } catch {Write-Host "Ожидание служебного файла: $($_.Exception.Message)"}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Текущая ступень' -Completed
Read-Host 'Enter — закрыть окно'
