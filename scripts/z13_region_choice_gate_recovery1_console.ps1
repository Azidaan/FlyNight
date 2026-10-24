$run='E:\Dissertation\runs\z13_region_choice_gate_20261025\recovery1'
$Host.UI.RawUI.WindowTitle='FlyVis: region choice gate / CPU'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: запас прямого выбора реальной области, CPU' -ForegroundColor Cyan
        Write-Host $s.stage
        Write-Host "Видео: $($s.done)/$($s.total); обновлено: $($s.updated)"
        Write-Host "Сохранение: $($p.save_at); остановка: $($p.hard_stop)"
        Write-Host 'Оставшееся время неизвестно; обучение и GPU не используются.'
        if ($s.total) {Write-Progress -Activity 'Проверка присутствия' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','failed')) {break}
    } catch {Write-Host $_.Exception.Message}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Проверка присутствия' -Completed
Read-Host 'Enter - закрыть окно'