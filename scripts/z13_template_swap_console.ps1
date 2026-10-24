$run='E:\Dissertation\runs\z13_template_swap_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis — свой образец и чужой, неизменное состояние'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis: проверка зависимости считывателя от образца' -ForegroundColor Cyan
        Write-Host 'Один и тот же поиск; свой фильтр и три чужих. D / C / R1.'
        Write-Host "Этап: $($s.stage); вариант: $($s.variant)"
        Write-Host "Группы: $($s.done)/$($s.total). Прошло: $($s.elapsed_seconds) с"
        Write-Host "Обновлено: $($s.updated)"
        Write-Host "Сохранение: $($p.save_at); конец дерева: $($p.hard_stop)"
        if (Test-Path -LiteralPath "$run\price.json") {
            $t=Get-Content -LiteralPath "$run\price.json" -Raw | ConvertFrom-Json
            Write-Host "Консервативная оценка после первых двух групп: $([Math]::Ceiling($t.conservative_remaining_seconds/60)) мин"
        } else {Write-Host 'Оставшееся время пока неизвестно — идёт замер.'}
        if ($s.total) {Write-Progress -Activity 'Три варианта, группы эпизодов' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','failed')) {Write-Host "Расчёт окончен: $($s.state)";break}
    } catch {Write-Host "Ожидание чтения: $($_.Exception.Message)"}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Три варианта, группы эпизодов' -Completed
Read-Host 'Enter — закрыть окно'
