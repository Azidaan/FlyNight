param([Parameter(Mandatory=$true)][string]$RunDirectory)
$Host.UI.RawUI.WindowTitle='FlyVis — односторонняя ошибка фона, CPU'
while ($true) {
    try {
        $s=Get-Content -LiteralPath "$RunDirectory\status.json" -Raw | ConvertFrom-Json
        $c=Get-Content -LiteralPath "$RunDirectory\controller.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis — односторонняя ошибка фона против балансировки (CPU)' -ForegroundColor Cyan
        Write-Host ('Этап: '+$s.stage+'; вариант: '+$s.arm)
        Write-Host ('Готовых вариантов: '+$s.arms_done+' / 3; фрагментов текущего: '+$s.done+' / '+$s.total)
        if ($s.total -gt 0) {Write-Progress -Id 1 -Activity ('Текущий вариант '+$s.arm) -Status ($s.done.ToString()+' / '+$s.total) -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($null -ne $s.elapsed_seconds) {Write-Host ('Прошло: '+[TimeSpan]::FromSeconds($s.elapsed_seconds).ToString('hh\:mm\:ss'))}
        Write-Host ('Последнее обновление: '+$s.updated)
        $t=Get-Content -LiteralPath "$RunDirectory\technical.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($null -ne $t.estimated_remaining_seconds) {Write-Host ('Оценка всей стоимости по короткой пробе: около '+[Math]::Ceiling($t.estimated_remaining_seconds/60)+' мин; это не обещанный срок завершения.')}
        Write-Host ('Сохранение: '+$c.save_at+'; конец своего дерева: '+$c.hard_stop)
        if (Test-Path -LiteralPath "$RunDirectory\record_blocked.json") {Write-Host 'Расчёт сохранён; служебной записи нужна проверка. См. record_blocked.json' -ForegroundColor Yellow;break}
        if ($s.state -in @('failed','paused')) {Write-Host 'Сохранено состояние остановки; завершённые части не повторять.' -ForegroundColor Yellow;break}
        if ($s.state -eq 'completed') {
            Write-Host 'Вычисления завершены. Проверяются источники, записи и публикация.'
            if (Test-Path -LiteralPath "$RunDirectory\publish_result.json") {Write-Host 'Результат проверен, записан и отправлен в Git.' -ForegroundColor Green;break}
        }
    } catch {Write-Host 'Ожидание полного состояния'}
    Start-Sleep -Seconds 3
}
Write-Progress -Id 1 -Activity 'Проверка считывания' -Completed
Write-Host 'Закрытие этого окна не меняет расчёт.'
