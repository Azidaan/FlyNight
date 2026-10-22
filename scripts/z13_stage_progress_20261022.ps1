param([Parameter(Mandatory=$true)][string]$RunDirectory)
$Host.UI.RawUI.WindowTitle = 'FlyVis — текущая проверка считывания'
while ($true) {
    try {
        $stageStatus=Get-Content -LiteralPath "$RunDirectory\status.json" -Raw | ConvertFrom-Json
        Clear-Host
        Write-Host 'FlyVis — проверка узла считывания' -ForegroundColor Cyan
        Write-Host ('Этап: '+$stageStatus.stage+'; вариант: '+$stageStatus.arm)
        Write-Host ('Готово: '+$stageStatus.done+' / '+$stageStatus.total)
        if ($stageStatus.total -gt 0) {Write-Progress -Activity $stageStatus.stage -PercentComplete ([Math]::Min(100,100*$stageStatus.done/$stageStatus.total))}
        Write-Host ('Последнее обновление: '+$stageStatus.updated)
        Write-Host 'Оставшееся время: пока неизвестно; оценка не подменяет фактический прогресс.'
        $stageControl=Get-Content -LiteralPath "$RunDirectory\controller.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
        Write-Host ('Сохранение: '+$stageControl.save_at+'; конец дерева: '+$stageControl.hard_stop)
        if ($stageStatus.state -in @('completed','failed','paused')) {break}
    } catch {Write-Host 'Ожидание состояния'}
    Start-Sleep -Seconds 2
}
Write-Host 'Итог сохранён. Закрытие окна не меняет расчёт.'
