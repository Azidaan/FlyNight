param([string]$Run='E:\Dissertation\runs\z13_competing_candidates_20261024')
$Host.UI.RawUI.WindowTitle='FlyVis — совместный выбор кандидата и отказа'
while ($true) {
    Clear-Host
    Write-Host 'CPU: 18 новых моделей; прежние 18 контролей проверяются отдельно'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ("Состояние: {0}; этап: {1}; вариант: {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Модели: {0}/18; прошло: {1} с; обновлено: {2}" -f $s.models_done,$s.elapsed_seconds,$s.updated)
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) — текущая модель" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Сохранение: {0}; конец дерева: {1}" -f $c.save_at,$c.hard_stop)}
    Write-Host 'Общий остаток времени неизвестен. Каталог результата:' $Run
    if (Test-Path -LiteralPath "$Run\controller_result.json") {$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Завершено: код {0}; своё дерево окончено: {1}" -f $r.exit_code,$r.all_owned_ended);break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
