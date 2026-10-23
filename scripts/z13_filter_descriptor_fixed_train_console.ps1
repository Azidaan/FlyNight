param([string]$Run='E:\Dissertation\runs\z13_filter_descriptor_fixed_20261024\train')
$Host.UI.RawUI.WindowTitle='FlyVis — коэффициенты против среднего, 36 CPU моделей'
while ($true) {
    Clear-Host
    Write-Host '36 равных CPU моделей: D/C/R1,коэффициенты/среднее,активность/геометрия,3сида'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ("{0}: {1}; вариант {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Обновлено {0}; прошло {1}секунд; моделей готово {2}/36" -f $s.updated,$s.elapsed_seconds,$s.models_done)
    if ($s.models_done -ge 0) {Write-Progress -Id 1 -Activity 'Готовые модели' -Status "$($s.models_done)/36" -PercentComplete ([math]::Min(100,100.0*$s.models_done/36))}
    if ($s.total -gt 0) {Write-Progress -Id 2 -Activity $s.stage -Status "$($s.done)/$($s.total) — текущая модель или этап" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    Write-Host 'CPUдо1800секунд. Оставшееся время пока неизвестно; положительный итог не обещан.'
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Сохранение: {0}; конец дерева: {1}" -f $c.save_at,$c.hard_stop)}
    if (Test-Path -LiteralPath "$Run\controller_result.json") {Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8;break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
