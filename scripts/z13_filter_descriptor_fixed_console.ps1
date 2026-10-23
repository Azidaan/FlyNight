param([string]$Run='E:\Dissertation\runs\z13_filter_descriptor_fixed_20261024\cache')
$Host.UI.RawUI.WindowTitle='FlyVis — коэффициенты цели, неизменные кандидаты, CPU'
while ($true) {
    Clear-Host
    Write-Host 'Коэффициенты различающего фильтра; исходные кандидаты сохранены'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ("{0}: {1}; вариант {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Обновлено {0}; прошло {1} секунд" -f $s.updated,$s.elapsed_seconds)
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) групп" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    Write-Host '587готовыхгрупп повторно не вычисляются. CPUдо18000с; затем отдельные36CPUмоделей.'
    Write-Host 'Общий остаток неизвестен; бюджет этапа и крайний срок указаны ниже.'
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Сохранение: {0}; конец дерева: {1}" -f $c.save_at,$c.hard_stop)}
    if (Test-Path -LiteralPath "$Run\controller_result.json") {Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8;break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
