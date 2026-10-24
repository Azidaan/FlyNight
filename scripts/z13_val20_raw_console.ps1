param([string]$Run='E:\Dissertation\runs\z13_val20_matched_20261024\raw')
$Host.UI.RawUI.WindowTitle='FlyVis — 20 val, четыре исходных трекера'
while ($true) {
    Clear-Host
    Write-Host 'Общие кадры и детекции; исходные трекеры без MASA/FlyVis'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ("Состояние: {0}; этап: {1}; видео: {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Задания: {0}/{1}; прошло: {2} с; обновлено: {3}" -f $s.jobs_done,$s.jobs_total,$s.elapsed_seconds,$s.updated)
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) кадров" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Сохранение: {0}; конец дерева: {1}" -f $c.save_at,$c.hard_stop)}
    Write-Host 'Остаток времени пока неизвестен. Результаты:' $Run
    if (Test-Path -LiteralPath "$Run\controller_result.json") {$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Код: {0}; дерево окончено: {1}" -f $r.exit_code,$r.all_owned_ended);break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть окно наблюдения'
