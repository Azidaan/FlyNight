param([string]$Run='E:\Dissertation\runs\z13_matched_trackers_20261024')
$Host.UI.RawUI.WindowTitle='FlyVis — одинаковые условия сравнения'
while ($true) {
    Clear-Host
    Write-Host 'Сравнение на общих 7 val-видео; исходные кадры и детекции одинаковы'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 2;continue}
    Write-Host ("Этап: {0}; вариант: {1}; состояние: {2}" -f $s.stage,$s.variant,$s.state)
    Write-Host ("Задания текущей стадии: {0}/{1}; обновлено: {2}" -f $s.jobs_done,$s.jobs_total,$s.updated)
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) — только текущий этап" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    Write-Host 'CPU-стадия: 4 исходных трекера. Модели D и общая MASA — следующие стадии.'
    Write-Host 'Общий остаток пока неизвестен. Остановка своего дерева до 07:40 МСК24.10.'
    if (Test-Path -LiteralPath "$Run\controller_result.json") {$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Стадия завершена. Код: {0}; своё дерево завершено: {1}" -f $r.exit_code,$r.all_owned_ended);break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
