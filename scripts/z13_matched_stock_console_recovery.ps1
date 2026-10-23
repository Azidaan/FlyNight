param([string]$Run='E:\Dissertation\runs\z13_matched_trackers_20261024\stock_controls')
$Host.UI.RawUI.WindowTitle='FlyVis — штатное ядро и контроли без мухи'
while ($true) {
    Clear-Host
    Write-Host 'Общие 7 видео: штатная C с двумя считывателями; два контроля без FlyVis'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    $n=@(Get-ChildItem -LiteralPath "$Run\jobs" -Filter '*.json' -ErrorAction SilentlyContinue).Count
    Write-Host ("Состояние: {0}; этап: {1}; вариант: {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Готовые задания: {0}/35; обновлено: {1}" -f $n,$s.updated)
    Write-Progress -Id 1 -Activity 'Задания всей очереди; оценка HOTA показана отдельно' -Status "$n/35" -PercentComplete ([math]::Min(100,100.0*$n/35))
    if ($s.total -gt 0) {Write-Progress -Id 2 -Activity $s.stage -Status "$($s.done)/$($s.total) — текущий этап" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    $last=Get-ChildItem -LiteralPath "$Run\logs" -Filter '*.stdout.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($last) {Write-Host 'Последняя строка текущего задания:'; Get-Content -LiteralPath $last.FullName -Tail 1 -Encoding UTF8; Write-Host ("Свежесть журнала: {0}" -f $last.LastWriteTime)}
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Начало: {0}; сохранение: {1}; конец своего дерева: {2}" -f $c.started,$c.save_at,$c.hard_stop)}
    Write-Host 'Пока основной прогон не завершён, очередь не занимает GPU. Остаток времени неизвестен.'
    Write-Host 'Бюджет после старта до 10800 секунд; абсолютное сохранение07:25 и конец до07:40МСК24.10.'
    if (Test-Path -LiteralPath "$Run\controller_result.json") {$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Завершено. Код: {0}; своё дерево завершено: {1}" -f $r.exit_code,$r.all_owned_ended);break}
    if (Test-Path -LiteralPath "$Run\queue_recovery_failure.json") {Get-Content -LiteralPath "$Run\queue_recovery_failure.json" -Raw -Encoding UTF8;break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
