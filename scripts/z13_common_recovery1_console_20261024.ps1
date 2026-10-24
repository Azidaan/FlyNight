$run='E:\Dissertation\runs\z13_val20_matched_20261024\common_authorized_recovery1'
$Host.UI.RawUI.WindowTitle='FlyVis — продолжение20val, ожидание GPU и прогресс'
while ($true) {
    Clear-Host
    try {$s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json;$p=Get-Content -LiteralPath "$run\protocol.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ('Состояние: {0}; этап: {1}; вариант: {2}' -f $s.state,$s.stage,$s.variant)
    Write-Host ('Готово заданий: {0}/{1}; обновлено: {2}' -f $s.jobs_done,$s.jobs_total,$s.updated)
    Write-Host ('Сохранение: {0}; полная остановка: {1}' -f $p.save_at,$p.hard_stop)
    if ($s.state -eq 'queued') {Write-Host ('GPU свободно: {0} МиБ; загрузка: {1}%; чужих игровых процессов: {2}' -f $s.resources.free_gpu_mib,$s.resources.gpu_utilization_percent,$s.resources.active_game_identities.Count);Write-Host 'Очередь не использует GPU. Срок ожиданием не продлевается.'}
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) кадров текущего задания" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    if ($s.state -in @('completed','failed','paused')) {break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть окно наблюдения'
