param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle='FlyVis — проверка воспроизведения, CPU'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: служебная проверка прежнего ответа; эффективность не измеряется'
    try {$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json} catch {Start-Sleep 3;continue}
    Write-Host ("Состояние: {0}; этап: {1}; вариант: {2}" -f $s.state,$s.stage,$s.variant)
    Write-Host ("Обновлено: {0}; прошло: {1} секунд" -f $s.updated,$s.elapsed_seconds)
    if ($s.total -gt 0) {Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) этапов" -PercentComplete ([math]::Min(100,100.0*$s.done/$s.total))}
    if (Test-Path -LiteralPath "$Run\controller.json") {$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json;Write-Host ("Сохранение: {0}; конец дерева: {1}" -f $c.save_at,$c.hard_stop)}
    Write-Host 'CPU, до300секунд; оставшееся время по этапам неизвестно. GPUзанята общим сравнением.'
    if (Test-Path -LiteralPath "$Run\controller_result.json") {Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8;break}
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
