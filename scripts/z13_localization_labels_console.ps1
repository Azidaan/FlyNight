param([string]$Run = 'E:\Dissertation\runs\z13_localization_labels_20261023')
$Host.UI.RawUI.WindowTitle = 'FlyVis — новая цель выбора центра'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: только цель обучения выборщика; CPU, 18 моделей'
    try { $s = Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json } catch { Start-Sleep 2; continue }
    Write-Host ("Состояние: {0}; этап: {1}; вариант: {2}" -f $s.state, $s.stage, $s.variant)
    Write-Host ("Готово моделей: {0}/18. Прошло: {1} с. Обновлено: {2}" -f $s.models_done, $s.elapsed_seconds, $s.updated)
    if ($s.total -gt 0) {
        $pct = [math]::Min(100, 100.0 * $s.done / $s.total)
        Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total) — только текущий этап" -PercentComplete $pct
    }
    Write-Host 'Общий процент и остаток времени неизвестны; подготовка не считается обучением.'
    if (Test-Path -LiteralPath "$Run\controller.json") {
        $c = Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host ("Сохранение/остановка: {0}; предельное завершение: {1}" -f $c.save_at, $c.hard_stop)
    }
    Write-Host "Сохранение: $Run"
    if (Test-Path -LiteralPath "$Run\controller_result.json") {
        $r = Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host ("Конец своего дерева: {0}; код: {1}; лимит: {2}" -f $r.all_owned_ended, $r.exit_code, $r.hard_timeout)
        break
    }
    Start-Sleep 3
}
Read-Host 'Enter — закрыть только окно наблюдения'
