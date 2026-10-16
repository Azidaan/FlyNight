param([string]$Run = 'E:\Dissertation\runs\z13_target_model_20261017', [string]$Title = 'z13_target_model_probe_v1')
# Консоль прогресса: только читает status.json; закрытие окна расчёт не останавливает.
$Host.UI.RawUI.WindowTitle = "$Title — прогресс"
while ($true) {
    Clear-Host
    Write-Host "Опыт: $Title"
    if (Test-Path "$Run\status.json") {
        try { $s = Get-Content "$Run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json } catch { Start-Sleep 5; continue }
        $arms = @('R0', 'C0', 'C', 'D'); $done = $arms | Where-Object { Test-Path "$Run\$_\rows.json" }
        Write-Host ("Готовые варианты: {0} из 4 ({1})" -f $done.Count, ($done -join ', '))
        Write-Host ("Состояние: {0}; вариант: {1} {2}; этап: {3}" -f $s.state, $s.variant, $s.variant_index, $s.stage)
        if ($s.episodes_total) {
            $pct = [math]::Round(100.0 * $s.episodes_done / $s.episodes_total, 1)
            Write-Progress -Activity "Текущий этап (не весь опыт): $($s.stage)" -Status "$($s.episodes_done)/$($s.episodes_total) эпизодов, $pct %" -PercentComplete $pct
            Write-Host ("Эпизоды этапа: {0}/{1} ({2} %)" -f $s.episodes_done, $s.episodes_total, $pct)
        } else { Write-Host 'Объём этапа неизвестен — процент не показывается' }
        Write-Host ("Прошло: {0} с; обновлено: {1}; срок: {2}" -f $s.elapsed_s, $s.updated, $s.deadline)
        if ($s.state -eq 'completed') { Write-Host "ИТОГ: отчёт $Run\report.json"; break }
    } else { Write-Host 'status.json ещё не создан' }
    if (Test-Path "$Run\controller.exit") { Write-Host ("Процесс завершился, код {0}" -f (Get-Content "$Run\controller.exit")); break }
    Start-Sleep 10
}
Read-Host 'Enter — закрыть окно (расчёт не затрагивается)'
