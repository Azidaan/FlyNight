# Консоль прогресса z13_core_continue_v1: только читает status.json и журналы; закрытие окна расчёт не останавливает.
$run = 'E:\Dissertation\runs\z13_continue_20261017'
$Host.UI.RawUI.WindowTitle = 'z13_core_continue_v1 — прогресс'
while ($true) {
    Clear-Host
    Write-Host 'Опыт: z13_core_continue_v1 — D-train против D-frozen, ступень 1 (по 1000 доп. шагов)'
    if (Test-Path "$run\status.json") {
        try { $s = Get-Content "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json } catch { Start-Sleep 5; continue }
        $done = @('baseline', 'D-frozen', 'D-train') | Where-Object { Test-Path "$run\$_\done.json" }
        Write-Host ("Готовые этапы: {0} из 3 ({1})" -f $done.Count, ($done -join ', '))
        Write-Host ("Состояние: {0}; этап: {1}; вариант: {2} {3}" -f $s.state, $s.stage, $s.variant, $s.variant_index)
        if ($s.extra_total) {
            $pct = [math]::Round(100.0 * $s.extra_step / $s.extra_total, 1)
            Write-Progress -Activity "Обучение $($s.variant) (только этот вариант, не вся пара)" -Status "$($s.extra_step)/$($s.extra_total) доп. шагов, $pct %" -PercentComplete $pct
            Write-Host ("Доп. шаги варианта: {0}/{1} ({2} %); с/шаг: {3}; осталось по варианту: {4} с" -f $s.extra_step, $s.extra_total, $pct, $s.seconds_per_step, $s.eta_variant_s)
        } else { Write-Host 'Объём текущего этапа в шагах не задан (аудит или оценка) — процент не показывается' }
        Write-Host ("Прошло: {0} с; последнее сохранение: {1}; обновлено: {2}" -f $s.elapsed_s, $s.last_saved, $s.updated)
        Write-Host ("Срок остановки: {0}" -f $s.deadline)
        if ($s.state -eq 'completed') { Write-Host 'ИТОГ: пара завершена, отчёт runs\z13_continue_20261017\report.json'; break }
    } else { Write-Host 'status.json ещё не создан' }
    if (Test-Path "$run\controller.err.log") { $e = Get-Content "$run\controller.err.log" -Tail 3 -Encoding UTF8; if ($e -match 'Error|Traceback') { Write-Host 'ОШИБКА:'; $e; } }
    if (Test-Path "$run\controller.exit") { Write-Host ("Процесс завершился, код {0}" -f (Get-Content "$run\controller.exit")); break }
    Start-Sleep 10
}
Read-Host 'Нажмите Enter, чтобы закрыть окно (расчёт это не затрагивает)'
