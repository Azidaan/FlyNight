param([string]$Run='E:\Dissertation\runs\z13_unperturbed_readout_20261024')
$Host.UI.RawUI.WindowTitle='FlyVis — обучение считывателя, 36 моделей CPU'
while($true){
 Clear-Host
 try{$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8|ConvertFrom-Json}catch{Start-Sleep 3;continue}
 Write-Host 'Сравнение обучения без сдвига окна и равной смешанной выборки. Ядро не обучается.'
 Write-Host "$($s.state): $($s.stage); $($s.variant)"
 Write-Host "Готово моделей: $($s.models_done)/36; прошло: $($s.elapsed_seconds) с; обновлено: $($s.updated)"
 if($s.total -gt 0){Write-Progress -Activity 'Текущая модель' -Status "$($s.done)/$($s.total) шагов" -PercentComplete ([math]::Min(100,100*$s.done/$s.total))}
 if(Test-Path -LiteralPath "$Run\controller.json"){$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Сохранение: $($c.save_at); конец: $($c.hard_stop)"}
 Write-Host 'Остаток времени пока неизвестен. Сохранение:' $Run
 if(Test-Path -LiteralPath "$Run\controller_result.json"){$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "РАСЧЁТ ЗАКОНЧЕН: код $($r.exit_code); дерево завершено: $($r.all_owned_ended). Окно просто оставлено для чтения итога.";break}
 Start-Sleep 3
}
Read-Host 'Enter закроет это окно; вычисления уже закончены'
