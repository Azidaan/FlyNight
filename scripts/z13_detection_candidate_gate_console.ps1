param([string]$Run='E:\Dissertation\runs\z13_detection_candidate_gate_20261024')
$Host.UI.RawUI.WindowTitle='FlyVis — кандидаты детектора, обучающие эпизоды'
while($true){
  Clear-Host
  try{$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8|ConvertFrom-Json}catch{Start-Sleep 2;continue}
  Write-Host 'FlyVis: проверка запаса трёх рамок вместо трёх пиков. Только CPU.'
  Write-Host "$($s.state): $($s.stage) — $($s.variant)"
  if($s.total -gt 0){Write-Progress -Activity 'Обучающие видео' -Status "$($s.done)/$($s.total)" -PercentComplete ([math]::Min(100,100*$s.done/$s.total))}
  Write-Host "Прошло: $($s.elapsed_seconds) с; обновлено: $($s.updated). Остаток неизвестен."
  if(Test-Path -LiteralPath "$Run\controller.json"){$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Сохранение: $($c.save_at); полная остановка: $($c.hard_stop)"}
  Write-Host "Результаты: $Run"
  if(Test-Path -LiteralPath "$Run\controller_result.json"){$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Код: $($r.exit_code); своё дерево окончено: $($r.all_owned_ended)";break}
  Start-Sleep 2
}
Read-Host 'Enter — закрыть только окно наблюдения'
