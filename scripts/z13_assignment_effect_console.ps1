param([string]$Run='E:\Dissertation\runs\z13_assignment_effect_20261024')
$Host.UI.RawUI.WindowTitle='FlyVis — последствия поправок для сопоставления, CPU'
while($true){
 Clear-Host
 try{$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8|ConvertFrom-Json}catch{Start-Sleep 3;continue}
 Write-Host 'Точное повторение исходного трекера. Без видео, GPU и обучения.'
 Write-Host "$($s.state): $($s.stage); $($s.variant)"
 Write-Host "Текущий ролик: $($s.done)/$($s.total); обновлено: $($s.updated); прошло: $($s.elapsed_seconds) с"
 if($s.total -gt 0){Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total)" -PercentComplete ([math]::Min(100,100*$s.done/$s.total))}
 $n=@(Get-ChildItem -LiteralPath "$Run\videos" -Filter '*.json' -ErrorAction SilentlyContinue).Count
 Write-Host "Сохранено полных видео: $n/130. Остаток времени пока неизвестен."
 if(Test-Path -LiteralPath "$Run\controller.json"){$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Сохранение: $($c.save_at); конец: $($c.hard_stop)"}
 if(Test-Path -LiteralPath "$Run\controller_result.json"){$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "РАСЧЁТ ЗАКОНЧЕН: код $($r.exit_code); своё дерево завершено: $($r.all_owned_ended). Окно оставлено для чтения результата.";break}
 Start-Sleep 3
}
Read-Host 'Enter закроет только это окно; расчёт уже закончен'
