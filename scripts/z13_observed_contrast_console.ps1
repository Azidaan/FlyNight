param([string]$Run='E:\Dissertation\runs\z13_observed_contrast_20261025')
$Host.UI.RawUI.WindowTitle='FlyVis — цель против наблюдавшегося фона'
while($true){
 Clear-Host
 try{$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8|ConvertFrom-Json}catch{Start-Sleep 2;continue}
 Write-Host 'Новый узел считывания:цель и конкретные фоновые образцы'
 Write-Host 'Сохранённые признаки;CPU;без обучения и новых данных.'
 Write-Host "$($s.state): $($s.stage); $($s.variant); $($s.done)/$($s.total)"
 Write-Host "Последнее обновление: $($s.updated);точный остаток времени неизвестен"
 if($s.total -gt 0){Write-Progress -Activity $s.stage -Status "$($s.done)/$($s.total)" -PercentComplete ([math]::Min(100,100*$s.done/$s.total))}
 if(Test-Path "$Run\controller.json"){$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Сохранение:$($c.save_at);конец:$($c.hard_stop)"}
 if(Test-Path "$Run\controller_result.json"){$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "РАСЧЁТ ЗАКОНЧЕН:код$($r.exit_code);своё дерево окончено:$($r.all_owned_ended). Окно оставлено для чтения.";break}
 Start-Sleep 2
}
Read-Host 'Enter закроет окно'
