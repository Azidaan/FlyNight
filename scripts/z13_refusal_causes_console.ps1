param([string]$Run='E:\Dissertation\runs\z13_refusal_causes_20261025')
$Host.UI.RawUI.WindowTitle='FlyVis — причины неверного вмешательства'
while($true){
 Clear-Host
 try{$s=Get-Content -LiteralPath "$Run\status.json" -Raw -Encoding UTF8|ConvertFrom-Json}catch{Start-Sleep 2;continue}
 Write-Host 'Проверка причин: шанс помочь → предложение → разрешение → последствие'
 Write-Host 'Только сохранённые данные. Без обучения, GPU и изменения порога.'
 Write-Host "$($s.state): $($s.stage); $($s.variant)"
 Write-Host "Варианты: $($s.done)/$($s.total); обновлено: $($s.updated)"
 if($s.total -gt 0){Write-Progress -Activity 'Разбор фиксированного решения' -Status "$($s.done)/$($s.total)" -PercentComplete ([math]::Min(100,100*$s.done/$s.total))}
 if(Test-Path -LiteralPath "$Run\controller.json"){$c=Get-Content -LiteralPath "$Run\controller.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "Сохранение: $($c.save_at); конец: $($c.hard_stop)"}
 if(Test-Path -LiteralPath "$Run\controller_result.json"){$r=Get-Content -LiteralPath "$Run\controller_result.json" -Raw -Encoding UTF8|ConvertFrom-Json;Write-Host "РАСЧЁТ ЗАКОНЧЕН: код $($r.exit_code); дерево завершено: $($r.all_owned_ended). Окно оставлено для чтения.";break}
 Start-Sleep 2
}
Read-Host 'Enter закроет окно'
