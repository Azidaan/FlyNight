$run='E:\Dissertation\runs\z13_region_assignment_20261025\recovery2'
$Host.UI.RawUI.WindowTitle='FlyVis: learned region readout'
while ($true) {
    try {
        $root=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        $s=$root
        if ($root.phase -in @('labels','features','training')) {
            $phaseFile="$run\$($root.phase)\status.json"
            if (Test-Path -LiteralPath $phaseFile) {$s=Get-Content -LiteralPath $phaseFile -Raw -Encoding UTF8 | ConvertFrom-Json}
        }
        Clear-Host
        Write-Host 'FlyVis: обучаемый выбор реальной области и отказ' -ForegroundColor Cyan
        Write-Host "Этап: $($root.phase); $($s.stage)"
        Write-Host "Вариант: $($s.variant); выполнено на текущем этапе: $($s.done)/$($s.total)"
        if ($s.models_total) {Write-Host "Готовые модели: $($s.models_done)/$($s.models_total)"}
        Write-Host "Обновлено самим расчётом: $($s.updated)"
        Write-Host "Общий срок сохранения: $($p.pipeline_save_at)"
        Write-Host 'Свой бюджет каждого этапа меньше общего. Оставшееся время неизвестно.'
        if ($s.total) {Write-Progress -Activity 'Текущий этап опыта, не весь опыт' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($root.state -in @('completed','failed') -or (Test-Path -LiteralPath "$run\pipeline_failure.json")) {break}
    } catch {Write-Host $_.Exception.Message}
    Start-Sleep -Seconds 3
}
Write-Progress -Activity 'Текущий этап опыта, не весь опыт' -Completed
Read-Host 'Enter - закрыть окно'