$run='E:\Dissertation\runs\z13_region_context_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis: observable assignment context'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: assignment context / equal blank controls; CPU only' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.phase): $($s.stage) | $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Models: $($s.models_done)/$($s.models_total); worker update: $($s.updated)"
        if ($s.total) {Write-Progress -Activity 'Current phase' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
    }
    if (Test-Path -LiteralPath "$run\finish_recorded.json") {Write-Host 'Full result saved';break}
    if (Test-Path -LiteralPath "$run\pipeline_failure.json") {Write-Host 'Inspect saved pipeline failure' -ForegroundColor Red;break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Current phase' -Completed
Read-Host 'Enter - close'
