$run='E:\Dissertation\runs\z13_region_product_20261025\recovery1'
$Host.UI.RawUI.WindowTitle='FlyVis direct interaction: cache-loader recovery1'
while ($true) {
    Clear-Host
    Write-Host 'CPU24 readouts: direct original-channel products / equal blank controls' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.stage) | $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Models: $($s.models_done)/$($s.models_total); update: $($s.updated)"
        Write-Host "Elapsed: $($s.elapsed_seconds) s; original hard stop 05:12:33 MSK."
        if ($s.total) {Write-Progress -Activity 'Current model' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
    }
    if (Test-Path -LiteralPath 'E:\Dissertation\runs\z13_region_product_20261025\finish_recorded.json') {Write-Host 'Full result saved';break}
    if (Test-Path -LiteralPath "$run\pipeline_failure.json") {Write-Host 'Inspect recovery1 failure' -ForegroundColor Red;break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Current model' -Completed
Read-Host 'Enter - close'
