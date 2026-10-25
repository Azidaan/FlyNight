$run='E:\Dissertation\runs\z13_region_product_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis: direct target/activity interaction'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: original 68-channel products / equal blank controls; CPU only' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.stage) | $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Models: $($s.models_done)/$($s.models_total); worker update: $($s.updated)"
        Write-Host "Elapsed: $($s.elapsed_seconds) seconds; remaining estimate unknown until first model."
        if ($s.total) {Write-Progress -Activity 'Current model' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
    }
    if (Test-Path -LiteralPath "$run\protocol.json") {
        $p=Get-Content -LiteralPath "$run\protocol.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "Save: $($p.save_at); hard stop: $($p.hard_stop) Europe/Moscow"
    }
    if (Test-Path -LiteralPath "$run\finish_recorded.json") {Write-Host 'Full result saved';break}
    if (Test-Path -LiteralPath "$run\pipeline_failure.json") {Write-Host 'Inspect saved pipeline failure' -ForegroundColor Red;break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Current model' -Completed
Read-Host 'Enter - close'
