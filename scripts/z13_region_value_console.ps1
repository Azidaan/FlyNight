$run='E:\Dissertation\runs\z13_region_value_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis: signed assignment utility'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: help +1 / neutral 0 / harm -1; CPU only' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.stage): $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Models saved: $($s.models_done)/$($s.models_total); update: $($s.updated)"
        if ($s.total) {Write-Progress -Activity 'Current model' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','failed')) {break}
    }
    if (Test-Path -LiteralPath "$run\monitor_failure.json") {Write-Host 'Inspect saved monitor error' -ForegroundColor Red;break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Current model' -Completed
Read-Host 'Enter - close'
