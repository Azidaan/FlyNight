$run='E:\Dissertation\runs\z13_region_value_20261025'
$control="$run\recovery1"
$Host.UI.RawUI.WindowTitle='FlyVis: signed utility - recovery1'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis signed utility: recovered CPU worker' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.stage): $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Models saved: $($s.models_done)/$($s.models_total); worker update: $($s.updated)"
        Write-Host 'Old parent errors are preserved. Current controller: recovery1.'
        if ($s.total) {Write-Progress -Activity 'Current model' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','failed')) {break}
    }
    if (Test-Path -LiteralPath "$control\monitor_failure.json") {Write-Host 'New recovery monitor error' -ForegroundColor Red;break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Current model' -Completed
Read-Host 'Enter - close'
