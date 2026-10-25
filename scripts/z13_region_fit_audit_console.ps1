$run='E:\Dissertation\runs\z13_region_fit_audit_20261025'
$Host.UI.RawUI.WindowTitle='FlyVis: fit/calibration audit'
while ($true) {
    Clear-Host
    Write-Host 'FlyVis: frozen model audit; no training, no GPU' -ForegroundColor Cyan
    if (Test-Path -LiteralPath "$run\status.json") {
        $s=Get-Content -LiteralPath "$run\status.json" -Raw -Encoding UTF8 | ConvertFrom-Json
        Write-Host "$($s.state): $($s.variant) | $($s.done)/$($s.total)"
        Write-Host "Worker update: $($s.updated)"
        if ($s.total) {Write-Progress -Activity 'Fit/calibration audit' -Status "$($s.done)/$($s.total)" -PercentComplete ([Math]::Min(100,100*$s.done/$s.total))}
        if ($s.state -in @('completed','partial','failed')) {break}
    }
    if (Test-Path -LiteralPath "$run\controller_result.json") {break}
    Start-Sleep -Seconds 2
}
Write-Progress -Activity 'Fit/calibration audit' -Completed
Read-Host 'Enter - close'
