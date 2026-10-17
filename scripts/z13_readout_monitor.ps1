param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis: readout experiment'
while ($true) {
    $path = Join-Path $Run 'status.json'
    if (Test-Path -LiteralPath $path) {
        try {
            $s = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            Clear-Host
            Write-Host 'FlyVis readout experiment'
            Write-Host $Run
            Write-Host ($s | ConvertTo-Json -Depth 4)
            Write-Host ('Last update: ' + (Get-Item -LiteralPath $path).LastWriteTime)
            Write-Host 'Closing this window does not stop the worker.'
            if ($s.total -gt 0) { Write-Progress -Activity 'Current stage' -Status $s.stage -PercentComplete ([Math]::Min(100,100*$s.done/$s.total)) }
            if ($s.state -in @('completed','failed','paused')) { Write-Progress -Activity 'Current stage' -Completed; break }
        } catch { Write-Host 'Waiting for a complete status file' }
    }
    Start-Sleep -Seconds 2
}
Read-Host 'Finished. Press Enter to close'
