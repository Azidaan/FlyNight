param([Parameter(Mandatory=$true)][string]$Run)
$Host.UI.RawUI.WindowTitle = 'FlyVis: full CPU target-center comparison'
$started = Get-Date
function Read-SharedJson([string]$Path) {
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, ([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
}
while ($true) {
    try {
        $s = Read-SharedJson (Join-Path $Run 'status.json')
        $receiptPath = Join-Path $Run 'full_launch.json'
        Clear-Host
        Write-Host 'FlyVis D / C / R1: full CPU comparison'
        if (Test-Path -LiteralPath $receiptPath) {
            $receipt = Read-SharedJson $receiptPath
            $started = [datetimeoffset]::Parse($receipt.started_at).LocalDateTime
            Write-Host ('Stop deadline: ' + $receipt.soft_deadline + '; hard stop: ' + $receipt.hard_deadline)
        }
        $completed = @('D','C','R1') | Where-Object { Test-Path -LiteralPath (Join-Path $Run ($_ + '/done.json')) }
        Write-Host ('Completed variants: ' + @($completed).Count + '/3')
        Write-Host ('Stage: ' + $s.stage + '; state: ' + $s.state)
        Write-Host ('Elapsed: ' + ((Get-Date)-$started).ToString('hh\:mm\:ss'))
        Write-Host ('Last status update: ' + (Get-Item -LiteralPath (Join-Path $Run 'status.json')).LastWriteTime)
        Write-Host 'ETA: unknown. Stage percentage is not total experiment percentage.'
        Write-Host 'Saved results: completed variants in D/C/R1. Current unfinished variant may not be saved.'
        Write-Host 'Closing this window does not stop or restart the computation.'
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Current stage: {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity 'Current stage only' -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Current stage total: unknown' }
        if ($s.state -in @('completed','failed','paused')) {
            Write-Progress -Activity 'Current stage only' -Completed
            break
        }
    } catch { Write-Host 'Waiting for a complete status record' }
    Start-Sleep -Seconds 2
}
Read-Host 'Finished. Press Enter to close'
