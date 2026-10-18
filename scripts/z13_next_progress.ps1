param([Parameter(Mandatory=$true)][string]$Run, [string]$Title = 'FlyVis')
$Host.UI.RawUI.WindowTitle = $Title
function Read-SharedJson([string]$Path) {
    $stream=[System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader=New-Object System.IO.StreamReader($stream,[System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
}
while ($true) {
    try {
        $s=Read-SharedJson (Join-Path $Run 'status.json')
        Clear-Host
        Write-Host $Title
        Write-Host ('State: '+$s.state+'; variant: '+$s.variant+'; stage: '+$s.stage)
        $controller=Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $controller) {
            $c=Read-SharedJson $controller
            Write-Host ('Elapsed: '+((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
            Write-Host ('Stop deadline: '+[datetimeoffset]::Parse($c.hard_deadline).LocalDateTime)
        }
        Write-Host ('Last update: '+$s.updated)
        Write-Host 'ETA: unknown unless shown below. Completed outputs remain saved.'
        Write-Host 'Closing this window does not stop or repeat the calculation.'
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Current stage: {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity $Title -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Current stage total: unknown' }
        if ($s.state -in @('completed','failed','paused')) { Write-Progress -Activity $Title -Completed; break }
    } catch { Write-Host 'Waiting for status' }
    Start-Sleep -Seconds 3
}
Read-Host 'Finished. Press Enter to close'
