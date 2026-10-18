param([string]$Run = 'E:\Dissertation\runs\z13_target_center_dev_20261018')
$Host.UI.RawUI.WindowTitle = 'FlyVis dev35: frozen target-center validation'
function Read-SharedJson([string]$Path) {
    $stream=[System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    $reader=New-Object System.IO.StreamReader($stream,[System.Text.Encoding]::UTF8)
    try { return ($reader.ReadToEnd() | ConvertFrom-Json) } finally { $reader.Dispose() }
}
while ($true) {
    try {
        $s=Read-SharedJson (Join-Path $Run 'status.json')
        Clear-Host
        Write-Host 'FlyVis dev35: frozen parameters, CPU'
        $controllerPath=Join-Path $Run 'controller.json'
        if (Test-Path -LiteralPath $controllerPath) {
            $c=Read-SharedJson $controllerPath
            Write-Host ('Elapsed: ' + ((Get-Date)-[datetimeoffset]::Parse($c.started_at).LocalDateTime).ToString('hh\:mm\:ss'))
            Write-Host ('Soft deadline: ' + [datetimeoffset]::Parse($c.soft_deadline).LocalDateTime)
            Write-Host ('Hard deadline: ' + [datetimeoffset]::Parse($c.hard_deadline).LocalDateTime)
        }
        Write-Host ('State: '+$s.state+'; stage: '+$s.stage)
        Write-Host ('Variant: '+$s.variant+'; video: '+$s.video)
        Write-Host ('Last update: '+(Get-Item -LiteralPath (Join-Path $Run 'status.json')).LastWriteTime)
        $dataCount=@(Get-ChildItem -LiteralPath (Join-Path $Run 'data') -Filter '*.npz' -ErrorAction SilentlyContinue).Count
        Write-Host ('Extracted videos: '+$dataCount+'/35')
        foreach ($arm in @('D','C','R1')) {
            $count=@(Get-ChildItem -LiteralPath (Join-Path $Run $arm) -Filter 'done.json' -Recurse -ErrorAction SilentlyContinue).Count
            Write-Host ($arm+': saved videos '+$count+'/35')
        }
        Write-Host 'ETA: unknown. Percentage below concerns only the current stage.'
        Write-Host 'Completed video results are saved separately. Closing this window does not stop or repeat the run.'
        if ($s.total -gt 0) {
            $percent=[Math]::Min(100,100*$s.done/$s.total)
            Write-Host ('Current stage {0}/{1} ({2:N1}%)' -f $s.done,$s.total,$percent)
            Write-Progress -Activity 'Current stage only' -Status $s.stage -PercentComplete $percent
        } else { Write-Host 'Current stage total: unknown' }
        if ($s.state -in @('completed','failed','paused')) { Write-Progress -Activity 'Current stage only' -Completed; break }
    } catch { Write-Host 'Waiting for status' }
    Start-Sleep -Seconds 2
}
Read-Host 'Finished. Press Enter to close'
