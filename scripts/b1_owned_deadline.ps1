param([Parameter(Mandatory=$true)][string]$RunDir)
$ErrorActionPreference = 'Stop'
$guardTaskLaunch = Get-Content -LiteralPath (Join-Path $RunDir 'launch.json') -Raw | ConvertFrom-Json
$guardTaskProtocol = Get-Content -LiteralPath (Join-Path $RunDir 'protocol.json') -Raw | ConvertFrom-Json
$guardTaskDeadline = [DateTimeOffset]::Parse($guardTaskProtocol.deadline)
$guardTaskRoot = Get-Process -Id $guardTaskLaunch.pid
if ([math]::Abs(($guardTaskRoot.StartTime.ToUniversalTime() - [DateTimeOffset]::Parse($guardTaskLaunch.created).UtcDateTime).TotalSeconds) -gt 0.01) { throw 'Root PID identity changed' }
$guardTaskProcesses = @(@{pid=$guardTaskRoot.Id; created=$guardTaskRoot.StartTime.ToUniversalTime().ToString('o')})
$guardTaskChildren = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($guardTaskRoot.Id)")
foreach ($guardTaskChild in $guardTaskChildren) {
    $guardTaskProcess = Get-Process -Id $guardTaskChild.ProcessId -ErrorAction SilentlyContinue
    if ($guardTaskProcess) { $guardTaskProcesses += @{pid=$guardTaskProcess.Id; created=$guardTaskProcess.StartTime.ToUniversalTime().ToString('o')} }
}
$guardTaskProcesses | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir 'owned_process_tree.json') -Encoding UTF8
$guardTaskTimedOut = $false
while ($true) {
    $guardTaskAlive = @()
    foreach ($guardTaskIdentity in $guardTaskProcesses) {
        $guardTaskProcess = Get-Process -Id $guardTaskIdentity.pid -ErrorAction SilentlyContinue
        if ($guardTaskProcess -and [math]::Abs(($guardTaskProcess.StartTime.ToUniversalTime() - [DateTimeOffset]::Parse($guardTaskIdentity.created).UtcDateTime).TotalSeconds) -le 0.01) { $guardTaskAlive += $guardTaskProcess }
    }
    if ($guardTaskAlive.Count -eq 0) { break }
    if ([DateTimeOffset]::UtcNow -ge $guardTaskDeadline.AddSeconds(-3)) {
        $guardTaskTimedOut = $true
        foreach ($guardTaskProcess in ($guardTaskAlive | Sort-Object Id)) { Stop-Process -InputObject $guardTaskProcess -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 1
        continue
    }
    Start-Sleep -Seconds 2
}
@{timed_out=$guardTaskTimedOut; all_owned_ended=$true; owned_pids=@($guardTaskProcesses | ForEach-Object {$_.pid}); updated=[DateTimeOffset]::Now.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDir 'deadline_guard_result.json') -Encoding UTF8
