param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference = 'Stop'
$runPath = (Resolve-Path -LiteralPath $RunDirectory).Path
$authorization = Get-Content -LiteralPath (Join-Path $runPath 'full_authorization.json') -Raw | ConvertFrom-Json
$launch = Get-Content -LiteralPath (Join-Path $runPath 'launch_process.json') -Raw | ConvertFrom-Json
$deadline = [DateTimeOffset]::Parse($authorization.deadline)
$guardDirectory = Join-Path $runPath ('deadline_guard_' + (Get-Date -Format yyyyMMdd_HHmmss))
New-Item -ItemType Directory -Path $guardDirectory | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$known = @{}
$known[[string]$launch.launcher_pid] = [string]$launch.launcher_start_ticks
$started = @{ state='armed'; deadline=$authorization.deadline; launcher_pid=$launch.launcher_pid; guard_pid=$PID }
[IO.File]::WriteAllText((Join-Path $guardDirectory 'started.json'), ($started | ConvertTo-Json), $utf8)
$outcome = 'deadline_reached'
try {
    while ([DateTimeOffset]::Now -lt $deadline.AddSeconds(-15)) {
        $statusPath = Join-Path $runPath 'status.json'
        try {
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            foreach ($entry in @($status.controller_pid, $status.child_pid)) {
                if ($entry) {
                    $process = Get-Process -Id ([int]$entry) -ErrorAction SilentlyContinue
                    if ($process -and $process.ProcessName -match '^python(w)?$' -and
                        $process.StartTime.Ticks -ge [long]$launch.launcher_start_ticks) {
                        $known[[string]$entry] = [string]$process.StartTime.Ticks
                    }
                }
            }
            if ($status.state -in @('completed','failed','paused')) {
                $outcome = 'controller_' + $status.state
                break
            }
        } catch { }
        Start-Sleep -Seconds 2
    }
    if ($outcome -eq 'deadline_reached') {
        [IO.File]::WriteAllText((Join-Path $runPath 'stop.request'), 'Authorized ten-hour window ending; preserve completed videos.', $utf8)
        while ([DateTimeOffset]::Now -lt $deadline) { Start-Sleep -Milliseconds 250 }
        foreach ($key in @($known.Keys)) {
            $process = Get-Process -Id ([int]$key) -ErrorAction SilentlyContinue
            if ($process -and [string]$process.StartTime.Ticks -eq $known[$key]) {
                & "$env:SystemRoot\System32\taskkill.exe" /PID $key /T /F | Out-Null
            }
        }
    }
} finally {
    $result = @{ state=$outcome; known_owned_processes=$known; ended=[DateTimeOffset]::Now.ToString('o') }
    [IO.File]::WriteAllText((Join-Path $guardDirectory 'result.json'), ($result | ConvertTo-Json -Depth 4), $utf8)
}
