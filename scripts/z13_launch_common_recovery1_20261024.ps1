$ErrorActionPreference='Stop'
$root='E:\Dissertation';$run="$root\runs\z13_val20_matched_20261024\common_authorized_recovery1"
if (Test-Path -LiteralPath "$run\queue.json") {throw 'Queue already registered; no duplicate'}
$queue=Start-Process -FilePath "$root\.venv\Scripts\python.exe" -WorkingDirectory $root -WindowStyle Hidden -PassThru -ArgumentList @('-u',"$root\scripts\z13_common_authorized_recovery1_20261024.py",'--queue') -RedirectStandardOutput "$run\queue.stdout.log" -RedirectStandardError "$run\queue.stderr.log"
$queue.PriorityClass='Idle'
$console=Start-Process -FilePath 'powershell.exe' -WorkingDirectory $root -WindowStyle Normal -PassThru -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"$root\scripts\z13_common_recovery1_console_20261024.ps1")
Start-Sleep 3;$console.Refresh()
@{at=(Get-Date).ToUniversalTime().ToString('o');queue_launcher_pid=$queue.Id;console_pid=$console.Id;console_window_handle=$console.MainWindowHandle.ToInt64()} | ConvertTo-Json | Set-Content -LiteralPath "$run\launch.json" -Encoding UTF8
Get-Content -LiteralPath "$run\launch.json"
