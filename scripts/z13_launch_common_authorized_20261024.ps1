$ErrorActionPreference = 'Stop'
$root = 'E:\Dissertation'
$run = "$root\runs\z13_val20_matched_20261024\common_authorized"
$python = "$root\.venv\Scripts\python.exe"
if (Test-Path -LiteralPath "$run\controller.json") { throw 'Controller already exists; do not duplicate' }
if (-not (Test-Path -LiteralPath "$run\protocol.json")) { throw 'Prepare and verify protocol first' }
$controller = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru -ArgumentList @('-u', "$root\scripts\z13_bounded_recovery_controller_20261022.py", '--run', $run, '--script', "$root\scripts\z13_val20_common_authorized_20261024.py") -RedirectStandardOutput "$run\controller.stdout.log" -RedirectStandardError "$run\controller.stderr.log"
$controller.PriorityClass = 'Idle'
$monitor = Start-Process -FilePath $python -WorkingDirectory $root -WindowStyle Hidden -PassThru -ArgumentList @('-u', "$root\scripts\z13_val20_common_authorized_20261024.py", '--monitor') -RedirectStandardOutput "$run\monitor.stdout.log" -RedirectStandardError "$run\monitor.stderr.log"
$monitor.PriorityClass = 'Idle'
$console = Start-Process -FilePath 'powershell.exe' -WorkingDirectory $root -WindowStyle Normal -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$root\scripts\z13_val20_common_console.ps1", '-Run', $run)
Start-Sleep -Seconds 3
$console.Refresh()
@{at=(Get-Date).ToUniversalTime().ToString('o');launcher_pid=$controller.Id;monitor_launcher_pid=$monitor.Id;console_pid=$console.Id;console_window_handle=$console.MainWindowHandle.ToInt64()} | ConvertTo-Json | Set-Content -LiteralPath "$run\launch.json" -Encoding utf8
Get-Content -LiteralPath "$run\launch.json"
