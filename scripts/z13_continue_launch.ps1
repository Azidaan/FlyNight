# Запуск z13_core_continue_v1 (только после разрешения автора): один процесс GPU, приоритет Idle, срок остановки, консоль прогресса.
param([Parameter(Mandatory = $true)][string]$Deadline)   # ISO с поясом, например 2026-10-17T23:30:00+03:00
$root = 'E:\Dissertation'; $run = "$root\runs\z13_continue_20261017"
if (Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$root\*" -and (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)").CommandLine -match 'z13_continue.py --all' }) { throw 'Прогон уже идёт — второй не запускается' }
$cmd = "cd /d $root && set PYTHONIOENCODING=utf-8 && .venv\Scripts\python.exe -u scripts\z13_continue.py --all --deadline $Deadline > `"$run\controller.out.log`" 2> `"$run\controller.err.log`" & echo %ERRORLEVEL% > `"$run\controller.exit`""
Remove-Item "$run\controller.exit" -ErrorAction SilentlyContinue
$p = Start-Process cmd.exe -ArgumentList '/c', $cmd -WindowStyle Hidden -PassThru
Start-Sleep 2
Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" | ForEach-Object { (Get-Process -Id $_.ProcessId).PriorityClass = 'Idle' }
$p.PriorityClass = 'Idle'
$c = Start-Process powershell.exe -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "$root\scripts\z13_continue_console.ps1" -PassThru
Start-Sleep 2
if (-not (Get-Process -Id $c.Id -ErrorAction SilentlyContinue)) { Write-Warning "Консоль не открылась; вручную: powershell -NoExit -File $root\scripts\z13_continue_console.ps1" } else { "консоль открыта, PID $($c.Id)" }
"контроллер запущен, PID $($p.Id)"
