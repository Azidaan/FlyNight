$ErrorActionPreference = 'Stop'
$projectRoot = 'E:\Dissertation'
$runRoot = Join-Path $projectRoot 'runs\z13_codex_window_20261013'
$started = [DateTimeOffset]::Now
# Prior main training occupied under ten minutes. This window plus its
# four-minute stop margin keeps the original three-hour training budget.
$saveAt = $started.AddMinutes(165).ToString('yyyy-MM-ddTHH:mm:sszzz')
$hardStop = $started.AddMinutes(169).ToString('yyyy-MM-ddTHH:mm:sszzz')
$record = [ordered]@{
    authorized = 'Continue until B, C and D training completes; user reply 2026-10-13'
    started = $started.ToString('o')
    save_at = $saveAt
    hard_stop = $hardStop
    resume_arm = 'B'
    resume_step = 2827
    previous_checkpoint_verified = $true
    provenance_files = 138
    training_steps_per_arm = 4000
    heldout_val_opened = $false
    scope = 'B/C/D training and train-only threshold selection; no full dev/MASA evaluation'
}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'resume_authorization_20261013.json') -Encoding utf8
$arguments = @('-u', 'scripts/z13_window_runner.py', '--run-root', 'runs/z13_codex_window_20261013', '--save-at', $saveAt, '--hard-stop', $hardStop, '--resume')
$logPrefix = 'resume_' + $started.ToString('HHmmss')
$process = Start-Process -FilePath (Join-Path $projectRoot '.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot ($logPrefix + '.stdout.log')) -RedirectStandardError (Join-Path $runRoot ($logPrefix + '.stderr.log')) -PassThru
try { $process.PriorityClass = 'Idle' } catch { Write-Warning $_ }
$record.controller_wrapper_pid = $process.Id
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'resume_authorization_20261013.json') -Encoding utf8
$record | ConvertTo-Json -Depth 5
