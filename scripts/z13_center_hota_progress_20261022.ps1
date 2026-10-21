param([string]$Run='E:\Dissertation\runs\z13_center_hota_20261021')
$source=Get-Content -LiteralPath 'E:\Dissertation\scripts\z13_center_hota_progress_20261021.ps1' -Raw -Encoding UTF8
$source=$source.Replace("'protocol.json'","'resume1_authorization.json'").Replace("'queue_status.json'","'resume1_queue_status.json'").Replace("'active_job.json'","'resume1_active_job.json'").Replace("'night_finished.json'","'resume1_night_finished.json'").Replace("'progress_console.json'","'resume1_progress_console.json'")
$source=$source.Replace('FlyVis — центрированный считыватель в полном трекере','FlyVis — продолжение полного сравнения22.10')
# Original source is preserved, including read-only console behavior and final hold.
Invoke-Expression $source
