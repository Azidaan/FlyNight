$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'Прогресс: проверка на новых val-видео'
Set-Location 'E:\Dissertation'
while ($true) {
    Clear-Host
    & '.\.venv\Scripts\python.exe' 'scripts\newval_progress.py'
    Write-Host ''
    Write-Host 'Обновление каждые 30 с. Закрыть окно — расчёт не прервётся.'
    Start-Sleep -Seconds 30
}
