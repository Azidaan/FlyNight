@echo off
setlocal
cd /d "%~dp0"
"..\..\.venv\Scripts\python.exe" "server.py"
if errorlevel 1 pause
