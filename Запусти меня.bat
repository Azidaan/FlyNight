@echo off
rem Zapusk paneli upravleniya obucheniem odnim dvoynym klikom.
rem Fayl sohranyon v kodirovke OEM (cp866) - inache cmd ne prochtyot
rem kirillicu v imeni .py-fayla.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto novenv
start "" ".venv\Scripts\pythonw.exe" "Запусти меня.py"
exit /b 0

:novenv
echo.
echo Ne nayden .venv\Scripts\pythonw.exe
echo Okruzhenie proekta dolzhno lezhat v E:\Dissertation\.venv
echo.
pause
exit /b 1