@echo off
rem ASCII only on purpose: cmd parses .bat bytes in the console codepage,
rem so Cyrillic here breaks parsing before chcp can take effect.
cd /d "%~dp0"

if not exist "build\PotionNight.exe" goto nobuild

start "" "build\PotionNight.exe"
exit /b 0

:nobuild
echo.
echo Build not found: build\PotionNight.exe
echo Build it first:
echo     powershell -File tools\build_exe.ps1
echo.
pause
exit /b 1
