$ErrorActionPreference = 'Stop'
$version = '4.7.1'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine'
$zip = Join-Path $env:TEMP "godot-$version.zip"
$url = "https://github.com/godotengine/godot/releases/download/$version-stable/Godot_v$version-stable_win64.exe.zip"

New-Item -ItemType Directory -Force $engine | Out-Null
Write-Output "Downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip
Expand-Archive -Path $zip -DestinationPath $engine -Force

$exe = Get-ChildItem $engine -Filter 'Godot_v*.exe' | Select-Object -First 1
if ($null -eq $exe) { throw "Godot executable not found after extraction" }
Move-Item $exe.FullName (Join-Path $engine 'Godot.exe') -Force

& (Join-Path $engine 'Godot.exe') --version
