$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$godot = Join-Path $root 'engine\Godot_console.exe'
$out = Join-Path $root 'build\PotionNight.exe'

if (-not (Test-Path $godot)) {
    throw "Engine not found. Run tools\fetch_godot.ps1 first."
}

$templates = Join-Path $root 'engine\editor_data\export_templates\4.7.1.stable'
if (-not (Test-Path $templates)) {
    throw "Export templates missing at $templates. Run tools\fetch_export_templates.ps1 first."
}

New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null

Write-Output 'Importing assets...'
& $godot --headless --path $root --import | Out-Null

Write-Output 'Exporting release build...'
& $godot --headless --path $root --export-release 'Windows Desktop' $out

if (-not (Test-Path $out)) {
    throw 'Export produced no file.'
}
'{0:N1} MB  {1}' -f ((Get-Item $out).Length / 1MB), $out
