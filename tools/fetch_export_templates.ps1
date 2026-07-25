$ErrorActionPreference = 'Stop'
$version = '4.7.1'
$root = Split-Path -Parent $PSScriptRoot
$engine = Join-Path $root 'engine'

# Self-contained mode: Godot keeps its data next to the binary instead of
# %APPDATA%, so the whole toolchain stays on the project drive.
$sc = Join-Path $engine '_sc_'
if (-not (Test-Path $sc)) { New-Item -ItemType File $sc | Out-Null }

$dest = Join-Path $engine "editor_data\export_templates\$version.stable"
if (Test-Path (Join-Path $dest 'windows_release_x86_64.exe')) {
    Write-Output "Templates already present at $dest"
    exit 0
}

$url = "https://github.com/godotengine/godot/releases/download/$version-stable/Godot_v$version-stable_export_templates.tpz"
$tpz = Join-Path $env:TEMP "godot_templates_$version.tpz"

if (-not (Test-Path $tpz)) {
    Write-Output "Downloading $url (about 1 GB)"
    Invoke-WebRequest -Uri $url -OutFile $tpz
}

# Expand-Archive refuses .tpz by extension, so unzip through .NET instead.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$staging = Join-Path $env:TEMP "godot_templates_$version"
if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
[System.IO.Compression.ZipFile]::ExtractToDirectory($tpz, $staging)

New-Item -ItemType Directory -Force $dest | Out-Null
Copy-Item (Join-Path $staging 'templates\*') $dest -Recurse -Force

Write-Output "Templates installed to $dest"
