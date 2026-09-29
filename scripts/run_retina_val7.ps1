# Н9/М1: фовеальный поток на val-семёрке по режимам сетчатки, по одному видео на процесс
# (один длинный процесс на 7 видео зависал при заполнении памяти GPU 18.09 20:52–21:27).
$py = "E:\Dissertation\.venv\Scripts\python.exe"
$videos = @("b1c66a42-6f7d68ca","b1c81faa-3df17267","b1cac6a7-04e33135","b1cebfb7-284f5117","b232c7c9-d251d9ee","b1d22449-15fb948f","b251064f-e7a165fd")
$modes = @(@("photoreceptor","runs\flyvis_fovea_flow_retina_photo"), @("lamina","runs\flyvis_fovea_flow_retina_lamina"), @("norm","runs\flyvis_fovea_flow_retina_norm"))
foreach ($m in $modes) {
    foreach ($v in $videos) {
        if (Test-Path "E:\Dissertation\$($m[1])\$v\fovea_flow.npz") { continue }
        & $py -u scripts\build_foveated_readout_cache.py --decode-flow --retina $m[0] --output $m[1] --video-id $v 2>&1 | Select-String -Pattern "^\[OK\]|Error|Traceback" | ForEach-Object { "$(Get-Date -Format 'HH:mm:ss') [$($m[0])] $_" }
    }
}
"$(Get-Date -Format 'HH:mm:ss') DONE"
