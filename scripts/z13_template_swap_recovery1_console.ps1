$sourcePath='E:\Dissertation\scripts\z13_template_swap_console.ps1'
$sourceText=[System.IO.File]::ReadAllText($sourcePath,[System.Text.Encoding]::UTF8)
$sourceText=$sourceText.Replace("$"+"run='E:\Dissertation\runs\z13_template_swap_20261025'","$"+"run='E:\Dissertation\runs\z13_template_swap_20261025\recovery1'")
& ([ScriptBlock]::Create($sourceText))
