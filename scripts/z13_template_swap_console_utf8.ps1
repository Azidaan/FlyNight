$sourcePath='E:\Dissertation\scripts\z13_template_swap_console.ps1'
$sourceText=[System.IO.File]::ReadAllText($sourcePath,[System.Text.Encoding]::UTF8)
& ([ScriptBlock]::Create($sourceText))
