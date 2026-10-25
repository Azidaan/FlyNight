"""Hide only verified project progress consoles, without stopping their processes."""
from pathlib import Path
import sys
import ctypes
import re
import subprocess
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import write,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
OUT=ROOT/'runs/z13_hidden_consoles_20261025'


def main():
    query="Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'powershell.exe' -and $_.CommandLine -match '(?i)-File +(?:\"?)E:\\\\Dissertation\\\\scripts\\\\z13_[^ ]*console[^ ]*\\.ps1'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    # Filter again below by a literal resolved project script path.
    query="Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'powershell.exe' -and $_.CommandLine -like '*-File*Dissertation*console*.ps1*'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],cwd=ROOT).decode('utf-8-sig')
    processes=json.loads(raw) if raw.strip() else []
    if isinstance(processes,dict):processes=[processes]
    kernel=ctypes.windll.kernel32;user=ctypes.windll.user32
    kernel.GetConsoleWindow.restype=ctypes.c_void_p
    user.IsWindowVisible.argtypes=[ctypes.c_void_p];user.ShowWindow.argtypes=[ctypes.c_void_p,ctypes.c_int]
    ps=WindowsProcesses();result=[]
    kernel.FreeConsole()
    for p in processes:
        match=re.search(r'(?i)-File\s+"?(E:\\Dissertation\\scripts\\z13_[^" ]*console[^" ]*\.ps1)',p['CommandLine'])
        if not match:continue
        script=Path(match[1]).resolve();script.relative_to(ROOT.resolve())
        assert script.is_file()
        pid=int(p['ProcessId']);birth=ps.creation(pid)
        if birth is None:continue
        attached=bool(kernel.AttachConsole(pid));handle=kernel.GetConsoleWindow() if attached else None
        before=bool(user.IsWindowVisible(handle)) if handle else False
        if handle:user.ShowWindow(handle,0)
        visible=bool(user.IsWindowVisible(handle)) if handle else False
        kernel.FreeConsole()
        result.append(dict(pid=pid,birth=birth,script=script.relative_to(ROOT).as_posix(),attached=attached,had_handle=bool(handle),was_visible=before,visible_after=visible,process_still_same=ps.creation(pid)==birth))
    assert all(not r['visible_after'] and r['process_still_same'] for r in result)
    write(OUT/'result.json',dict(at=now().isoformat(),consoles=result,processes_stopped=False,science_unchanged=True,hidden=sum(r['was_visible'] and not r['visible_after'] for r in result)))
    print(json.dumps(dict(checked=len(result),hidden=sum(r['was_visible'] for r in result),visible_after=sum(r['visible_after'] for r in result))))


if __name__=='__main__':main()
