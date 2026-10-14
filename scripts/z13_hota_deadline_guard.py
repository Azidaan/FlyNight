"""Independent deadline enforcement for an explicitly owned Windows process tree."""
import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_console_progress import read_json
from tracking.z13_checkpoint import atomic_json


class ProcessEntry(c.Structure):
    _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('pid', w.DWORD),
                ('heap', c.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
                ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD),
                ('exe', w.WCHAR * 260)]


class FileTime(c.Structure):
    _fields_ = [('low', w.DWORD), ('high', w.DWORD)]


class WindowsProcesses:
    def __init__(self):
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.k.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
        self.k.OpenProcess.restype = c.c_void_p
        self.k.CloseHandle.argtypes = (c.c_void_p,)
        self.k.GetProcessTimes.argtypes = (c.c_void_p, c.POINTER(FileTime), c.POINTER(FileTime), c.POINTER(FileTime), c.POINTER(FileTime))
        self.k.GetExitCodeProcess.argtypes = (c.c_void_p, c.POINTER(w.DWORD))
        self.k.CreateToolhelp32Snapshot.argtypes = (w.DWORD, w.DWORD)
        self.k.CreateToolhelp32Snapshot.restype = c.c_void_p
        self.k.Process32FirstW.argtypes = (c.c_void_p, c.POINTER(ProcessEntry))
        self.k.Process32NextW.argtypes = (c.c_void_p, c.POINTER(ProcessEntry))

    def creation(self, pid):
        handle = self.k.OpenProcess(0x1000, False, pid)
        if not handle:
            return None
        try:
            exit_code = w.DWORD()
            if not self.k.GetExitCodeProcess(handle, c.byref(exit_code)) or exit_code.value != 259:
                return None
            times = [FileTime() for _ in range(4)]
            if not self.k.GetProcessTimes(handle, *[c.byref(t) for t in times]):
                return None
            return (times[0].high << 32) | times[0].low
        finally:
            self.k.CloseHandle(handle)

    def children(self):
        snapshot = self.k.CreateToolhelp32Snapshot(2, 0)
        if snapshot in (None, c.c_void_p(-1).value):
            raise OSError(c.get_last_error())
        rows = []
        try:
            entry = ProcessEntry(); entry.size = c.sizeof(entry)
            ok = self.k.Process32FirstW(snapshot, c.byref(entry))
            while ok:
                rows.append((int(entry.pid), int(entry.parent)))
                ok = self.k.Process32NextW(snapshot, c.byref(entry))
        finally:
            self.k.CloseHandle(snapshot)
        return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path, required=True); args = ap.parse_args()
    run = args.run.resolve(); run.relative_to(ROOT)
    authorization = read_json(run / 'full_authorization.json')
    launch = read_json(run / 'launch_process.json')
    deadline = datetime.fromisoformat(authorization['deadline'])
    assert deadline.tzinfo is not None
    processes = WindowsProcesses()
    launcher = int(launch['launcher_pid'])
    known = {launcher: int(launch['launcher_start_filetime'])}
    assert processes.creation(launcher) == known[launcher], 'Launch process identity mismatch'
    folder = run / ('deadline_guard_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    atomic_json(folder / 'started.json', {'state': 'armed', 'deadline': authorization['deadline'],
                                         'launcher_pid': launcher, 'pid': __import__('os').getpid()})
    outcome = 'deadline_reached'
    while datetime.now().astimezone() < deadline - timedelta(seconds=15):
        live = {pid for pid, creation in known.items() if processes.creation(pid) == creation}
        rows = processes.children()
        changed = True
        while changed:
            changed = False
            for pid, parent in rows:
                if parent in live and pid not in live:
                    creation = processes.creation(pid)
                    if creation is not None and creation >= known[parent]:
                        known[pid] = creation; live.add(pid); changed = True
        status = read_json(run / 'status.json')
        if status.get('state') in ('completed', 'failed', 'paused'):
            outcome = 'controller_' + status['state']; break
        if not live:
            outcome = 'owned_processes_exited'; break
        time.sleep(1)
    if outcome == 'deadline_reached':
        (run / 'stop.request').write_text('Approved window ending; preserve completed videos.', encoding='utf-8')
        while datetime.now().astimezone() < deadline:
            time.sleep(.2)
        # Exact creation times prevent killing a reused PID; taskkill also stops descendants.
        errors = []
        for pid in [launcher] + [p for p in known if p != launcher]:
            if processes.creation(pid) == known[pid]:
                result = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               capture_output=True, check=False)
                if result.returncode and processes.creation(pid) == known[pid]:
                    errors.append({'pid': pid, 'exit_code': result.returncode})
        if errors:
            atomic_json(folder / 'termination_errors.json', errors)
            raise RuntimeError('Owned process termination denied; deadline guarantee unavailable')
    atomic_json(folder / 'result.json', {'state': outcome, 'ended': datetime.now().astimezone().isoformat(),
                                         'known_owned_processes': known})


if __name__ == '__main__':
    main()
