"""Run one verification in a Windows Job with a hard total commit-memory limit."""
import argparse
import ctypes
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

class BASIC(ctypes.Structure):
    _fields_ = [('ProcessTime',ctypes.c_longlong),('JobTime',ctypes.c_longlong),('Flags',w.DWORD),
                ('MinimumWorkingSet',ctypes.c_size_t),('MaximumWorkingSet',ctypes.c_size_t),
                ('ActiveProcessLimit',w.DWORD),('Affinity',ctypes.c_size_t),
                ('PriorityClass',w.DWORD),('SchedulingClass',w.DWORD)]

class IO(ctypes.Structure):
    _fields_ = [(name,ctypes.c_ulonglong) for name in ('ReadOps','WriteOps','OtherOps','ReadBytes','WriteBytes','OtherBytes')]

class EXTENDED(ctypes.Structure):
    _fields_ = [('Basic',BASIC),('Io',IO),('ProcessMemoryLimit',ctypes.c_size_t),
                ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),
                ('PeakJobMemoryUsed',ctypes.c_size_t)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',required=True)
    ap.add_argument('--mib',type=int,default=384)
    ap.add_argument('--timeout',type=int,default=300)
    ap.add_argument('command',nargs=argparse.REMAINDER)
    args=ap.parse_args()
    assert os.name=='nt' and 64<=args.mib<=512 and 1<=args.timeout<=360
    out=(ROOT/args.out).resolve();out.relative_to((ROOT/'runs/flyvis_repairs_20261106').resolve())
    assert not out.exists()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    assert command
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateJobObjectW.argtypes=[w.LPVOID,w.LPCWSTR];api.CreateJobObjectW.restype=w.HANDLE
    api.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,w.LPVOID,w.DWORD];api.SetInformationJobObject.restype=w.BOOL
    api.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];api.AssignProcessToJobObject.restype=w.BOOL
    api.GetCurrentProcess.restype=w.HANDLE
    api.QueryInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,w.LPVOID,w.DWORD,w.LPVOID]
    api.QueryInformationJobObject.restype=w.BOOL
    def checked(ok):
        if not ok: raise ctypes.WinError(ctypes.get_last_error())
    job=api.CreateJobObjectW(None,None);checked(job)
    limits=EXTENDED();limits.Basic.Flags=0x100|0x200|0x2000
    limits.ProcessMemoryLimit=limits.JobMemoryLimit=args.mib*1024*1024
    checked(api.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)))
    checked(api.AssignProcessToJobObject(job,api.GetCurrentProcess()))
    # The child inherits this Job; if setup fails, no verification is launched.
    out.mkdir(parents=True)
    start=time.monotonic()
    report=dict(command=command,hard_job_memory_bytes=limits.JobMemoryLimit,
                current_process_assigned_before_spawn=True,child_inherits_job=True,completed=False)
    (out/'started.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        with (out/'output.log').open('wb') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                  timeout=args.timeout,creationflags=subprocess.CREATE_NO_WINDOW,
                                  env=dict(os.environ,PYTHONUTF8='1',
                                           PYTEST_ADDOPTS='--basetemp='+(out/'pytest_temp').as_posix()))
        report.update(returncode=result.returncode,completed=True)
    except subprocess.TimeoutExpired:
        report.update(returncode=124,timed_out=True)
    finally:
        observed=EXTENDED()
        checked(api.QueryInformationJobObject(job,9,ctypes.byref(observed),ctypes.sizeof(observed),None))
        report.update(seconds=time.monotonic()-start,peak_process_bytes=observed.PeakProcessMemoryUsed,
                      peak_job_bytes=observed.PeakJobMemoryUsed)
        (out/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)
    # OS closes the owned job handle on process exit and kills remaining children.
    raise SystemExit(report['returncode'])

if __name__=='__main__':
    main()
