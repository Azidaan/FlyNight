"""Protect system reserve around the real three-frame pipeline check."""
from pathlib import Path
import subprocess,time,json,psutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
script=str(OUT/'pipeline_smoke.py')
assert psutil.virtual_memory().available>=14*1024**3
own={};minimum=psutil.virtual_memory().available;peak=0;reason=None
with (OUT/'pipeline_smoke.log').open('wb') as log:
    started=time.monotonic()
    child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',script],cwd=ROOT,
        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    proc=psutil.Process(child.pid);own[proc.pid]=proc.create_time()
    while child.poll() is None:
        try:
            for p in proc.children(recursive=True):
                if script in p.cmdline():own.setdefault(p.pid,p.create_time())
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        available=psutil.virtual_memory().available;minimum=min(minimum,available);private=0
        for pid,birth in own.items():
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-birth)<.01:private+=p.memory_info().private
            except psutil.NoSuchProcess:pass
        peak=max(peak,private)
        if available<12.5*1024**3:reason='system_reserve';break
        if time.monotonic()-started>150:reason='timeout';break
        time.sleep(.2)
    if reason:
        for pid,birth in reversed(list(own.items())):
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-birth)<.01 and script in p.cmdline():p.kill()
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    code=child.wait()
report=dict(returncode=code,reason=reason,peak_private_bytes=peak,minimum_available_bytes=minimum,owned=own)
(OUT/'pipeline_watchdog.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report));raise SystemExit(code or (1 if reason else 0))
