"""Protect system memory while running the single real-network test."""
from pathlib import Path
import subprocess,sys,time,json,psutil,os
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
assert psutil.virtual_memory().available>=14*1024**3
owned={};reason=None;peak=0;minimum=psutil.virtual_memory().available
start=time.monotonic()
with (OUT/'native4.log').open('wb') as log:
    child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(OUT/'run_regression.py'),'native4'],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
    proc=psutil.Process(child.pid);owned[proc.pid]=proc.create_time()
    while child.poll() is None:
        try:
            for p in [proc]+proc.children(recursive=True):owned.setdefault(p.pid,p.create_time())
        except psutil.NoSuchProcess:pass
        private=0
        for pid,birth in owned.items():
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-birth)<.01:private+=p.memory_info().private
            except psutil.NoSuchProcess:pass
        peak=max(peak,private);free=psutil.virtual_memory().available;minimum=min(minimum,free)
        if free<12.5*1024**3:reason='system_reserve';break
        if time.monotonic()-start>180:reason='timeout';break
        time.sleep(.2)
    if reason:
        for pid,birth in reversed(list(owned.items())):
            try:
                p=psutil.Process(pid)
                if abs(p.create_time()-birth)<.01:p.kill()
            except psutil.NoSuchProcess:pass
    code=child.wait()
report=dict(returncode=code,reason=reason,peak_private_bytes=peak,minimum_available_bytes=minimum,owned=owned)
(OUT/'native_watchdog.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report));raise SystemExit(code or (1 if reason else 0))
