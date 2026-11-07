"""Run remaining tests in fresh processes, sequentially, without shared caches."""
from pathlib import Path
import sys,os,json,time,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent/'per_file'
if len(sys.argv)>1:
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from tracking.windows_memory_job import MemoryJob
    limit_gib=int(sys.argv[3]) if len(sys.argv)>3 else 2
    assert limit_gib in (2,4)
    job=MemoryJob(limit_gib*1024**3)
    import torch,pytest
    torch.set_default_device('cpu');torch.set_num_threads(2)
    # CUDA_VISIBLE_DEVICES='' does not make is_available() false in this
    # Windows build. CPU-only verification must also prevent lazy CUDA init.
    torch.cuda.is_available=lambda:False
    torch.cuda.device_count=lambda:0
    def cpu_only(*args,**kwargs):
        raise RuntimeError('CUDA execution is outside this CPU-only verification')
    torch.cuda._lazy_init=cpu_only
    print('CPU_GUARD_V2: GPU branches intentionally not evaluated',flush=True)
    test=sys.argv[1];temp=sys.argv[2]
    assert test.startswith('tests/test_') and test.endswith('.py')
    code=pytest.main([test,'-vv','-x','--tb=short','-p','no:cacheprovider','--basetemp='+temp])
    print('MEMORY_JOB '+json.dumps(job.read()),flush=True)
    raise SystemExit(code or (2 if torch.cuda.is_initialized() else 0))
import psutil
OUT.mkdir(exist_ok=False)
files=json.loads((OUT.parent/'remaining/command.json').read_bytes())['files']
rows=[]
for test in files:
    assert psutil.virtual_memory().available>=13*1024**3
    name=Path(test).stem
    with (OUT/(name+'.log')).open('wb') as log:
        start=time.monotonic()
        p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__).resolve()),test,(OUT/name).as_posix()],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        owner=psutil.Process(p.pid);own={owner.pid:owner.create_time()};reason=None;peak=0
        while p.poll() is None:
            try:
                for child in owner.children(recursive=True):own.setdefault(child.pid,child.create_time())
            except psutil.NoSuchProcess:pass
            if psutil.virtual_memory().available<12.5*1024**3:reason='system_reserve';break
            if time.monotonic()-start>120:reason='timeout';break
            time.sleep(.2)
        if reason:
            # Stop only the known Python worker/shim with this exact test argument.
            for pid,birth in reversed(list(own.items())):
                try:
                    child=psutil.Process(pid)
                    if abs(child.create_time()-birth)<.01 and test in child.cmdline():child.kill()
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        try:code=p.wait(timeout=10)
        except subprocess.TimeoutExpired:code=124
    row=dict(test=test,returncode=code,reason=reason,seconds=time.monotonic()-start)
    rows.append(row)
    (OUT/'status.json').write_text(json.dumps(dict(done=len(rows),total=len(files),results=rows),indent=2),encoding='utf-8')
    print(json.dumps(row),flush=True)
    if reason:break
(OUT/'result.json').write_text(json.dumps(dict(completed=len(rows)==len(files),results=rows),indent=2),encoding='utf-8')
