"""Neutral CPU/GPU helpers; no CUDA visibility changes merely on import."""
import json,hashlib,os,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision'
TRAIN=ROOT/'runs/full_template_identity_20261109'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [cast(x) for x in v]
    if hasattr(v,'tolist'):return v.tolist()
    return v
def write(name,obj,*,new=True):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    if new:
        with path.open('x',encoding='utf8',newline='\n') as f:json.dump(cast(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        temporary=path.with_suffix(path.suffix+'.pending')
        with temporary.open('w',encoding='utf8',newline='\n') as f:json.dump(cast(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        temporary.replace(path)
def status(stage,**kwargs):write('status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kwargs),new=False)
def verify(protocol):
    for name,value in protocol['sources'].items():assert sha(ROOT/name)==value,name
def start(device='cpu',timeout=7200):
    os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
    if device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    else:assert device=='gpu'
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    if not (OUT/'status.json').exists():status('starting')
    remaining=(STOP-datetime.now(MSK)).total_seconds();assert remaining>0
    from system_memory_guard import guard_entrypoint
    guard_entrypoint(timeout=min(timeout,remaining))
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)

def journal(title,body):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';before=path.read_bytes();nl=b'\r\n' if b'\r\n' in before else b'\n';marker='## Журнал изменений'.encode()+nl
    assert before.count(marker)==1
    stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK');entry=('**'+stamp+'** — '+title+'\n\n'+body+'\n\n').replace('\n',nl.decode()).encode()
    temp=path.with_name('ЕДИНЫЙ_ПЛАН.native_template.pending');assert not temp.exists();temp.write_bytes(before.replace(marker,marker+nl+entry,1));assert path.read_bytes()==before;temp.replace(path)
