"""CPU-only guarded evaluation helpers; no tracking imports or mutation."""
import os,sys,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'runs/full_template_native_20261109';LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def cast(value):
    if isinstance(value,dict):return {k:cast(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [cast(v) for v in value]
    return value.tolist() if hasattr(value,'tolist') else value
def write(name,value,*,new=True):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(cast(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if new:
        with path.open('x',encoding='utf8') as f:f.write(data)
    else:
        temp=path.with_suffix(path.suffix+'.pending');temp.write_text(data,encoding='utf8')
        for _ in range(30):
            try:temp.replace(path);return
            except PermissionError:time.sleep(.1)
        raise PermissionError(str(path))
def status(stage,**kw):write('evaluation_status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**kw),new=False)
def verify(passport):
    for name,digest in passport['sources'].items():assert sha(ROOT/name)==digest,name
def start(timeout=7200):
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    if not (OUT/'evaluation_status.json').exists():status('starting')
    remaining=(STOP-datetime.now(MSK)).total_seconds();assert remaining>0
    from system_memory_guard import guard_entrypoint
    guard_entrypoint(timeout=min(timeout,remaining))
    import psutil
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
