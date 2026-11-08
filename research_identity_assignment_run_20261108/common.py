import hashlib,json,os,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/identity_assignment_20261108'
MSK=timezone(timedelta(hours=3))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v,*,new=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if new:
        with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        temporary=p.with_suffix(p.suffix+'.pending')
        with temporary.open('w',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        os.replace(temporary,p)


def status(stage,**values):write(RUN/'status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**values))


def guard():
    sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
    from system_memory_guard import guard_entrypoint
    # Sunday08.11 -> nearest weekday stop Monday09.11.2026,07:40MSK.
    deadline=datetime(2026,11,9,7,40,tzinfo=MSK)
    return guard_entrypoint(timeout=max(1,int((deadline-datetime.now(MSK)).total_seconds())))


def verify_phase(name):
    p=read(RUN/name)
    for key,h in p['sources'].items():assert sha(ROOT/key)==h,key
    return p
