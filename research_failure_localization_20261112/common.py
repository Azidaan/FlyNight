"""Bounded archived-event diagnostics. Never instantiate or update a tracker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json, hashlib, sys, gzip
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
NATIVE=ROOT/'runs/domain_matched_native_20261109'
LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision'
ID='identity_failure_localization_20261112'
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    return v.tolist() if hasattr(v,'tolist') else v
def save(n,d):
    with (OUT/n).open('x',encoding='utf8') as f:json.dump(cast(d),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def events(video):
    with gzip.open(NATIVE/'rollouts/C34'/(video+'.events.json.gz'),'rt',encoding='utf8') as f:return json.load(f)
def annotations(video):
    gt=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
    by={int(x.get('frameIndex',i)):x for i,x in enumerate(gt)}
    clock=read(LEGACY/'runtime_inputs'/(video+'.json'))
    ann={}
    for fr,a in clock['anchors'].items():
        row=by[a['frameIndex']];assert row['name']==a['name'];ann[int(fr)]=row.get('labels') or []
    return ann
def hits(box,objects):
    import numpy as np
    a=np.array(box);out=[]
    for o in objects:
        if o['category']!='car' or (o.get('attributes') or {}).get('crowd',False):continue
        b=np.array([o['box2d'][k] for k in ['x1','y1','x2','y2']]);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
        if union>0 and inter/union>=.5:out.append(dict(identity=str(o['id']),iou=float(inter/union),box=b.tolist()))
    return out
def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p
def journal(message):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=p.read_bytes();s=raw.decode('utf8');newline='\r\n' if '\r\n' in s else '\n';marker='## Журнал изменений'+newline;assert s.count(marker)==1
    entry='\n**'+datetime.now(ZoneInfo('Europe/Moscow')).strftime('%d.%m.%Y, %H:%M MSK')+'** — '+message+'\n\n'
    assert p.read_bytes()==raw;p.write_bytes(s.replace(marker,marker+entry.replace('\n',newline),1).encode('utf8'))
    with (OUT/'journal_additions.txt').open('a',encoding='utf8') as f:f.write(entry)
