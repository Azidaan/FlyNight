"""Shared file/metadata utilities; no Torch or model imports."""
import hashlib,json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()
def save(n,obj):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify():
    p=read(OUT/'protocol.json')
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p
def objects(video,frame):
    clock=read(ROOT/'runs/identity_assignment_20261108/full_precision/runtime_inputs'/(video+'.json'))
    a=clock['anchors'][str(frame)]
    labels=read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')
    by={int(x.get('frameIndex',i)):x for i,x in enumerate(labels)}
    row=by[a['frameIndex']];assert row['name']==a['name']
    return [dict(identity=str(o['id']),box=[o['box2d'][k] for k in ['x1','y1','x2','y2']]) for o in row['labels'] if o['category']=='car' and not (o.get('attributes') or {}).get('crowd',False)]
def overlap(a,b):
    import numpy as np
    a=np.array(a);b=np.array(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter
    return float(inter/union) if union>0 else 0.
def hits(box,objs):return [dict(identity=o['identity'],iou=overlap(box,o['box']),box=o['box']) for o in objs if overlap(box,o['box'])>=.5]
