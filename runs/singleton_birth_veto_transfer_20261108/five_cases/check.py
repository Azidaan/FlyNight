"""Independent scalar check of the two-frame warning and candidate/alternative overlap."""
import json,hashlib
from statistics import median
from pathlib import Path
from math import log,exp,isclose
import numpy as np
P=Path(__file__).resolve().parent;D=P.parent;ROOT=D.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def ov(a,b):
    x,y,X,Y=map(float,a);u,v,U,V=map(float,b);i=max(0,min(X,U)-max(x,u))*max(0,min(Y,V)-max(y,v))
    return i/((X-x)*(Y-y)+(U-u)*(V-v)-i)
def val(b):
    x,y,X,Y=b;return [(x+X)/2,(y+Y)/2,log(X-x),log(Y-y)]
f=read(D/'frozen.json');out=[]
for r in read(P/'evidence.json')['rows']:
    vid=r['video'];t=r['t'];e=next(e for e in read(ROOT/'runs/bbox_confirmation_inventory_20261108/native'/f'{vid}.json')['events'] if e['frame']==t)
    tr,=[a for a in e['rows'] if a['old_tid']==r['old']];c,=[x for x in tr['candidates'] if x['new_tid']==r['new']]
    hh={x['frame']:val(x['box']) for x in tr['row_data']['history'][-4:]};ts=sorted(hh);last=ts[-1]
    v=[median((hh[b][i]-hh[a][i])/(b-a) for a,b in zip(ts,ts[1:])) for i in range(4)]
    actual=[];tracked=max((a for a in r['alternatives_birth'] if a['raw_now'] is not None),key=lambda a:a['birth_iou']);cross=[]
    with np.load(ROOT/f['videos'][vid]['raw']) as z:
        ff,cc,cl,boxes=[z[k] for k in ['frame_idx','conf','cls','xyxy']]
        for i,(u,b,idx) in enumerate([(t-1,c['birth']['box'],tracked['raw_birth']),(t,c['box'],tracked['raw_now'])]):
            x,y,w,h=[hh[last][k]+(u-last)*v[k] for k in range(4)];w=exp(w)/2;h=exp(h)/2;p=[x-w,y-h,x+w,y+h]
            own=ov(b,p);assert isclose(own,r['own_iou'][i],abs_tol=1e-9)
            ids=np.flatnonzero((ff==u)&(cc>=.1)&(cl==r['cls']));warn=[int(j) for j in ids if ov(boxes[j],b)<.5 and ov(boxes[j],p)>own]
            assert warn==([a['raw_birth'] for a in r['alternatives_birth']] if i==0 else [a['index'] for a in r['alternatives_now']])
            actual.append(bool(warn));cross.append(ov(b,boxes[idx]))
    out.append(dict(video=vid,key=[t,r['old'],r['new']],kind=r['kind'],candidate_vs_linked_alternative_IoU=cross,birth_and_now=all(actual)))
with (P/'audit.json').open('x',encoding='utf-8') as g:json.dump(dict(passed=True,source_sha256=hashlib.sha256((P/'evidence.json').read_bytes()).hexdigest(),rows=out),g,indent=2)
