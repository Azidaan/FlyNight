"""Offline paired analysis; no tracker, dataset loader, or model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib,math
from statistics import median
from collections import Counter,defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=90)
from tracking.native_confirm_identity import NativeConfirmedIdentity
D=ROOT/'runs/singleton_working_domain_20261108'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
manifest=D/'scientific_checksums.json'
assert sha(manifest)=='55cd48a7d707d0bad322bcd9ce19f63ebc8de78d212851fa7d13ba94409e3250'
sums=read(manifest); used={}
def verified(p):
    k=p.relative_to(ROOT).as_posix();used[k]=sha(p);assert used[k]==sums[k];return p
def coords(b):
    x,y,X,Y=b
    return np.array([(X+x)/2,(Y+y)/2,math.log(X-x),math.log(Y-y)])
def dist(h,obs,mode):
    h=h[-4:]; last=coords(h[-1]['box']); t=h[-1]['frame']
    if mode=='H' and len(h)<2:return None
    v=np.median([(coords(b['box'])-coords(a['box']))/(b['frame']-a['frame']) for a,b in zip(h,h[1:])],axis=0) if mode=='H' else np.zeros(4)
    errors=[]
    for f,box in (obs[-1:] if mode=='S' else obs):
        pred=last+(f-t)*v; delta=np.abs(coords(box)-pred)
        errors.append((delta/[math.exp(pred[2])/2,math.exp(pred[3])/2,math.log(2),math.log(2)]).tolist())
    return {'score':max(map(max,errors)),'errors':errors}
rows=[]; revoked=Counter(); arms=('H','S','D')
for p in sorted((D/'outcomes').glob('*.json')):
    video=p.stem;o=read(verified(p)); n=read(verified(D/'native'/p.name)); ev={a['frame']:a for a in n['events']}
    with np.load(verified(D/'native'/f'{video}.npz'),allow_pickle=False) as z: native=z['outputs']
    def key(a):return(a['frame'],a['old_tid'],a['new_tid'])
    prior={key(a) for a in o['stages']['solver_selected']['rows']}
    assert prior=={key(a) for a in o['stages']['cost_qualified']['rows']}
    now={}; byframe=defaultdict(list)
    for a in o['stages']['singleton']['rows']:
        t,old,new=key(a);r,=[x for x in ev[t]['rows'] if x['old_tid']==old]
        c,=[x for x in r['candidates'] if x['forward_iou']>=.2];birth=c['birth'];hist=r['row_data']['history']
        assert c['new_tid']==new and hist[-1]['frame']<birth['frame']==t-1 and r['age']<=2
        assert (np.float32(c['cost']-.025)<=.8)==(key(a) in prior)
        row=dict(video=video,frame=t,old_tid=old,new_tid=new,kind=a['kind'],baseline=key(a) in prior,history_length=len(hist),scores={},selected={})
        for arm in arms:
            row['scores'][arm]=dist(hist,[(birth['frame'],birth['box']),(t,c['box'])],arm);row['selected'][arm]=False
        byframe[t].append(row);now[key(a)]=row;rows.append(row)
    assert len(now)==sum(map(len,byframe.values()))
    counts=np.bincount(native[:,0].astype(int),minlength=n['frames']);off=np.r_[0,counts.cumsum()]
    for arm in arms:
        policy=NativeConfirmedIdentity(alpha=0)
        for t,(lo,hi) in enumerate(zip(off,off[1:])):
            edges=[]
            for row in byframe.get(t,[]):
                metric=row['scores'][arm]
                if metric and metric['score']<=1:
                    d=metric['score'];edges.append(dict(old_tid=row['old_tid'],new_tid=row['new_tid'],accepted=True,cost=1.6*d/(1+d)))
            raw=native[lo:hi,1:];out,dec=policy.update(t,raw,ev.get(t,{}).get('newly_confirmed',[]),edges)
            assert np.array_equal(out[:,[0,1,2,3,5,6,7]],raw[:,[0,1,2,3,5,6,7]])
            for e in dec['links']:now[(t,e['old_tid'],e['new_tid'])]['selected'][arm]=True
            revoked[arm]+=len(dec['revoked'])
assert len(rows)==93 and len(set((r['video'],r['frame'],r['old_tid'],r['new_tid']) for r in rows))==93
assert Counter(r['kind'] for r in rows if r['baseline'])=={'correct':6,'foreign':1,'unknown':23}
def totals(rs):return dict(Counter(r['kind'] for r in rs))
res={}
for a in arms:
    ranked=sorted((r for r in rows if r['scores'][a]),key=lambda r:(r['scores'][a]['score'],r['video'],r['frame'],r['old_tid'],r['new_tid']))
    res[a]=dict(selected=totals(r for r in rows if r['selected'][a]),admitted=totals(r for r in ranked if r['scores'][a]['score']<=1),added=totals(r for r in rows if r['selected'][a] and not r['baseline']),lost=totals(r for r in rows if not r['selected'][a] and r['baseline']),top30=totals(ranked[:30]),revoked=revoked[a])
report=dict(baseline=totals(r for r in rows if r['baseline']),total=totals(rows),arms=res,source_head='d1519ce1d96679902f4e5ee4a23cdbb8dc530914',input_sha256=used,code_sha256=sha(Path(__file__)),policy_sha256=sha(ROOT/'tracking/native_confirm_identity.py'),protocol_sha256=sha(OUT/'protocol.md'))
assert all(sha(ROOT/k)==v for k,v in used.items())
save(OUT/'pairs.json',rows);report['pairs_sha256']=sha(OUT/'pairs.json');save(OUT/'result.json',report);print(json.dumps(res))
