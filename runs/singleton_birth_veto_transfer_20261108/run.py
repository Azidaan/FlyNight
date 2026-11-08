"""Frozen paired public-ID comparison on saved frames. No detector or tracker."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,sys,json,math,subprocess
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
P=Path(__file__).resolve().parent; ROOT=P.parents[1]; sys.path.insert(0,str(ROOT))
INV=ROOT/'runs/bbox_confirmation_inventory_20261108'
OLD=ROOT/'runs/singleton_box_history_pairs_20261108'
L=ROOT/'runs/bbox_singleton_pilot_20261108/labels'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def rel(p):return p.relative_to(ROOT).as_posix()
def now():return datetime.now().astimezone().isoformat()
def freeze():
    inv=j(INV/'protocol.json');orig=set(j(ROOT/'runs/singleton_working_domain_20261108/protocol.json')['videos'])
    files=[P/'protocol.md',P/'run.py',P/'audit.py',INV/'protocol.json',ROOT/'runs/singleton_working_domain_20261108/protocol.json',OLD/'compare.py',OLD/'recovery1/launch.py',OLD/'conflict_review/inspect_pairs.py',ROOT/'tracking/native_confirm_identity.py',ROOT/'tracking/bbox_singleton_gate.py',ROOT/'.venv/Lib/site-packages/ultralytics/trackers/utils/matching.py'];vs={}
    for v,e in inv['videos'].items():
        if e['split'] not in ('fit20','evaluate26'):continue
        assert v not in orig
        receipt=INV/'native'/f'{v}.receipt.json';rec=j(receipt);assert rec['passed'] and rec['video']==v
        assert all(sha(ROOT/k)==s for k,s in rec['artifacts'].items())
        raw=ROOT/e['raw']; assert sha(raw)==e['sources'][e['raw']]
        files.extend([receipt,raw,L/f'{v}.json',*(ROOT/k for k in rec['artifacts'])]);vs[v]=dict(split=e['split'],raw=e['raw'])
    assert Counter(x['split'] for x in vs.values())=={'fit20':20,'evaluate26':26}
    save(P/'frozen.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),videos=vs,excluded_development=sorted(orig),inputs={rel(p):sha(p) for p in files}))
def run():
    import numpy as np
    from tracking.native_confirm_identity import NativeConfirmedIdentity
    from tracking.bbox_singleton_gate import proposals
    from ast import parse,Module,FunctionDef,fix_missing_locations
    # Execute only the original three pure geometry helpers, not the old analysis.
    tree=parse((OLD/'conflict_review/inspect_pairs.py').read_text(encoding='utf-8'))
    ns={'np':np};exec(compile(fix_missing_locations(Module(body=[x for x in tree.body if isinstance(x,FunctionDef) and x.name in ('iou','val','box')],type_ignores=[])),'frozen_geometry','exec'),ns)
    val,box,iou=[ns[k] for k in ('val','box','iou')]
    f=j(P/'frozen.json');assert all(sha(ROOT/k)==s for k,s in f['inputs'].items())
    for v,e in f['videos'].items():
        n=j(INV/'native'/f'{v}.json'); mapped={a['frame']:a for a in n['events']}
        with np.load(INV/'native'/f'{v}.npz') as z:o=z['outputs']
        with np.load(ROOT/e['raw']) as z:rf,rb,rc,rs=[z[k] for k in ('frame_idx','xyxy','cls','conf')]
        candidates={};edges=defaultdict(list)
        for t,event in mapped.items():
            for edge in proposals(event,cap=2,threshold=.2,bidirectional=False):
                old,new=edge['old_tid'],edge['new_tid'];k=f'{t}|{old}|{new}'
                row,=[r for r in event['rows'] if r['old_tid']==old];c,=[c for c in row['candidates'] if c['column']==edge['column']]; asserts=c['birth']['frame']==t-1 and c['new_tid']==new;assert asserts
                h=row['row_data']['history'][-4:];hist={a['frame']:val(a['box']) for a in h};ts=sorted(hist);assert all(np.array_equal(hist[a['frame']],val(a['box'])) for a in h) and ts[-1]<t-1
                assert k not in candidates
                cinfo=candidates[k]=dict(old=old,new=new,t=t,score=None,veto=False,alternatives=[])
                if len(ts)<2:continue
                last=ts[-1];vel=np.median([(hist[y]-hist[x])/(y-x) for x,y in zip(ts,ts[1:])],axis=0)
                ds=[]
                for at,b in [(t-1,c['birth']['box']),(t,c['box'])]:
                    pred=hist[last]+(at-last)*vel;ds.extend(np.abs(val(b)-pred)/[math.exp(pred[2])/2,math.exp(pred[3])/2,math.log(2),math.log(2)])
                d=float(max(ds));cinfo['score']=d
                if d>1:continue
                b=c['birth']['box'];pred=box(hist[last]+(t-1-last)*vel);own=iou(b,pred);ii=np.flatnonzero((rf==t-1)&(rc==row['row_data']['cls'])&(rs>=.1));alt=[int(i) for i in ii if iou(rb[i],b)<.5 and iou(rb[i],pred)>own]
                cinfo.update(veto=bool(alt),alternatives=alt);edges[t].append((dict(accepted=True,cost=1.6*d/(1+d),old_tid=old,new_tid=new),bool(alt)))
        offs=np.r_[0,np.bincount(o[:,0].astype(int),minlength=n['frames']).cumsum()];arms={}
        for arm in ('H','V'):
            pol=NativeConfirmedIdentity(alpha=0);links={}; rev={};active={}
            for t,(lo,hi) in enumerate(zip(offs,offs[1:])):
                raw=o[lo:hi,1:];result,dec=pol.update(t,raw,mapped.get(t,{}).get('newly_confirmed',[]),[edge for edge,flag in edges.get(t,[]) if arm=='H' or not flag])
                assert np.array_equal(result[:,[0,1,2,3,5,6,7]],raw[:,[0,1,2,3,5,6,7]]) and len(set(result[:,4]))==len(raw)
                for r in dec['revoked']:
                    k=active.pop(r['native_id']);assert links[k]==r['old_canonical_id'];rev[k]=dict(frame=t,kept=r['kept_native_id'])
                for r in dec['links']:
                    k=f"{t}|{r['old_tid']}|{r['new_tid']}";assert k not in links;links[k]=r['canonical_id'];active[r['new_tid']]=k
            arms[arm]=dict(links=links,revocations=rev)
        # Labels accessed only after both decisions.
        labels=j(L/f'{v}.json')['rows']
        for k,a in candidates.items():
            l=labels[k];assert [l[x] for x in ('video','frame','old_tid','new_tid')]==[v,a['t'],a['old'],a['new']]
            a['kind']='correct' if l['known'] and l['positive'] else 'foreign' if l['known'] else 'unknown'
        save(P/'videos'/f'{v}.json',dict(video=v,split=e['split'],candidates=candidates,arms=arms))
    assert all(sha(ROOT/k)==s for k,s in f['inputs'].items());save(P/'complete.json',dict(finished=now(),frozen_sha256=sha(P/'frozen.json'),outputs={v:sha(P/'videos'/f'{v}.json') for v in f['videos']}))
if __name__=='__main__':
    if 'freeze' in sys.argv:freeze()
    else:
        sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'));from system_memory_guard import guard_entrypoint
        guard_entrypoint(timeout=180)
        (P/'videos').mkdir(exist_ok=False);run()
