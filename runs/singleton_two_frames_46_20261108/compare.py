"""Saved outputs; third policy only; verify each matched counterfactual by pair."""
import os,sys,json,hashlib,math,subprocess
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
P=Path(__file__).resolve().parent;ROOT=P.parents[1];D=P.parent/'singleton_birth_veto_transfer_20261108';INV=P.parent/'bbox_confirmation_inventory_20261108/native'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,o):
    with p.open('x',encoding='utf-8') as fp:json.dump(o,fp,indent=2,ensure_ascii=False,allow_nan=False)
def prep():
    manifest=j(D/'frozen.json');end=j(D/'complete.json')
    assert sha(D/'frozen.json')==end['frozen_sha256'] and set(end['outputs'])==set(manifest['videos']) and len(end['outputs'])==46
    files=[P/'protocol.md',P/'compare.py',P/'verify.py',D/'frozen.json',D/'complete.json',D/'five_cases/audit.json',D/'result.json']
    save(P/'before.json',dict(time=datetime.now().astimezone().isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files={p.relative_to(ROOT).as_posix():sha(p) for p in files}))
def run():
    import numpy as np
    sys.path.insert(0,str(ROOT));from tracking.native_confirm_identity import NativeConfirmedIdentity
    old=j(D/'frozen.json');end=j(D/'complete.json')
    def check():
        assert all(sha(ROOT/k)==s for k,s in j(P/'before.json')['files'].items())
        assert all(sha(ROOT/k)==s for k,s in old['inputs'].items())
        assert all(sha(D/'videos'/f'{v}.json')==s for v,s in end['outputs'].items())
    check()
    def geom(b):
        b=np.asarray(b,float);return np.r_[(b[:2]+b[2:])/2,np.log(b[2:]-b[:2])]
    def iou(a,b):
        a,b=np.asarray(a),np.asarray(b);i=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
        return float(i/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-i))
    out={}
    for v,meta in old['videos'].items():
        d=j(D/'videos'/f'{v}.json');n=j(INV/f'{v}.json');ev={e['frame']:e for e in n['events']};warnings={};edges=defaultdict(list);a=d['candidates']
        with np.load(ROOT/meta['raw']) as z:rf,rb,cl,cf=[z[k] for k in ('frame_idx','xyxy','cls','conf')]
        for k,r in a.items():
            if r['score'] is None or r['score']>1:continue
            t=r['t']; veto=False
            if r['veto']:
                row,=[x for x in ev[t]['rows'] if x['old_tid']==r['old']];c,=[x for x in row['candidates'] if x['new_tid']==r['new']]
                hist={o['frame']:geom(o['box']) for o in row['row_data']['history'][-4:]};ts=sorted(hist);last=ts[-1]
                vel=np.median([(hist[b]-hist[a])/(b-a) for a,b in zip(ts,ts[1:])],axis=0);pred=hist[last]+vel*(t-last);half=np.exp(pred[2:])/2;box=np.r_[pred[:2]-half,pred[:2]+half];det=np.asarray(c['box']);threshold=iou(det,box)
                ii=np.flatnonzero((rf==t)&(cf>=.1)&(cl==row['row_data']['cls']))
                alt=[int(i) for i in ii if iou(rb[i].astype(float),det)<.5 and iou(rb[i].astype(float),box)>threshold]
                warnings[k]=alt;veto=bool(alt)
            if not veto:
                score=r['score'];edges[t].append(dict(old_tid=r['old'],new_tid=r['new'],cost=1.6*score/(1+score),accepted=True))
        same= 'V' if all(warnings.values()) else ('H' if not any(warnings.values()) else None)
        if same:arm=d['arms'][same]
        else:
            with np.load(INV/f'{v}.npz') as z:o=z['outputs']
            offs=np.r_[0,np.bincount(o[:,0].astype(int),minlength=n['frames']).cumsum()]
            pol=NativeConfirmedIdentity(alpha=0);links={};rev={};active={}
            for t,(lo,hi) in enumerate(zip(offs,offs[1:])):
                raw=o[lo:hi,1:];changed,dec=pol.update(t,raw,ev.get(t,{}).get('newly_confirmed',[]),edges.get(t,[]))
                assert np.array_equal(changed[:,[0,1,2,3,5,6,7]],raw[:,[0,1,2,3,5,6,7]]) and len(set(changed[:,4]))==len(changed)
                for rr in dec['revoked']:
                    k=active.pop(rr['native_id']);assert links[k]==rr['old_canonical_id'];rev[k]=dict(frame=t,kept=rr['kept_native_id'])
                for rr in dec['links']:
                    k=f"{t}|{rr['old_tid']}|{rr['new_tid']}";active[rr['new_tid']]=k;links[k]=rr['canonical_id']
            arm=dict(links=links,revocations=rev)
        out[v]=dict(arm=arm,split=meta['split'],same_inputs_as=same,confirmation_alternatives_on_birth_warnings=warnings)
    check();save(P/'decisions.json',out)
if __name__=='__main__':
    if 'freeze' in sys.argv:prep()
    else:
        sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'));from system_memory_guard import guard_entrypoint
        guard_entrypoint(timeout=120);run()
