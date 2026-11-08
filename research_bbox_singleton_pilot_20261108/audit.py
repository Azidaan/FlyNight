"""Independent ordinary-box arithmetic, native parity, causal labels, LAP and CIs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_singleton_pilot_20261108.common import *


def overlap(a,b):
    import numpy as np
    a=np.asarray(a,float);b=np.asarray(b,float);left=np.maximum(a[:2],b[:2]);right=np.minimum(a[2:],b[2:]);inter=float(np.prod(np.maximum(0,right-left)));area=float(np.prod(np.maximum(0,a[2:]-a[:2]))+np.prod(np.maximum(0,b[2:]-b[:2]))-inter)
    return inter/area if area>0 else 0.


def independently_proposed(event,cfg):
    import numpy as np
    edges=[]
    for r in event['rows']:
        if r['age']>cfg['cap']:continue
        possible=[c for c in r['candidates'] if c['forward_iou']>=.2]
        if len(possible)!=1 or possible[0]['new_tid'] is None:continue
        c=possible[0]
        if cfg['bidirectional']:
            now=np.array(c['box'],float);before=np.array(c['birth']['box'],float);assert event['frame']-c['birth']['frame']==1
            delta=(now[2:]+now[:2]-before[2:]-before[:2])/2;back=now.copy();back[:2]-=delta*r['age'];back[2:]-=delta*r['age']
            if overlap(back,r['row_data']['history'][-1]['box'])<.2:continue
        edges.append(dict(old_tid=r['old_tid'],new_tid=c['new_tid'],cost=c['cost']))
    return edges


def replay(native,data,cfg,ledger):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    aliases={};owners={};seen=set();logs={r['frame']:r for r in ledger};events={e['frame']:e for e in data['events']};result=native.copy();checks=Counter();counts=np.bincount(native[:,0].astype(int),minlength=data['frames']);offset=np.r_[0,counts.cumsum()]
    for f in range(data['frames']):
        lo,hi=offset[f:f+2];ids=list(map(int,native[lo:hi,5]));root=lambda t:aliases.get(t,t);event=events.get(f,dict(frame=f,rows=[],newly_confirmed=[]));d=logs.get(f,dict(links=[],revoked=[],allowed=[]))
        for r in d['revoked']:
            assert r['native_id'] in ids and r['kept_native_id'] in ids and aliases[r['native_id']]==r['old_canonical_id']==root(r['kept_native_id']);del aliases[r['native_id']]
        current={root(t) for t in ids};fresh=set(event['newly_confirmed']) & set(ids)-seen;eligible={}
        for e in independently_proposed(event,cfg) if cfg else []:
            old,new=e['old_tid'],e['new_tid'];canonical=root(old)
            if new not in fresh or old not in seen or old in ids or owners.get(canonical)!=old or canonical in current or e['cost']>=1:continue
            eligible[(old,new)]=dict(old_tid=old,new_tid=new,canonical_id=canonical,cost=e['cost'],corrected_cost=e['cost']-.2)
        assert sorted(eligible)==sorted((e['old_tid'],e['new_tid']) for e in d['allowed'])
        rr=sorted({a for a,b in eligible});cc=sorted({b for a,b in eligible});R,C=len(rr),len(cc)
        if R and C:
            matrix=np.full((R+C,R+C),np.inf);matrix[R:,C:]=0
            for i in range(R):matrix[i,C+i]=.4
            for j in range(C):matrix[R+j,j]=.4
            for (a,b),e in eligible.items():matrix[rr.index(a),cc.index(b)]=np.float32(e['corrected_cost'])
            ii,jj=linear_sum_assignment(matrix);actual=sum(float(np.float32(e['corrected_cost'])) for e in d['links'])+.4*(R+C-2*len(d['links']));assert abs(actual-matrix[ii,jj].sum())<1e-5;checks['independent_optional_assignments']+=1
        for e in d['links']:assert e==eligible[(e['old_tid'],e['new_tid'])];aliases[e['new_tid']]=e['canonical_id']
        canonical=[root(t) for t in ids];assert len(canonical)==len(set(canonical));result[lo:hi,5]=canonical
        for t in ids:owners[root(t)]=t
        seen.update(ids);checks['causal_frames']+=1
    return result,checks


def main(obs):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from datasets.bdd100k_loader import CLASS_NAMES
    p=phase();report=read(RUN/'comparison_report.json');inventory=read(INVENTORY/'protocol.json');checks=Counter();totals=defaultdict(Counter)
    for video,e in inventory['videos'].items():
        data=read(INVENTORY/'native'/(video+'.json'))
        with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/e['baseline'],allow_pickle=False) as z:assert np.array_equal(native[:,:8],z['outputs']) and data['frames']==int(z['frame_count'])
        for event in data['events']:
            for row in event['rows']:
                mean=np.array(row['row_data']['mean'][:4]);pred=np.r_[mean[:2]-mean[2:]/2,mean[:2]+mean[2:]/2]
                for c in row['candidates']:
                    value=overlap(pred,c['box']);assert abs(value-c['forward_iou'])<2e-5,(video,event['frame'],value,c['forward_iou']);assert abs(c['cost']-(1-value*c['confidence']))<2e-6;checks['independent_forward_iou']+=1
                    if c['new_tid'] is not None:assert c['birth']['frame']==event['frame']-1 and c['new_tid'] in event['newly_confirmed']
        checks['complete_native_baseline_videos']+=1
        if e['split']=='val20':continue
        labels=read(RUN/'labels'/(video+'.json'));annotations=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json');align=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches'];gt={int(a['video_frame_index']):fr for fr,a in zip(annotations,align)}
        for frame in labels['new_needed_frame_assignments']:
            f=frame['frame'];out=native[native[:,0]==f];wanted={int(row[5]):None for row in out};objects=[l for l in gt[f].get('labels') or [] if l['category'] in CLASS_NAMES.values() and not (l.get('attributes') or {}).get('crowd',False)]
            for category in CLASS_NAMES.values():
                rows=[r for r in out if CLASS_NAMES[int(r[7])]==category];objs=[l for l in objects if l['category']==category]
                if not rows or not objs:continue
                scores=np.array([[overlap(r[1:5],[l['box2d'][k] for k in ['x1','y1','x2','y2']]) for l in objs] for r in rows]);scores[scores<.5]=0;ii,jj=linear_sum_assignment(-scores)
                for i,j in zip(ii,jj):
                    if scores[i,j]>0:wanted[int(rows[i][5])]=str(objs[j]['id'])
            assert wanted==dict(frame['matches']);checks['independent_new_label_frames']+=1
        for l in labels['rows'].values():
            old={g for f,g in l['target_anchors']};new={g for f,g in l['fragment_anchors']};known=len(old)==len(new)==1;assert known==l['known']
            if known:assert l['positive']==(next(iter(old))==next(iter(new)))
        for arm,cfg in [('B0',None),*p['configs'].items()]:
            value=read(RUN/'outcomes'/arm/(video+'.json'));out,c=replay(native,data,cfg,value['decisions']);checks.update(c);assert hashlib.sha256(out.tobytes()).hexdigest()==value['outputs_sha256'];computed=Counter(N=sum(l['known'] for l in labels['rows'].values()))
            for d in value['decisions']:computed['revoked']+=len(d['revoked'])
            for link in value['links']:computed['links']+=1;computed[link['kind']]+=1
            computed['J']=computed['correct']-computed['foreign'];assert dict(computed)==value['totals'];assert value['totals']==report['results'][e['split']]['per_video'][video][arm];checks['outcome_ledgers']+=1
        status('bbox_pilot_independent_audit',video=video,checks=dict(checks),guard=obs.read())
    selection=read(RUN/'selection.json');assert sha(RUN/'selection.json')==report['selection_sha256'];fit=report['results']['fit20']['totals'];eligible=[a for a in p['configs'] if fit[a].get('foreign',0)==0];chosen=max(eligible,key=lambda a:(fit[a].get('correct',0),-p['configs'][a]['cap'],-int(p['configs'][a]['bidirectional']))) if eligible else 'B0';assert chosen==selection['selected']==report['selected']
    names=list(report['results']['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)))
    for arm in p['configs']:
        rows=[report['results']['evaluate26']['per_video'][v][arm] for v in names];d=np.array([r['J'] for r in rows]);n=np.array([r['N'] for r in rows]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));value=report['comparisons'][arm+'-B0'];assert value['delta_J']==int(d.sum());assert np.allclose(value['ci95'],np.percentile(sample,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(sample,[2.5/12,100-2.5/12]),atol=1e-12,rtol=0);checks['bootstrap_intervals']+=1
    phase();write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),scope='Independent geometric arithmetic,66complete native baselines, new actual-anchor label assignments, causal canonical IDs, SciPy optional LAP objective, selected configuration and bootstrap. No full benefit or FlyVis contribution claim.'),new=True)


if __name__=='__main__':main(guard())
