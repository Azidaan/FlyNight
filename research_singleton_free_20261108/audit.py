"""Independent costs, native-lock invariants, labels, selection and paired intervals."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_singleton_free_20261108.common import *


def independent_labels(video,raw):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from datasets.bdd100k_loader import CLASS_NAMES
    annotation=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json')
    align=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches'];truth={}
    for a,b in zip(annotation,align):
        objects=[]
        for o in a.get('labels') or []:
            if o['category'] not in CLASS_NAMES.values() or (o.get('attributes') or {}).get('crowd',False):continue
            bb=o['box2d'];objects.append((str(o['id']),o['category'],[bb[k] for k in ['x1','y1','x2','y2']]))
        truth[int(b['video_frame_index'])]=objects
    cache={}
    def label(frame):
        if frame in cache:return cache[frame]
        allids=np.flatnonzero(raw['frame_idx']==frame);answer={};objects=truth.get(frame,[])
        for mask in [raw['conf'][allids]>=.25,(raw['conf'][allids]>.1)&(raw['conf'][allids]<.25)]:
            ids=allids[mask]
            for category in sorted(CLASS_NAMES.values()):
                di=[i for i in ids if CLASS_NAMES[int(raw['cls'][i])]==category];gg=[o for o in objects if o[1]==category]
                if not di or not gg:continue
                A=raw['xyxy'][di].astype(float);B=np.array([o[2] for o in gg],float)
                low=np.maximum(A[:,None,:2],B[None,:,:2]);high=np.minimum(A[:,None,2:],B[None,:,2:]);wh=np.maximum(0,high-low)
                overlap=wh.prod(-1);aa=np.maximum(0,A[:,2:]-A[:,:2]).prod(-1);bb=np.maximum(0,B[:,2:]-B[:,:2]).prod(-1)
                iou=overlap/np.maximum(aa[:,None]+bb[None,:]-overlap,1e-12);iou[iou<.5]=0
                ri,ci=linear_sum_assignment(-iou)
                for i,j in zip(ri,ci):
                    if iou[i,j]>0:answer[int(di[i])]=gg[j][0]
        cache[frame]=answer;return answer
    return label


def main(obs):
    import numpy as np
    from ultralytics.trackers.utils import matching
    phase('audit_phase.json');report=read(RUN/'comparison_report.json');selection=read(RUN/'selection.json');counts=Counter();ledgers={}
    for name,h in report['outputs'].items():assert sha(ROOT/name)==h
    for sources in report['source_reuse_manifests'].values():
        for name,h in sources.items():assert sha(ROOT/name)==h
    for split in ['fit20','evaluate26']:
        ledgers[split]={a:read(RUN/'outcomes'/(split+'_'+a+'.json')) for a in ARMS}
    exports=read(BASE/'export_complete.json')['videos']
    needed={r['video'] for value in ledgers['fit20'].values() for r in value['rows']}|{r['video'] for value in ledgers['evaluate26'].values() for r in value['rows']}
    for number,video in enumerate(sorted(needed)):
        item=exports[video];source=ROOT/item['path'];assert sha(source)==item['sha256'];data=read(source)
        assert sha(source.with_name(video+'.states.zip'))==data['states_sha256']
        score=read(BASE/'scores'/(video+'.json'));scenarios={}
        rawpath=ROOT/f'runs/train_cache/{video}/detections/raw_video.npz'
        with np.load(rawpath,allow_pickle=False) as z:raw={n:z[n] for n in z.files}
        labels=independent_labels(video,raw)
        with np.load(source.with_suffix('.npz'),allow_pickle=False) as matrices:
            for frame,snap in data['snapshots'].items():
                splits={q['split'] for q in snap['queries'] if q['split'] in ledgers}
                if not splits:continue
                assert len(splits)==1;split=splits.pop();C=matrices['cost_'+frame];plan=score['frames'][frame]
                native,rr,cc=matching.linear_assignment(C,thresh=.8);native=np.asarray(native,int).reshape(-1,2)
                assert native.tolist()==snap['native_matches'];rowset=set(map(int,rr));colset=set(map(int,cc))
                for arm,param in report['parameters'].items():
                    alpha=param['alpha'];beta=param['beta'];D=C.copy();allowed=[]
                    if arm=='U_THRESHOLD':
                        for i in rr:
                            for j in cc:
                                if snap['rows'][i]['state']==2 and snap['rows'][i]['cls']==snap['classes'][j] and C[i,j]<1:
                                    D[i,j]-=alpha;allowed.append((int(i),int(j)))
                    elif arm!='B0':
                        allcost=C.copy()
                        for i,j in plan['votes']:allcost[i,j]-=alpha
                        mapped='G' if arm=='U_GEO' else arm.replace('U_','G_',1)
                        for j,col in plan['columns'].items():
                            if not col['geometry_valid'] or len(col['rows'])<2:continue
                            ii=np.array(col['rows'],int)
                            if arm=='U_SHIFT':allcost[ii,int(j)]-=beta;continue
                            values=col['scores'].get(mapped)
                            if values is None or any(v is None for v in values):continue
                            values=np.clip(np.array(values,float),-1,1);allcost[ii,int(j)]-=beta*(values-values.mean())
                        for i,j in plan['votes']:
                            row=snap['rows'][i]
                            assert row['state']==2 and len(row['choices'])==2 and [jj for jj,g in zip(row['choices'],row['geometry']) if g['valid']]==[j]
                            if i in rowset and j in colset and row['cls']==snap['classes'][j] and C[i,j]<1:
                                D[i,j]=allcost[i,j];allowed.append((i,j))
                    masked=np.full((len(rr),len(cc)),np.inf,dtype=C.dtype);rmap={int(i):k for k,i in enumerate(rr)};cmap={int(j):k for k,j in enumerate(cc)}
                    for i,j in allowed:masked[rmap[i],cmap[j]]=D[i,j]
                    if alpha==0 and beta==0 or np.array_equal(D,C):added=np.empty((0,2),int)
                    else:
                        ij=np.asarray(matching.linear_assignment(masked,thresh=.8)[0],int).reshape(-1,2)
                        added=np.column_stack([np.asarray(rr)[ij[:,0]],np.asarray(cc)[ij[:,1]]]) if len(ij) else np.empty((0,2),int)
                    expected=sorted(map(tuple,np.concatenate([native,added],axis=0)))
                    if arm=='B0':
                        actual=snap['native_matches'];emitted={int(v[4]) for v in snap['native_output']};final={k:v for k,v in snap['native_final_assignments'].items() if int(k) in emitted}
                    else:
                        if arm not in scenarios:scenarios[arm]=read(RUN/'counterfactual'/video/f"{arm}_a{alpha:g}_b{beta:g}.json")
                        scenario=scenarios[arm]['frames'][frame];actual=scenario['first'];final=scenario['final']
                    assert sorted(map(tuple,actual))==expected and set(map(tuple,native)).issubset(set(map(tuple,actual)))
                    assert actual==sorted(actual)
                    for i,j in actual:
                        tid=str(snap['rows'][i]['tid'])
                        if tid in final:assert final[tid]==snap['raw_ids'][j]
                    for row in ledgers[split][arm]['rows']:
                        if row['video']!=video or row['frame']!=int(frame):continue
                        outcome=row['outcome'];raw_id=final.get(str(row['tid']));assert outcome['raw_id']==raw_id
                        identity=labels(int(frame)).get(raw_id)
                        assert outcome['identity']==identity
                        kind='none' if raw_id is None else ('unknown' if identity is None or outcome['target'] is None else ('correct' if identity==outcome['target'] else 'wrong'))
                        assert outcome['kind']==kind and outcome['J']=={'none':0,'unknown':0,'correct':1,'wrong':-1}[kind]
                        counts['independent_labeled_rows']+=1
                    counts['independent_matrices']+=1
        status('independent_first_problem_audit',complete=number+1,total=len(needed),checks=dict(counts),guard=obs.read())
    corrected={}
    recomputed={}
    for split,arms in ledgers.items():
        recomputed[split]={}
        for arm,ledger in arms.items():
            total=Counter();per_video=defaultdict(Counter)
            for row in ledger['rows']:
                if not row['primary_known']:continue
                o=row['outcome'];total[o['kind']]+=1;total['J']+=o['J'];total['N']+=1
                per_video[row['video']]['J']+=o['J'];per_video[row['video']]['N']+=1
            for key in ['J','N','correct','wrong','none','unknown']:assert total[key]==ledger['totals'].get(key,0)
            for video in ledger['per_video']:
                for key in ['J','N']:assert per_video[video][key]==ledger['per_video'][video].get(key,0)
            recomputed[split][arm]=per_video
    for key,value in report['comparisons'].items():
        a,b=key.split('-');A=ledgers['evaluate26'][a];B=ledgers['evaluate26'][b];videos=sorted(A['per_video'])
        AA=recomputed['evaluate26'][a];BB=recomputed['evaluate26'][b]
        d=np.array([AA[v]['J']-BB[v]['J'] for v in videos]);n=np.array([AA[v]['N'] for v in videos])
        ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));samples=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1))
        assert int(d.sum())==value['delta_J'] and np.allclose(np.percentile(samples,[2.5,97.5]),value['ci95'],rtol=0,atol=1e-12)
        corrected[key]=np.percentile(samples,[2.5/11,100-2.5/11]).tolist();counts['bootstrap_comparisons']+=1
    for arm,candidates in selection['candidates'].items():
        chosen=max(candidates,key=lambda r:(r['totals'].get('J',0),-r['totals']['changed_native_bindings'],-r['alpha'],-r['beta']))
        assert selection['parameters'][arm]=={k:chosen[k] for k in ['alpha','beta']}
    phase('audit_phase.json');write(RUN/'independent_audit.json',dict(passed=True,checks=dict(counts),comparison_sha256=sha(RUN/'comparison_report.json'),
        selection_sha256=sha(RUN/'selection.json'),ci99_545454545_family11=corrected,
        scope='Independent cost construction, all locked native pairs, saved final bindings, raw detection-to-GT labels and bootstrap. Historical one-step diagnostic, not full-tracker benefit.'),new=True)


if __name__=='__main__':main(guard())
