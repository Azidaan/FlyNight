"""Leave-one-video-out threshold diagnosis on fit20only; no tracker run."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_readout_calibration_20261108.common import *
SOURCES=['IMAGE','R1','DIS','FLYVIS']
ARMS=['SUPPORT',*[s+'_FIXED' for s in SOURCES],*[s+'_FIT' for s in SOURCES],'RANDOM_MATCHED','PERM_MATCHED']


def accept(value,threshold):return threshold!='off' and value is not None and value>=threshold


def threshold(rows,source):
    import numpy as np
    pos=[r['scores'][source] for r in rows if r['kind']=='positive' and r['scores'][source] is not None];neg=[r['scores'][source] for r in rows if r['kind']=='foreign' and r['scores'][source] is not None]
    if not pos or not neg:return 'off'
    t=float(np.nextafter(max(neg),np.inf));return t if max(pos)>=t else 'off'


def match_rate(rows,source,t):
    target=sum(accept(r['scores']['FLYVIS'],t) for r in rows);v=sorted((r['scores'][source] for r in rows if r['scores'][source] is not None),reverse=True)
    cutoff='off' if not target or not v else v[-1] if target>=len(v) else (v[target-1]+v[target])/2 if v[target-1]>v[target] else v[target-1]
    return cutoff,dict(target=target,actual=sum(accept(x,cutoff) for x in v),defined=len(v))


def load_rows():
    inventory=read(DATA/'inventory.json');out=[];labels={};scores={};sources={}
    for e in inventory['events']:
        video=e['video'];assert e['split']=='fit20'
        if video not in labels:
            lp=PRIOR_PILOT/'labels'/(video+'.json');sp=DATA/'scores'/(video+'.json');labels[video]=read(lp)['rows'];scores[video]=read(sp)['edges'];sources[lp.relative_to(ROOT).as_posix()]=sha(lp);sources[sp.relative_to(ROOT).as_posix()]=sha(sp)
        key=f"{e['frame']}|{e['old_tid']}|{e['new_tid']}";label=labels[video][key];kind='positive' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';out.append(dict(video=video,frame=e['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],kind=kind,scores=scores[video][key]['scores']))
    return out,sources


def main(obs):
    import numpy as np
    p=phase();assert read(DATA/'queue_result.json')['passed'];assert read(DATA/'queue_result.json')['data_only'];rows,inputs=load_rows();write(RUN/'execution_inputs.json',dict(registered=datetime.now(MSK).isoformat(),sources=inputs,data_features_sha256=sha(DATA/'features_complete.json'),data_scores_sha256=sha(DATA/'scores_complete.json')),new=True);videos=p['fit_videos'];folds={};per={a:{} for a in ARMS};ledgers=[]
    for video in videos:
        train=[r for r in rows if r['video']!=video];held=[r for r in rows if r['video']==video];fitted={s:threshold(train,s) for s in SOURCES};rt,rr=match_rate(train,'RANDOM',fitted['FLYVIS']);pt,pr=match_rate(train,'PERM',fitted['FLYVIS']);configs={'SUPPORT':dict(source='SUPPORT',threshold=-1)}
        for s in SOURCES:configs[s+'_FIXED']=dict(source=s,threshold=p['fixed_thresholds'][s]);configs[s+'_FIT']=dict(source=s,threshold=fitted[s])
        configs['RANDOM_MATCHED']=dict(source='RANDOM',threshold=rt);configs['PERM_MATCHED']=dict(source='PERM',threshold=pt);folds[video]=dict(thresholds=fitted,configs=configs,rate_controls=dict(RANDOM_MATCHED=rr,PERM_MATCHED=pr),train_videos=[v for v in videos if v!=video])
        for arm,cfg in configs.items():
            counts=Counter(N=sum(r['kind']!='unknown' for r in held));accepted=[]
            for r in held:
                yes=accept(r['scores'][cfg['source']],cfg['threshold'])
                if yes:counts[r['kind']]+=1;accepted.append(dict(frame=r['frame'],old_tid=r['old_tid'],new_tid=r['new_tid'],kind=r['kind']))
            counts['J']=counts['positive']-counts['foreign'];per[arm][video]=dict(counts);ledgers.append(dict(video=video,arm=arm,accepted=accepted))
        status('fit_only_leave_one_video_out',video=video,complete=len(folds),total=len(videos),guard=obs.read())
    totals={a:{k:sum(r.get(k,0) for r in per[a].values()) for k in set().union(*(r.keys() for r in per[a].values()))} for a in ARMS};ix=np.random.default_rng(42).integers(0,len(videos),(20000,len(videos)));comparisons={};pairs=[('FLYVIS_FIT',a) for a in ARMS if a!='FLYVIS_FIT']+[(s+'_FIT',s+'_FIXED') for s in ['IMAGE','R1','DIS']]
    for a,b in pairs:
        d=np.array([per[a][v]['J']-per[b][v]['J'] for v in videos]);n=np.array([per[a][v]['N'] for v in videos]);samples=100*d[ix].sum(1)/np.maximum(1,n[ix].sum(1));comparisons[a+'-'+b]=dict(delta_J=int(d.sum()),delta_per100=100*d.sum()/max(1,n.sum()),ci95=np.percentile(samples,[2.5,97.5]).tolist(),ci_corrected=np.percentile(samples,[2.5/20,100-2.5/20]).tolist())
    c=totals['FLYVIS_FIT'];qualified=c.get('positive',0)>=3 and c.get('foreign',0)==0 and comparisons['FLYVIS_FIT-FLYVIS_FIXED']['ci95'][0]>0
    for n,h in inputs.items():assert sha(ROOT/n)==h
    phase();write(RUN/'comparison_report.json',dict(passed=True,rows=rows,folds=folds,per_video=per,totals=totals,ledgers=ledgers,comparisons=comparisons,all_fit_thresholds={s:threshold(rows,s) for s in SOURCES},recalibration_hypothesis_gate=bool(qualified),no_26_or_20val_evaluation=True,no_tracker_detector_or_core_calls=True,scope='Readout classification of raw native-fragment proposals. This is neither public-ID tracking nor full-tracker acceptance. Unknown labels are excluded from class fitting but retained in intervention-rate matching.'),new=True)


if __name__=='__main__':main(guard())
