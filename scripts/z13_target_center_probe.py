"""Bounded CPU-only paired probe of the online target label; old sources stay untouched."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/z13_target_center_20261018'
MODES=('binary','center')
T0=time.monotonic()
LIMIT=1200


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def status(**kw):
    if time.monotonic()-T0>LIMIT or (RUN/'stop.request').exists():
        raise TimeoutError('CPU budget/stop requested; completed artifacts preserved')
    write(RUN/'status.json',dict(state='running',elapsed_seconds=round(time.monotonic()-T0),budget_seconds=LIMIT,**kw))


def paired_rows(torch,arm,bb,geom,eye_size,data,lambdas,stats,stage,max_batches=None):
    from scripts.z13_target_model import episode_features,center_and_presence
    from scripts.z13_train import success
    from tracking.z13_readout import N_EXEMPLAR,to_frame
    from tracking.z13_target_center import observed_center_target,ridge_scores
    by_ep={};vid=[]
    for v,n in data.meta: vid.extend([v]*n)
    for i,lab in enumerate(data.labels):
        if not lab[8]: by_ep.setdefault(lab[0],[]).append(i)
    eps=sorted(by_ep); rows={m:[] for m in MODES};frames=0; t0=time.monotonic()
    for left in range(0,len(eps),8):
        if max_batches is not None and left//8>=max_batches: break
        status(stage=f'{arm}: {stage}',done=left,total=len(eps),eta='unknown')
        ch=eps[left:left+8];mx=[max(data.labels[i][1] for i in by_ep[e]) for e in ch]
        hx,ms=data.batch(torch,'cpu',ch,mx)
        need={N_EXEMPLAR-1}|{N_EXEMPLAR+data.labels[i][1] for e in ch for i in by_ep[e]}
        f=episode_features(torch,arm,bb,hx,geom.neighbors,need)
        frames+=len(ch)*hx.shape[1]
        for j,e in enumerate(ch):
            xe=(f[N_EXEMPLAR-1][j].T-stats[0])/stats[1]
            idx=by_ep[e];xs=np.stack([(f[N_EXEMPLAR+data.labels[i][1]][j].T-stats[0])/stats[1] for i in idx])
            mask=ms[j].numpy()
            targets={'binary':mask.astype(float),'center':observed_center_target(mask,geom.points,geom.spacing)}
            for mode in MODES:
                scores=ridge_scores(xe,targets[mode],xs,lambdas[mode])
                for si,i in enumerate(idx):
                    lab=data.labels[i]; vals=[]
                    for li in range(len(lambdas[mode])):
                        ce,pres=center_and_presence(scores[li,si],geom)
                        center=np.asarray(to_frame(ce,lab[7],eye_size))
                        x=dict(p_present=pres,center=center.tolist(),pred=np.asarray(lab[6],float).tolist())
                        if lab[2]==1:
                            box=np.asarray(lab[5]);x.update(err=float(np.linalg.norm(center-lab[4])),err_pred=float(np.linalg.norm(np.asarray(lab[6])-lab[4])),scale=float(max(box[2]-box[0],box[3]-box[1])))
                            x['useful']=bool(success(dict(kind=1,**x)))
                        elif lab[2]==0: x['useful']=False
                        vals.append(x)
                    rows[mode].append(dict(label=i,video=vid[e],kind=int(lab[2]),pos=int(lab[1]),by_lambda=vals))
    return rows,dict(seconds=time.monotonic()-t0,frames=frames,episodes=min(len(eps),8*max_batches) if max_batches else len(eps))


def summarize(rows):
    from scripts.z13_continue import per_video_useful,coverage_table
    pos=[r for r in rows if r['kind']==1]
    return dict(rows=len(rows),positives=len(pos),useful=sum(r['useful'] for r in pos),
                useful_mean_video=float(np.mean(list(per_video_useful(rows).values()))),
                median_error_rel=float(np.median([r['err']/r['scale'] for r in pos])),coverage=coverage_table(rows))


def verify_sources(proto):
    for rel,digest in proto['sha256'].items():
        if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest: raise RuntimeError('Changed source: '+rel)


def main():
    global LIMIT
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');ap.add_argument('--budget',type=int,default=1200)
    a=ap.parse_args();LIMIT=a.budget
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from scripts.z13_continue import paired,per_video_useful
    p=json.loads((RUN/'protocol.json').read_text(encoding='utf-8'));verify_sources(p)
    if not a.benchmark and (RUN/'report.json').exists(): raise RuntimeError('Completed probe must not be repeated')
    source=json.loads((ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json').read_text())['config']['job']
    fit=Episodes(source['fit'][:20]);ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    status(stage='CPU build',done=0,total=3,eta='unknown')
    if a.benchmark:
        bank,geom,bb=old.build_arm('D',torch,ck)
        if str(bb.dev)!='cpu':raise RuntimeError('CPU required')
        _,timing=paired_rows(torch,'D',bb,geom,bank.eye_size,fit,{m:old.LAMBDAS for m in MODES},(np.zeros(68),np.ones(68)),'timing only',max_batches=2)
        # Measure padded integration work, not number of labels. All existing arrays only.
        costs={}
        for name,vids in [('fit20',source['fit'][:20]),('train26',source['predict'])]:
            maxima=[]
            for v in vids:
                with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
                    active={}
                    for e,pos in zip(z['lab_ep'],z['lab_pos']):
                        if not z['ep_pert'][e]: active[int(e)]=max(active.get(int(e),-1),int(pos))
                    maxima.extend(active[e]+5 for e in sorted(active))
            costs[name]=sum(len(maxima[i:i+8])*max(maxima[i:i+8]) for i in range(0,len(maxima),8))
        estimate=timing['seconds']/timing['frames']*(sum(costs.values())+200*5)*3
        write(RUN/'benchmark.json',dict(timing=timing,padded_frame_cost=costs,rough_total_seconds=estimate,caveat='Two batches, unnormalized timing only; factor three conservatively includes R1 as FlyVis; load time excluded. Not a scientific result.'))
        write(RUN/'status.json',dict(state='completed',stage='CPU timing completed',done=1,total=1,elapsed_seconds=round(time.monotonic()-T0)))
        return
    evaluation=Episodes(source['predict']);allrows={};choices={}
    for arm in p['arms']:
        directory=RUN/arm
        if (directory/'done.json').exists():
            for m in MODES:allrows[arm+'_'+m]=json.loads((directory/(m+'_rows.json')).read_text())
            choices[arm]=json.loads((directory/'choice.json').read_text());continue
        status(stage=arm+': feature normalization',done=0,total=200,eta='unknown')
        bank,geom,bb=old.build_arm(arm,torch,ck)
        if bb is not None and str(bb.dev)!='cpu':raise RuntimeError('CPU required')
        stats=old.feature_stats(torch,arm,bb,geom,fit)
        fr,_=paired_rows(torch,arm,bb,geom,bank.eye_size,fit,{m:old.LAMBDAS for m in MODES},stats,'train20 lambda selection')
        choice={}; lambdas={}
        for m in MODES:
            scores=[float(np.mean([r['by_lambda'][l]['useful'] for r in fr[m] if r['kind']==1])) for l in range(len(old.LAMBDAS))]
            idx=int(np.argmax(scores));lambdas[m]=[old.LAMBDAS[idx]];choice[m]=dict(lambda_value=lambdas[m][0],fit_scores=scores)
        er,timing=paired_rows(torch,arm,bb,geom,bank.eye_size,evaluation,lambdas,stats,'fixed train26 evaluation')
        for m in MODES:
            out=old.useful_rows(er[m],0);write(directory/(m+'_rows.json'),out);allrows[arm+'_'+m]=out
        reference=ROOT/('runs/z13_target_model_r1_20261017' if arm=='R1' else 'runs/z13_target_model_20261017')/arm
        saved={r['label']:r for r in json.loads((reference/'rows.json').read_text())}
        current=allrows[arm+'_binary']
        if len(saved)!=len(current) or set(saved)!={r['label'] for r in current}:raise RuntimeError('Control row mismatch')
        dc=max(float(np.max(np.abs(np.asarray(r['center'])-saved[r['label']]['center']))) for r in current)
        dp=max(abs(r['p_present']-saved[r['label']]['p_present']) for r in current)
        old_lambda=json.loads((reference/'choice.json').read_text())['lambda']
        reproduction=dict(max_center_px=dc,max_score_difference=dp,lambda_matches=choice['binary']['lambda_value']==old_lambda)
        write(directory/'reproduction.json',reproduction)
        if dc>=.5 or dp>=.001 or not reproduction['lambda_matches']:raise RuntimeError('Original binary control not reproduced')
        write(directory/'choice.json',choice);choices[arm]=choice
        write(directory/'done.json',dict(timing=timing,source_verified=True))
        del bb,bank
    prior=json.loads((ROOT/'runs/z13_continue_20261017/baseline/rows.json').read_text());allrows['D_prior']=prior
    keys=[{(r['label'],r['video'],r['kind']) for r in rows} for rows in allrows.values()]
    if not all(k==keys[0] for k in keys):raise RuntimeError('Pairing mismatch')
    pv={k:per_video_useful(v) for k,v in allrows.items()}
    pairs=[(arm+'_center',arm+'_binary') for arm in p['arms']]+[('D_center','D_prior'),('D_center','R1_center'),('C_center','R1_center'),('D_center','C_center')]
    rep=dict(summary={k:summarize(v) for k,v in allrows.items()},paired={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs},choices=choices,
             limitations='Exploratory reused train26; one fixed target shape; no HOTA; response-peak z scores are not calibrated presence probabilities.')
    d=rep['paired']['D_center-D_prior'];ctrl=rep['paired']['D_center-D_binary']
    rep['primary_pass']=d['ci95'][0]>0 and ctrl['ci95'][0]>0
    rep['decision']='candidate_for_separate_validation' if rep['primary_pass'] else 'do_not_replace_prior_readout'
    verify_sources(p);write(RUN/'report.json',rep)
    write(RUN/'status.json',dict(state='completed',stage='report written',done=3,total=3,elapsed_seconds=round(time.monotonic()-T0)))


if __name__=='__main__':
    try:main()
    except Exception as e:
        write(RUN/'status.json',dict(state='paused' if isinstance(e,TimeoutError) else 'failed',stage=str(e),done=0,total=0,elapsed_seconds=round(time.monotonic()-T0)))
        raise
