"""Frozen CPU validation on dev35; per-video persistence, no dev parameter selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/z13_target_center_dev_20261018'
TRAIN=ROOT/'runs/z13_target_center_20261018'
DEADLINE=None
CONTEXT={}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def write(path,value):
    from tracking.z13_checkpoint import atomic_json
    atomic_json(path,value)


def check():
    if (RUN/'stop.request').exists() or (DEADLINE and time.time()>=DEADLINE):
        raise TimeoutError('Deadline/stop request; completed videos retained')


def status(**values):
    check()
    stage=values.pop('stage',CONTEXT.get('stage','starting'))
    # The frozen spatial evaluator uses different progress field names.
    if 'episodes_done' in values: values['done']=values.pop('episodes_done')
    if 'episodes_total' in values: values['total']=values.pop('episodes_total')
    write(RUN/'status.json',{**CONTEXT,**values,'state':'running','updated':datetime.now(timezone.utc).isoformat(),
        'stage':stage,'eta':'unknown'})


def force_cpu():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    os.environ['TIMEZONE']='Europe/Moscow'
    import torch
    torch.cuda.is_available=lambda:False
    torch.set_default_device('cpu');torch.set_num_threads(2)
    assert torch.empty(1).device.type=='cpu' and not torch.cuda.is_initialized()
    return torch


def verify(protocol):
    for rel,digest in protocol['sha256'].items():
        if sha(ROOT/rel)!=digest: raise RuntimeError('Frozen source changed: '+rel)


def frozen_lambdas(arm,choices):
    return {mode:[choices[arm][mode]['lambda_value']] for mode in ('binary','center')}


def canonical(rows,video):
    return [dict(r,label=f'{video}:{r["label"]}',video=video) for r in rows]


def validate_video(video,path,geom,eye_size):
    from scripts.z13_data import raw_index,label_frame
    from scripts.output_filter_oracle import VALID
    from tracking.z13_readout import to_eye,to_frame,N_EXEMPLAR
    alignment=read(ROOT/f'runs/train_cache/{video}/alignment.json')['matches']
    truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json')
    raw=raw_index(truth,[int(x['video_frame_index']) for x in alignment],VALID)
    annotations=sorted(raw); adjacent={f:(annotations[i-1] if i else None,annotations[i+1] if i+1<len(annotations) else None) for i,f in enumerate(annotations)}
    report=dict(video=video,labels=0,positives=0,label_mismatch=0,empty_masks=0,nonconsecutive_search=0,max_eye_error=0.)
    with np.load(path,allow_pickle=False) as d:
        off=d['ep_offset'];frames=d['fr_frame'];report['empty_masks']=int((~d['ex_mask'].any(1)).sum())
        assert len(off)-1==len(d['ex_mask']) and int(off[-1])==len(d['fr_hex'])
        for e in range(len(off)-1):
            search=frames[off[e]+N_EXEMPLAR:off[e+1]]
            report['nonconsecutive_search']+=int(len(search)>1 and np.any(np.diff(search)!=1))
        for i,kind_saved in enumerate(d['lab_kind']):
            e=int(d['lab_ep'][i]);pos=int(d['lab_pos'][i]);assert 0<=e<len(off)-1 and 0<=pos<off[e+1]-off[e]-N_EXEMPLAR
            f=int(frames[off[e]+N_EXEMPLAR+pos]);crop=tuple(int(x) for x in d['lab_crop'][i])
            identity=str(d['ep_identity'][e]) or None;a,b=adjacent[f]
            around=identity is not None and a is not None and b is not None and any(x[0]==identity for x in raw[a]) and any(x[0]==identity for x in raw[b])
            kind,col,center,_=label_frame(identity,int(d['ep_id_dist'][e]),raw[f],crop,geom,eye_size,around)
            report['labels']+=1
            report['label_mismatch']+=int(kind!=int(kind_saved) or col!=int(d['lab_col'][i]))
            if kind==1:
                report['positives']+=1;point=np.asarray(to_eye(center,crop,eye_size))
                assert np.allclose(to_frame(point,crop,eye_size),center,atol=1e-3)
                report['max_eye_error']=max(report['max_eye_error'],float(np.linalg.norm(geom.points[col]-point)))
    report['passed']=report['label_mismatch']==report['empty_masks']==report['nonconsecutive_search']==0 and report['max_eye_error']<geom.spacing
    assert report['passed'],report
    return report


def extract_one(video,index,total):
    from scripts.z13_data import extract
    import cv2
    output=RUN/'data';output.mkdir(exist_ok=True)
    target=output/(video+'.npz');counts=output/(video+'.json')
    if target.exists() and counts.exists(): return
    # Preserve any partial output of this run rather than trusting or deleting it.
    for p in (target,counts):
        if p.exists():
            p.resolve().relative_to(RUN.resolve())
            p.rename(p.with_name(p.name+'.partial.'+str(time.time_ns())))
    with np.load(ROOT/f'runs/fly_signatures/{video}/signatures.npz') as z: expected=int(z['frame_count'])
    factory=cv2.VideoCapture;decoded=[0]
    class CheckedCapture:
        def __init__(self,*args,**kwargs): self.cap=factory(*args,**kwargs)
        def read(self):
            check();ok,frame=self.cap.read()
            if not ok and decoded[0]<expected: raise RuntimeError('Video decode ended early: '+video)
            decoded[0]+=int(ok)
            if decoded[0]%100==0: status(stage='extract episodes',done=decoded[0],total=expected,videos_done=index,videos_total=total)
            return ok,frame
        def release(self): return self.cap.release()
    cv2.VideoCapture=CheckedCapture
    try:
        extract((video,str(ROOT/'runs/train_cache'),'train',str(output),str(ROOT/'data/bdd100k/videos/train')))
        assert decoded[0]==expected,(decoded[0],expected)
    finally: cv2.VideoCapture=factory


def prior_rows(torch,data,bb,ro,geom,eye_size,video,max_batches=None):
    from tracking.z13_readout import run_sequence,center_estimate,to_frame
    from scripts.z13_train import success
    episodes={}
    for i,label in enumerate(data.labels):
        if not label[8]: episodes.setdefault(label[0],[]).append(i)
    eps=sorted(episodes);rows=[];ro.eval()
    with torch.no_grad():
        for left in range(0,len(eps),4):
            if max_batches is not None and left//4>=max_batches: break
            status(stage='D prior readout',done=left,total=len(eps))
            batch=eps[left:left+4]
            hx,ms=data.batch(torch,'cpu',batch,[max(data.labels[i][1] for i in episodes[e]) for e in batch])
            probs=run_sequence(torch,bb,ro,hx,ms).softmax(-1).cpu().numpy()
            for j,e in enumerate(batch):
                for i in episodes[e]:
                    label=data.labels[i];p=probs[j,label[1]];center=np.asarray(to_frame(center_estimate(p[:721],geom),label[7],eye_size))
                    row=dict(label=i,video=video,kind=int(label[2]),pos=int(label[1]),p_present=1-float(p[721]),center=center.tolist(),useful=False)
                    if label[2]==1:
                        box=np.asarray(label[5]);row.update(err=float(np.linalg.norm(center-label[4])),err_pred=float(np.linalg.norm(np.asarray(label[6])-label[4])),scale=float(max(box[2]-box[0],box[3]-box[1])))
                        row['useful']=bool(success(row))
                    rows.append(row)
    return rows


def technical(torch,protocol,fit,ck):
    from scripts import z13_target_model as old,z13_target_center_probe as probe,z13_train as train
    from tracking.z13_spatial import make_spatial_readout
    video=protocol['technical_train_video'];data=train.Episodes([video])
    bank,geom,bb=old.build_arm('D',torch,ck)
    stats=old.feature_stats(torch,'D',bb,geom,fit)
    result,_=probe.paired_rows(torch,'D',bb,geom,bank.eye_size,data,frozen_lambdas('D',protocol['choices']),stats,'technical saved train control',max_batches=1)
    report={}
    for mode in ('binary','center'):
        reference={x['label']:x for x in read(TRAIN/'D'/(mode+'_rows.json')) if x['video']==video}
        current=old.useful_rows(result[mode],0)
        report[mode]=dict(rows=len(current),max_center_px=max(float(np.max(np.abs(np.asarray(x['center'])-reference[x['label']]['center']))) for x in current),max_score_difference=max(abs(x['p_present']-reference[x['label']]['p_present']) for x in current))
        assert report[mode]['max_center_px']<.5 and report[mode]['max_score_difference']<.001,report
    ro=make_spatial_readout(torch)(34,geom.points,bank.eye_size,geom.neighbors);ro.load_state_dict(ck['readout'])
    current=prior_rows(torch,data,bb,ro,geom,bank.eye_size,video,max_batches=1)
    reference={x['label']:x for x in read(ROOT/'runs/z13_continue_20261017/baseline/rows.json') if x['video']==video}
    report['prior']=dict(rows=len(current),max_center_px=max(float(np.max(np.abs(np.asarray(x['center'])-reference[x['label']]['center']))) for x in current),max_score_difference=max(abs(x['p_present']-reference[x['label']]['p_present']) for x in current))
    assert report['prior']['max_center_px']<.5 and report['prior']['max_score_difference']<.001,report
    report['passed']=True;write(RUN/'technical.json',report)


def evaluate(torch,protocol,fit,ck):
    from scripts import z13_target_model as old,z13_target_center_probe as probe,z13_train as train
    from tracking.z13_spatial import make_spatial_readout
    for arm in ('D','C','R1'):
        CONTEXT.update(variant=arm,variants_total=3)
        if all((RUN/arm/v/'done.json').exists() for v in protocol['videos']):continue
        bank,geom,bb=old.build_arm(arm,torch,ck)
        # Only frozen training videos enter normalization; no dev refitting.
        stats_path=RUN/arm/'normalization.json'
        if stats_path.exists():
            st=read(stats_path);stats=np.asarray(st['mean']),np.asarray(st['sd'])
        else:
            status(stage='training-only normalization',done=0,total=0)
            stats=old.feature_stats(torch,arm,bb,geom,fit)
            write(stats_path,dict(mean=stats[0].tolist(),sd=stats[1].tolist(),fit=protocol['fit']))
        ro=None
        if arm=='D':
            ro=make_spatial_readout(torch)(34,geom.points,bank.eye_size,geom.neighbors);ro.load_state_dict(ck['readout'])
        for k,video in enumerate(protocol['videos']):
            folder=RUN/arm/video
            if (folder/'done.json').exists():continue
            CONTEXT.update(video=video,videos_done=k,videos_total=len(protocol['videos']))
            train.DATA=RUN/'data';data=train.Episodes([video])
            result,timing=probe.paired_rows(torch,arm,bb,geom,bank.eye_size,data,frozen_lambdas(arm,protocol['choices']),stats,'frozen dev evaluation')
            for mode in ('binary','center'):
                write(folder/(mode+'_rows.json'),canonical(old.useful_rows(result[mode],0),video))
            if ro is not None:
                write(folder/'prior_rows.json',canonical(prior_rows(torch,data,bb,ro,geom,bank.eye_size,video),video))
            write(folder/'done.json',dict(timing=timing,video=video,arm=arm,source_sha256=sha(RUN/'data'/(video+'.npz'))))
        del bb,bank,ro


def summarize(protocol):
    from scripts.z13_continue import per_video_useful,paired
    from scripts.z13_target_center_probe import summarize as summary
    rows={}
    for arm in ('D','C','R1'):
        for mode in ('binary','center'):
            rows[arm+'_'+mode]=[r for v in protocol['videos'] for r in read(RUN/arm/v/(mode+'_rows.json'))]
    rows['D_prior']=[r for v in protocol['videos'] for r in read(RUN/'D'/v/'prior_rows.json')]
    keys=[{(r['label'],r['video'],r['kind']) for r in values} for values in rows.values()]
    assert all(k==keys[0] for k in keys) and all(len(x)==len(keys[0]) for x in rows.values())
    pv={k:per_video_useful(v) for k,v in rows.items()}
    comparisons=protocol['comparisons']
    rep=dict(summary={k:summary(v) for k,v in rows.items()},paired={a+'-'+b:paired(pv[a],pv[b]) for a,b in comparisons},
        limitations='Fixed parameters; dev35 used earlier in this project; no HOTA or heldout75. harmful denotes known nonuseful, not measured tracking harm.')
    rep['primary_pass']=all(rep['paired'][k]['ci95'][0]>0 for k in protocol['primary'])
    rep['fly_control_pass']=rep['paired']['D_center-R1_center']['ci95'][0]>0
    rep['decision']='localization_transfers; intervention reliability still requires a separate gate' if rep['primary_pass'] and rep['fly_control_pass'] else 'transfer_not_confirmed; no dev retuning'
    for name,values in rows.items():write(RUN/'aggregate'/(name+'_rows.json'),values)
    verify(protocol);write(RUN/'report.json',rep)


def main():
    global DEADLINE
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',type=float,required=True);args=ap.parse_args();DEADLINE=args.deadline
    protocol=read(RUN/'protocol.json');verify(protocol);torch=force_cpu()
    from scripts import z13_train as train,z13_target_center_probe as probe
    probe.status=status
    fit=train.Episodes(protocol['fit'])
    ck=torch.load(ROOT/protocol['model'],map_location='cpu',weights_only=False)
    if not (RUN/'technical.json').exists(): technical(torch,protocol,fit,ck)
    bank,geom,_=__import__('scripts.z13_target_model',fromlist=['build_arm']).build_arm('R1',torch,ck)
    for k,video in enumerate(protocol['videos']):
        CONTEXT.update(variant='data',video=video,videos_done=k,videos_total=len(protocol['videos']))
        check();extract_one(video,k,len(protocol['videos']))
        marker=RUN/'validation'/(video+'.json')
        if not marker.exists():write(marker,validate_video(video,RUN/'data'/(video+'.npz'),geom,bank.eye_size))
    del bank
    write(RUN/'data_manifest.json',{v:sha(RUN/'data'/(v+'.npz')) for v in protocol['videos']})
    evaluate(torch,protocol,fit,ck);summarize(protocol)
    write(RUN/'status.json',dict(state='completed',stage='report written',done=3,total=3,updated=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(RUN/'status.json',dict(state='paused' if isinstance(error,TimeoutError) else 'failed',stage=str(error),updated=datetime.now(timezone.utc).isoformat()))
        raise
