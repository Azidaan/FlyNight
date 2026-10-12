"""Fixed dev35 stage 1; one GPU process, hard deadline, identical episodes.

The old eye's crop displacement is computed from the actual previous crop, not
from the last detector-updated track center. Source v1 is left unchanged.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.fly_efference_stage1 import episodes
from scripts.fly_efference_v2 import OUT, check_deadline, save

VARS=('none','old','new','ctrl')
STAGE=OUT/'stage1'


def video(vid,est):
    from ultralytics.trackers.basetrack import BaseTrack
    from scripts.fly_efference_eye import Ctrl,eye_raw
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import detections_for_frame,load_detection_cache
    from tracking.efference_eyes_v2 import ReferencedEfferenceEyeBank
    from tracking.foveated_eyes import EyeBank,crop_box
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking.retina import make_retina
    from tracking.xywh_bytetrack import XYWHByteTrack
    t0=time.time(); eps,n=episodes(vid)
    C={v:np.asarray(c) for v,c in json.loads((OUT/'calibration.json').read_text(encoding='utf-8'))['C'].items()}
    wanted={e['tid'] for e in eps}; start={}
    for i,e in enumerate(eps): start.setdefault(e['L'],[]).append(i)
    det=load_detection_cache(ROOT/'runs/train_cache'/vid/'detections/raw_video.npz')
    banks={'old':EyeBank(est,max_batch=16,decode_flow=True),'new':ReferencedEfferenceEyeBank(est,max_batch=16,decode_flow=True)}
    retina=make_retina('lamina',30.,blur_downscale=4)
    BaseTrack.reset_id(); tr=XYWHByteTrack(make_tracker_args(**XYWH_CONFIG))
    cap=cv2.VideoCapture(str(ROOT/'data/bdd100k/videos/train'/f'{vid}.mov'))
    active={}; errs={v:{} for v in VARS}; previous_tracks=[]; previous_img=None
    for frame_id in range(n):
        check_deadline()
        ok,frame=cap.read()
        if not ok: raise RuntimeError(f'Premature video EOF {vid}:{frame_id}/{n}')
        img=retina.process(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        for i in start.get(frame_id,[]):
            e=eps[i]; tid=e['tid']
            if tid not in banks['old'].crops or tid not in banks['new'].crops:
                raise RuntimeError(f'Missing eye at episode start {vid}:{tid}')
            active[i]={'c':{v:np.array(e['c0'],float) for v in VARS},'previous_crop':banks['old'].crops[tid], 'ctrl':Ctrl()}
            active[i]['ctrl'].prev=previous_img
        lost={eps[i]['tid'] for i in active}; preds={'old':{},'new':{}}
        for t in previous_tracks:
            tid=int(t.track_id)
            if tid in wanted and tid not in lost:
                p=tuple((t.mean[:4]+t.mean[4:8]).tolist())
                preds['old'][tid]=p; preds['new'][tid]=p
        for i,a in active.items():
            e=eps[i]; wh=e['wh']; vk=np.asarray(e['vk'])
            for var in ('old','new'):
                p=a['c'][var]+vk; preds[var][e['tid']]=(*p,*wh)
        for var in ('old','new'): banks[var].step(img,preds[var])
        for i in list(active):
            e=eps[i]; a=active[i]; tid=e['tid']; vk=np.asarray(e['vk']); wh=e['wh']
            for var in VARS:
                p=a['c'][var]+vk
                if var=='none': a['c'][var]=p
                elif var=='ctrl':
                    box=(*p,*wh); crop=crop_box(box,(img.shape[1],img.shape[0]))
                    r=a['ctrl'].raw(img,crop,box); a['ctrl'].prev=img
                    a['c'][var]+=C[var]@r
                else:
                    box=preds[var][tid]; r=eye_raw(banks[var],est,tid,box); velocity=C[var]@r
                    if var=='old':
                        pc=np.asarray(a['previous_crop']); nc=np.asarray(banks[var].crops[tid])
                        velocity+=(nc[:2]+nc[2:]-pc[:2]-pc[2:])/2
                        a['previous_crop']=tuple(nc)
                    a['c'][var]+=velocity
                if not np.isfinite(a['c'][var]).all(): raise ValueError('Non-finite position')
                if str(frame_id) in e['eval']:
                    errs[var].setdefault(i,{})[frame_id]=float(np.linalg.norm(a['c'][var]-np.asarray(e['eval'][str(frame_id)])))
            if frame_id>=e['end']: del active[i]
        tr.update(_DetSlice(*detections_for_frame(det,frame_id)))
        tracked=list(tr.tracked_stracks); alive={int(t.track_id) for t in tracked+tr.lost_stracks}
        births={int(t.track_id):tuple(t.mean[:4]) for t in tracked if int(t.track_id) in wanted and int(t.track_id) not in banks['old'].track_ids}
        for var in ('old','new'):
            banks[var].birth(img,births); banks[var].remove(set(banks[var].track_ids)-alive)
        previous_tracks=tracked; previous_img=img
        if frame_id%100==0:
            save(OUT/'status.json',{'phase':'stage1','video':vid,'frame':frame_id,'frames':n,'elapsed_video':time.time()-t0})
            print(f'[VIDEO] {vid} {frame_id}/{n} eyes={len(banks["old"].track_ids)} {time.time()-t0:.0f}s',flush=True)
    cap.release()
    per={v:[x for a in errs[v].values() for x in a.values()] for v in VARS}
    assert len({len(x) for x in per.values()})==1
    result={'video':vid,'episodes':len(eps),'eval_frames':len(per['none']),
            'mean_err':{v:float(np.mean(x)) if x else None for v,x in per.items()},
            'errs':{v:{str(i):x for i,x in errs[v].items()} for v in VARS},'seconds':time.time()-t0}
    save(STAGE/f'{vid}.json',result)
    return result


def report(vids):
    rows={v:json.loads((STAGE/f'{v}.json').read_text(encoding='utf-8')) for v in vids if (STAGE/f'{v}.json').exists()}
    use=[v for v in rows if rows[v]['eval_frames']>0]
    rep={'videos_total':len(vids),'videos_done':len(rows),'videos_with_eval':len(use),'complete':len(rows)==len(vids),
         'mean_err':{var:float(np.mean([rows[v]['mean_err'][var] for v in use])) for var in VARS} if use else {},'diff':{},'criterion':False}
    rng=np.random.default_rng(1)
    if use:
        indices=rng.integers(0,len(use),size=(5000,len(use)))
        for a,b in (('new','old'),('new','none'),('new','ctrl'),('old','none'),('ctrl','none')):
            d=np.array([rows[v]['mean_err'][a]-rows[v]['mean_err'][b] for v in use]); bs=d[indices].mean(1)
            rep['diff'][f'{a}-{b}']={'mean_per_video':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'better':int((d<0).sum()),'n':len(use)}
        rep['criterion']=rep['complete'] and all(rep['diff'][f'new-{b}']['ci95'][1]<0 for b in ('old','none','ctrl'))
    save(OUT/'stage1_report.json',rep)
    return rep


def run():
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    gate=json.loads((OUT/'gate.json').read_text(encoding='utf-8'))
    if not gate['all_passed']: raise RuntimeError('Technical gate failed: stage 1 forbidden')
    selection=json.loads((ROOT/'runs/train_cache/selection.json').read_text(encoding='utf-8'))
    vids=selection['videos']['dev']
    save(OUT/'stage1_manifest.json',{'videos':vids,'variants':VARS,'protocol':'paired per-video mean center error, same annotated episodes','started':time.time()})
    est=make_motion_estimator('flyvis_tiled',30.,.02)
    elapsed=[]
    try:
        for i,v in enumerate(vids):
            check_deadline()
            if (STAGE/f'{v}.json').exists(): continue
            remaining=float(os.environ['EFFERENCE_DEADLINE'])-time.time()
            if elapsed and remaining < 1.25*max(elapsed)+180:
                print('[STOP] Insufficient time for another complete video',flush=True); break
            r=video(v,est); elapsed.append(r['seconds'])
            print(f'[DONE] {i+1}/{len(vids)} {r["mean_err"]}',flush=True)
    finally:
        print(json.dumps(report(vids),ensure_ascii=False),flush=True)


if __name__=='__main__': run()
