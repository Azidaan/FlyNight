"""Bounded observed-memory probe. Frozen core, existing train20/train26 only.

Phases: prepare; CPU tracker replay; GPU timing/reproduction; full paired pilot.
No video decoding, downloads, dev, HOTA, held-out75 or core training.
"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/z13_observed_memory_20261021'
IDENT='z13_observed_memory_v1'
MSK=timezone(timedelta(hours=3))
T0=time.monotonic()
LIMIT=180
PHASE=''


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def now():return datetime.now(MSK)
def status(stage,**kw):
    if time.monotonic()-T0>LIMIT or (RUN/f'{PHASE}.stop').exists():raise TimeoutError('Phase budget/stop reached')
    write(RUN/'status.json',dict(state='running',phase=PHASE,stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0),budget_seconds=LIMIT,**kw))
def verify(p):
    for rel,digest in p['sha256'].items():
        f=(ROOT/rel).resolve();f.relative_to(ROOT.resolve())
        if sha(f)!=digest:raise RuntimeError('Changed frozen source: '+rel)
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');anchor='## Журнал изменений\n'
    assert s.count(anchor)==1
    path.write_text(s.replace(anchor,anchor+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')
def schedule():
    n=now()
    if n.weekday()<5 and n.hour>=7 and (n.hour>7 or n.minute>=40):raise RuntimeError('Weekday work window requires explicit return authorization')
    for k in range(8):
        d=n.date()+timedelta(days=k)
        t=datetime.combine(d,datetime.min.time(),MSK).replace(hour=7,minute=40)
        if d.weekday()<5 and t>n:return t
    raise AssertionError()


def prepare():
    if RUN.exists():raise RuntimeError('Run already exists; inspect, do not duplicate')
    source=read(ROOT/'runs/z13_target_center_20261018/protocol.json')
    names=['tracking/z13_observed_memory.py','scripts/z13_observed_memory_20261021.py',
           'scripts/z13_observed_memory_progress.ps1','tests/test_z13_observed_memory.py',
           'scripts/z13_target_model.py','scripts/z13_train.py','tracking/z13_target_center.py',
           'tracking/z13_readout.py','tracking/foveated_eyes.py','tracking/xywh_bytetrack.py',
           'scripts/fovea_target_30hz.py','scripts/freeze_bench.py','tracking/offline_association.py',
           'scripts/run_bdd_video_rate_experiment.py','tracking/gpu_lock.py','scripts/z13_continue.py',
           'runs/z13_target_center_20261018/protocol.json','runs/z13_target_center_20261018/report.json',
           'runs/z13_spatial_v2_20261013/D/full/model.pt']
    for v in source['fit']+source['evaluate']:
        names.extend([f'runs/z13_data/{v}.npz',f'runs/train_cache/{v}/detections/raw_video.npz'])
    for arm in ('C','D','R1'):
        names.extend([f'runs/z13_target_center_20261018/{arm}/center_rows.json',f'runs/z13_target_center_20261018/{arm}/choice.json'])
    p=dict(id=IDENT,created_at=now().isoformat(),schedule_hard_stop=schedule().isoformat(),
        authorization='Author: Давай дальше. Bounded preparation and pilot under existing plan; audit180s+15, timing180s+15, full only if conservative estimate<=1200s; no automatic larger budget.',
        fit=source['fit'],evaluate=source['evaluate'],arms=['C','D','R1'],lambdas=source['lambdas'],
        action='Replace one observed feature/target map by a bank of distinct pre-loss observations in the per-target center readout; no query-time memory updates.',
        paper='https://arxiv.org/html/2104.00324v1 sections3.1-3.3: separate historical observations. Our ridge-bank adaptation is not a reproduction of STMTrack attention.',
        memory='Use all unique frame numbers among exemplar slots1,2,3 (last occurrence on padding duplicates). Slot0 initializes the core and is excluded from model fitting. Each slot has its own replayed observed track box and Gaussian target. Concatenate observed training maps; ridge objective averaged over all observed columns, not sum.',
        control='Same entire feature sequence, same scaler, same center target and lambda grid; control uses slot3 only. Feature normalization from200 deterministic observed examples of fit20; independent lambda selection per arm/mode on fit20 only, criterion identical to original center probe.',
        primary='C_memory-C_single and C_memory-R1_memory: both paired-video95% CI lower bounds>0 on all26 videos. Modernization is judged first on stock C to separate readout change from learned D. Additional memory contribution over input control is reported; no anatomy claim.',
        secondary='D_memory-D_single, D_memory-C_memory, D_memory-R1_memory; difference-in-differences of memory effects C and R1. Known nonuseful and unknown at fixed global1/2/5% separately. No secondary rescue or threshold tuning.',
        reproduction='CPU replay must exactly match all cached exemplar frame numbers/crops and the final observed mask. Single-observation center control: same lambda; maximum coordinate difference<0.5px and peak-score difference<0.001 against saved center rows for each arm. Benchmark tests only reproduction/timing, no new efficacy decision.',
        counterargument='Consecutive pre-loss observations may be redundant or contaminated by a neighbour; multiple inconsistent target maps can degrade an already good single-observation model.',
        decision='If primary gate fails, stop this implementation without changing settings or asserting a cause. If it passes, record candidate status only; causal intervention policy and HOTA require separate validation and authorization.',
        prohibited=['video decoding','downloads','core training','dev','HOTA','heldout75','changing original scientific sources'],
        sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p)
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath)
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: память отдельных наблюдений до потери цели',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',
        tags=['fly-problem-P8','STMTrack','engineering-future'],
        question='Улучшает ли банк отдельных наблюдений до потери центрированное считывание FlyVis относительно одного образца и того же банка на входе глаза?',
        hypothesis='Несколько наблюдаемых представлений цели могут уменьшить ошибки по сравнению с последним образцом; это гипотеза, а не установленная причина прежних ошибок.',
        expected='Одна фиксированная замена внутреннего считывания, C/D/R1, прежние fit20/train26. Кадры поиска в память не добавляются. Сначала CPU-восстановление рамок и проба стоимости, затем ограниченный пилот только если оценка укладывается в1200с.',
        acceptance=[dict(criterion='Рамки прошлого восстановлены без разметки; вырезки, кадры и последняя маска совпадают; контроль одного образца воспроизведён с прежним lambda и допусками <0,5px/<0,001.',met=None),
                    dict(criterion='C память − C один образец: нижняя граница парного повидео95% ДИ>0 на всех train26.',met=None),
                    dict(criterion='C память − R1 память: нижняя граница парного повидео95% ДИ>0 на всех train26.',met=None)],
        summary='Подготовка ограниченной проверки памяти отдельных наблюдений; научного результата пока нет.',done='',obtained='',interpretation='',decision='',
        cannot_claim='Нет результата HOTA, доказательства анатомии связей или преимущества обучения D. Банк гребневой регрессии — собственная адаптация принципа исторических наблюдений, не полная реализация STMTrack.',
        metrics=[],commits=[],files=['tracking/z13_observed_memory.py','scripts/z13_observed_memory_20261021.py'],
        evidence=[dict(kind='file',ref=str((RUN/'protocol.json').relative_to(ROOT)).replace('\\','/'),claim='Постановка и критерии до результатов'),
                  dict(kind='file',ref='research_flyvis_mechanism_2026-10-16/ИНЖЕНЕРНЫЕ_АНАЛОГИ.md',locator='раздел5',claim='Механизм исторических наблюдений и ограничения переноса')]))
    write(catpath,cat)
    journal('постановка проверки памяти наблюдений','*Постановка.* '+p['action']+'\n\nОдна фиксированная проверка по `runs/z13_observed_memory_20261021/protocol.json`: сначала восстановление наблюдаемых рамок без видео и разметки, затем ограниченный замер времени и воспроизведения. Только при оценке не более1200с — полный функциональный пилот. Критерии записаны до результатов; параметры ядра, исходные кэши, dev, HOTA и75 отложенных не используются для изменения модели.')


def replay():
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='0'
    import torch
    torch.cuda.is_available=lambda:False
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from ultralytics.trackers.basetrack import BaseTrack,TrackState
    from scripts.fovea_target_30hz import oracle_tracker,new_stats
    from scripts.freeze_bench import XYWH_CONFIG
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,make_motion_estimator
    from tracking.foveated_eyes import EyeBank,crop_box,box_to_crop_frame
    from tracking.offline_association import _DetSlice,make_tracker_args
    from tracking.z13_readout import HexGeometry
    p=read(RUN/'protocol.json');verify(p)
    bank=EyeBank(make_motion_estimator('flyvis_tiled',30.,.02));geom=HexGeometry(bank.points_local,bank.eye_size)
    videos=p['fit']+p['evaluate'];records=[]
    for vi,v in enumerate(videos):
        status('Восстановление наблюдаемых рамок',variant=v,done=vi,total=len(videos))
        path=RUN/'observations'/f'{v}.npz'
        if path.exists():records.append(read(path.with_suffix('.json')));continue
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
            d={k:z[k] for k in ('ep_tid','ep_start','ep_offset','ep_pert','fr_frame','fr_crop','ex_mask')}
        hist={};pending={};found={};masks=np.zeros((len(d['ep_tid']),4,721),bool);frames=np.zeros((len(masks),4),np.int32)
        for e,(tid,start,pert) in enumerate(zip(d['ep_tid'],d['ep_start'],d['ep_pert'])):
            if not pert:pending.setdefault(int(start),{})[int(tid)]=e
        Base=oracle_tracker(None,{},new_stats());current=[0]
        class Replay(Base):
            def _pre_first_associate(self,pool,unconfirmed,img,results_high):
                super()._pre_first_associate(pool,unconfirmed,img,results_high)
                for t in pool:
                    tid=int(t.track_id)
                    if t.state==TrackState.Tracked or tid not in pending.get(current[0],{}):continue
                    e=pending[current[0]][tid];h=list(hist[tid]);h=[h[0]]*(4-len(h))+h[-4:]
                    offset=int(d['ep_offset'][e])
                    for j,(fr,box,crop) in enumerate(h):
                        if fr!=int(d['fr_frame'][offset+j]) or not np.array_equal(crop,d['fr_crop'][offset+j]):raise RuntimeError(f'Observed history mismatch {v}/{e}/{j}')
                        frames[e,j]=fr;masks[e,j]=geom.mask_in_box(box_to_crop_frame(box,crop,bank.eye_size))
                    if not np.array_equal(masks[e,3],d['ex_mask'][e]):raise RuntimeError(f'Last mask mismatch {v}/{e}')
                    found[e]=True
        BaseTrack.reset_id();tr=Replay(make_tracker_args(**XYWH_CONFIG))
        det=load_detection_cache(ROOT/f'runs/train_cache/{v}/detections/raw_video.npz')
        for f in range(max(pending)+1):
            if f%100==0:status('Восстановление наблюдаемых рамок',variant=v,done=vi,total=len(videos),frame=f)
            current[0]=f;xyxy,conf,cls=detections_for_frame(det,f);tr.update(_DetSlice(xyxy,conf,cls))
            for t in tr.tracked_stracks:
                if t.frame_id==tr.frame_id:
                    crop=crop_box(tuple(float(a) for a in t.mean[:4]),(1280,720))
                    hist.setdefault(int(t.track_id),deque(maxlen=4)).append((f,np.asarray(t.xyxy,float).copy(),crop))
        expected=int((~d['ep_pert']).sum())
        if len(found)!=expected:raise RuntimeError(f'Missing restored episodes {v}: {len(found)}/{expected}')
        path.parent.mkdir(exist_ok=True);np.savez_compressed(path,masks=masks,frames=frames)
        rec=dict(video=v,episodes=expected,crops_and_frames_equal=True,last_masks_equal=True,source_sha256=sha(ROOT/f'runs/z13_data/{v}.npz'),output_sha256=sha(path))
        write(path.with_suffix('.json'),rec);records.append(rec)
    write(RUN/'replay_report.json',dict(videos=len(records),episodes=sum(r['episodes'] for r in records),all_equal=True,seconds=time.monotonic()-T0,used_gt=False,decoded_video=False,records=records))
    verify(p)


def arrays(torch,arm,bb,bank,geom,data,lambdas,stats,stage,max_batches=None):
    import numpy as np
    from scripts.z13_target_model import episode_features,center_and_presence
    from tracking.z13_target_center import ridge_scores,observed_center_target
    from tracking.z13_observed_memory import memory_scores
    from tracking.z13_readout import to_frame
    from scripts.z13_train import success
    by_ep={};vid=[];masks=[];frames=[]
    for v,n in data.meta:
        vid.extend([v]*n)
        with np.load(RUN/'observations'/f'{v}.npz') as z:masks.extend(z['masks']);frames.extend(z['frames'])
    for i,l in enumerate(data.labels):
        if not l[8]:by_ep.setdefault(l[0],[]).append(i)
    eps=sorted(by_ep);result={'single':[],'memory':[]};nf=0;t0=time.monotonic()
    dev=bb.dev if bb is not None else 'cpu'
    for left in range(0,len(eps),8):
        if max_batches is not None and left//8>=max_batches:break
        status(stage,variant=arm,done=left,total=len(eps))
        ch=eps[left:left+8];mx=[max(data.labels[i][1] for i in by_ep[e]) for e in ch]
        hx,ms=data.batch(torch,dev,ch,mx)
        need={1,2,3}|{4+data.labels[i][1] for e in ch for i in by_ep[e]}
        F=episode_features(torch,arm,bb,hx,geom.neighbors,need);nf+=len(ch)*hx.shape[1]
        for j,e in enumerate(ch):
            observed=np.stack([np.zeros_like(F[1][j].T),*[F[t][j].T for t in (1,2,3)]])
            observed=(observed-stats[0])/stats[1]
            idx=by_ep[e];search=np.stack([(F[4+data.labels[i][1]][j].T-stats[0])/stats[1] for i in idx])
            if not np.array_equal(masks[e][3],ms[j].cpu().numpy()):raise RuntimeError('Observation pairing mismatch')
            score={
                'single':ridge_scores(observed[3],observed_center_target(masks[e][3],geom.points,geom.spacing),search,lambdas['single']),
                'memory':memory_scores(observed,np.asarray(masks[e]),geom.points,geom.spacing,search,lambdas['memory'],frames[e])}
            for mode in score:
                for si,i in enumerate(idx):
                    lab=data.labels[i];values=[]
                    for li in range(len(lambdas[mode])):
                        ce,pres=center_and_presence(score[mode][li,si],geom)
                        center=np.asarray(to_frame(ce,lab[7],bank.eye_size))
                        row=dict(center=center.tolist(),p_present=pres,pred=np.asarray(lab[6],float).tolist())
                        if lab[2]==1:
                            box=np.asarray(lab[5]);row.update(err=float(np.linalg.norm(center-lab[4])),err_pred=float(np.linalg.norm(np.asarray(lab[6])-lab[4])),scale=float(max(box[2]-box[0],box[3]-box[1])))
                            row['useful']=bool(success(dict(kind=1,**row)))
                        elif lab[2]==0:row['useful']=False
                        values.append(row)
                    result[mode].append(dict(label=i,video=vid[e],kind=int(lab[2]),pos=int(lab[1]),by_lambda=values))
        del F,hx,ms
    return result,dict(seconds=time.monotonic()-t0,padded_frames=nf,episodes=len(eps))


def reproduce(arm,rows,lam,partial=False):
    import numpy as np
    saved=read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')
    old={r['label']:r for r in saved}
    if not partial and len(rows)!=len(old):raise RuntimeError('Control row count mismatch')
    dc=max(float(np.max(np.abs(np.asarray(r['center'])-old[r['label']]['center']))) for r in rows)
    ds=max(abs(r['p_present']-old[r['label']]['p_present']) for r in rows)
    expected=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
    record=dict(rows=len(rows),max_center_px=dc,max_peak_score=ds,lambda_matches=lam==expected,partial=partial)
    if dc>=.5 or ds>=.001 or lam!=expected:raise RuntimeError('Single-observation reproduction failed: '+str(record))
    return record


def compute(benchmark):
    import gc
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tracking.gpu_lock import GpuLock
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p)
    if not read(RUN/'replay_report.json')['all_equal']:raise RuntimeError('Replay not complete')
    if not benchmark and not read(RUN/'benchmark.json')['fits_1200s']:raise RuntimeError('Full pilot not authorized by cost gate')
    checkpoint=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    fit=Episodes(p['fit']);evaluation=Episodes(p['evaluate'])
    allrows={};choices={}
    with GpuLock('FlyVis_z13_observed_memory_single_gpu'):
        for arm in (['D'] if benchmark else p['arms']):
            directory=RUN/arm
            if not benchmark and (directory/'done.json').exists():
                for mode in ('single','memory'):allrows[arm+'_'+mode]=read(directory/(mode+'_rows.json'))
                choices[arm]=read(directory/'choice.json');continue
            status('Подготовка фиксированного ядра',variant=arm,done=0,total=3)
            bank,geom,bb=old.build_arm(arm,torch,checkpoint)
            if bb is not None and not str(bb.dev).startswith('cuda'):raise RuntimeError('GPU inference explicitly required')
            tnorm=time.monotonic()
            stats=old.feature_stats(torch,arm,bb,geom,fit)
            normsec=time.monotonic()-tnorm
            if benchmark:
                lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
                r,timing=arrays(torch,arm,bb,bank,geom,evaluation,{m:[lam] for m in ('single','memory')},stats,'Замер времени без решения об эффективности',max_batches=2)
                one=old.useful_rows(r['single'],0);repro=reproduce(arm,one,lam,partial=True)
                costs={}
                for name,data in [('fit',fit),('evaluate',evaluation)]:
                    eps={}
                    for lab in data.labels:
                        if not lab[8]:eps[lab[0]]=max(eps.get(lab[0],-1),lab[1])
                    lengths=[4+eps[e]+1 for e in sorted(eps)]
                    costs[name]=sum(len(lengths[k:k+8])*max(lengths[k:k+8]) for k in range(0,len(lengths),8))
                estimate=2*(normsec*3+timing['seconds']/timing['padded_frames']*sum(costs.values())*3+45)
                write(RUN/'benchmark.json',dict(timing=timing,normalization_seconds=normsec,padded_frame_cost=costs,conservative_total_seconds=estimate,fits_1200s=estimate<=1200,reproduction=repro,caveat='Two batches; factor two uncertainty; all arms conservatively priced as FlyVis. Full fit uses five lambdas; estimate is uncertain.'))
                return
            fr,ftiming=arrays(torch,arm,bb,bank,geom,fit,{m:p['lambdas'] for m in ('single','memory')},stats,'Подбор lambda на прежних20 train')
            choice={};lambdas={}
            for mode in fr:
                scores=[float(np.mean([r['by_lambda'][li]['useful'] for r in fr[mode] if r['kind']==1])) for li in range(len(p['lambdas']))]
                idx=int(np.argmax(scores));lambdas[mode]=[p['lambdas'][idx]];choice[mode]=dict(lambda_value=lambdas[mode][0],fit_scores=scores)
            er,timing=arrays(torch,arm,bb,bank,geom,evaluation,lambdas,stats,'Оценка на всех26 train')
            for mode in er:allrows[arm+'_'+mode]=old.useful_rows(er[mode],0)
            repro=reproduce(arm,allrows[arm+'_single'],choice['single']['lambda_value'])
            for mode in er:write(directory/(mode+'_rows.json'),allrows[arm+'_'+mode])
            write(directory/'choice.json',choice);write(directory/'reproduction.json',repro)
            write(directory/'done.json',dict(timing=timing,fit_timing=ftiming,normalization_seconds=normsec))
            choices[arm]=choice
            del bb,bank,fr,er;gc.collect();torch.cuda.empty_cache()
    keys=[{(r['video'],r['label'],r['kind']) for r in rows} for rows in allrows.values()]
    if not all(k==keys[0] for k in keys):raise RuntimeError('Paired evaluation keys differ')
    pv={name:per_video_useful(rows) for name,rows in allrows.items()}
    pairs=[('C_memory','C_single'),('C_memory','R1_memory'),('D_memory','D_single'),('D_memory','C_memory'),('D_memory','R1_memory')]
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    vids=sorted(pv['C_single'])
    cm={v:pv['C_memory'][v]-pv['C_single'][v] for v in vids};rm={v:pv['R1_memory'][v]-pv['R1_single'][v] for v in vids}
    comp['C_memory_effect-R1_memory_effect']=paired(cm,rm)
    report=dict(summary={name:dict(rows=len(rows),useful_mean_video=float(np.mean(list(pv[name].values()))),coverage=coverage_table(rows)) for name,rows in allrows.items()},paired=comp,choices=choices,
        primary_pass=comp['C_memory-C_single']['ci95'][0]>0 and comp['C_memory-R1_memory']['ci95'][0]>0,
        limitations='Exploratory reused train26; one fixed bank, no query updates. Full video-paired intervals are exploratory. No HOTA, calibrated rejection or anatomy attribution.')
    verify(p);write(RUN/'report.json',report)


def controller(phase,budget):
    p=read(RUN/'protocol.json');verify(p)
    marker=RUN/(phase+'_controller_result.json')
    if marker.exists():raise RuntimeError('Completed phase must not be duplicated')
    hard=min(now()+timedelta(seconds=budget+15),schedule()-timedelta(seconds=10))
    soft=hard-timedelta(seconds=15)
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--phase',phase,'--budget',str(budget)]
    with (RUN/(phase+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(phase+'.stderr.log')).open('w',encoding='utf-8') as err:
        worker=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'controller.json',dict(phase=phase,worker_pid=worker.pid,command=cmd,started_at=now().isoformat(),save_stop_at=soft.isoformat(),hard_stop_at=hard.isoformat(),schedule_hard_stop=schedule().isoformat()))
        hard_kill=False
        while worker.poll() is None:
            if now()>=soft:(RUN/(phase+'.stop')).touch(exist_ok=True)
            if now()>=hard:
                subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],check=True,capture_output=True);hard_kill=True;break
            time.sleep(.5)
        code=worker.wait(timeout=15)
        write(marker,dict(worker_ended=True,exit_code=code,hard_timeout=hard_kill,ended_at=now().isoformat()))
        if code!=0:write(RUN/'status.json',dict(state='paused' if hard_kill else 'failed',stage='Этап остановлен; результаты сохранены',phase=phase,updated=now().isoformat()))
    return code


def main():
    global LIMIT,PHASE
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--control',choices=['audit','benchmark','full']);ap.add_argument('--phase',choices=['audit','benchmark','full']);ap.add_argument('--budget',type=int,default=180);a=ap.parse_args()
    if a.prepare:prepare();return
    if a.control:return controller(a.control,a.budget)
    if not a.phase:raise ValueError('Phase required')
    PHASE=a.phase;LIMIT=a.budget
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['OPENBLAS_NUM_THREADS']='2'
    try:
        if PHASE=='audit':replay()
        else:compute(PHASE=='benchmark')
        write(RUN/'status.json',dict(state='completed',phase=PHASE,stage='Этап сохранён',done=1,total=1,updated=now().isoformat()))
    except Exception as exc:
        write(RUN/'status.json',dict(state='paused' if isinstance(exc,TimeoutError) else 'failed',phase=PHASE,stage=str(exc),updated=now().isoformat()))
        raise

if __name__=='__main__':sys.exit(main())
