"""Bounded frozen-core candidate cache and oracle reserve; not a tracker result."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_observed_memory_20261021 import read,sha,verify
RUN=ROOT/'runs/z13_candidate_reserve_20261021'
IDENT='z13_candidate_reserve_v1';MSK=timezone(timedelta(hours=3))
T0=time.monotonic();LIMIT=1800;SPLIT='evaluate'


def now():return datetime.now(MSK)
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    for i in range(40):
        try:q.replace(p);return
        except PermissionError:
            if i==39:raise
            time.sleep(.025)
def check():
    if time.monotonic()-T0>LIMIT or (RUN/(SPLIT+'.stop')).exists():raise TimeoutError('Budget/stop reached')
    if now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Morning save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=SPLIT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    p.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run must not be duplicated')
    old=read(ROOT/'runs/z13_observed_memory_20261021/protocol.json');verify(old)
    names=['tracking/z13_candidates.py','scripts/z13_candidate_reserve_20261021.py','tests/test_z13_candidates.py']
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author explicitly authorized autonomous necessary runs, repairs, commits and push until07:40 MSK21.10.2026; no heartbeat automation.',
        fit=old['fit'],evaluate=old['evaluate'],arms=['C','D','R1'],k=3,
        action='Prepare one replacement of FlyVis readout: associate distinct response candidates through time instead of always choosing the tallest response. First measure whether the candidate set contains recoverable centers.',
        extraction='Top1 exactly existing argmax/local weighted center. Up to3 local maxima, sorted by raw response, minimum peak-column distance2 lattice spacings. No search label or target location enters candidate extraction. Same features, history, Gaussian target, scaler and saved center lambda for each arm.',
        cache='All real search frames up to the last evaluation label, candidate positions in frame coordinates, normalized local features and response height. Seeds from the last observed mask; no query GT enters cache generation. Evaluation labels are stored separately.',
        primary='Oracle best useful among3 minus top1 on all26 videos. Oracle uses GT for evaluation only. Reserve is sufficient for one functional attempt when mean>=0.02 and paired95%lower>0. If upper<0.02, stop this candidate set; otherwise uncertain reserve, no causal or impossibility verdict.',
        downstream='Only after reserve: fit a single temporal candidate selector using fit20, matched C/D/R1 and geometric-only controls. Proposed functional acceptance before any selector outcomes: mean useful-center gain>=0.02 and lower95%CI>0 against same-arm top1; stock C selector also better than R1 selector. Harm/unknown selected outcomes separately; no claim of safe deployment.',
        counterargument='The correct object may not form a separate response maximum; small crop or representation may limit reserve. Oracle gain does not prove causal association is possible.',
        limits='No core training, downloads, dev, heldout75 or HOTA in this step. Frozen original sources. One GPU process with common mutex. Per-batch cache checkpoints; completed batches and arms are not recomputed.',
        budget_seconds=1800,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/2103.16556v2 sections3.1-3.7. Our candidate probe is not a reproduction of KeepTrack.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: запас среди трёх кандидатов перед временным сопоставлением',area='flyvis',state='running',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','KeepTrack','engineering-future'],
        question='Есть ли среди трёх раздельных пиков запас полезных центров для замены считывателя на сопоставление кандидатов во времени?',
        hypothesis='Часть ошибок одного максимума может исправляться выбором другого отдельного пика; это проверяемый запас, не установленная причина ошибок.',
        expected='Один проход фиксированных C/D/R1, прежние fit20/train26, до1800с. Точное воспроизведение top1; сохранение кандидатов без будущей разметки. Оракул только для оценки запаса.',
        acceptance=[dict(criterion='Все8326 строк каждого варианта и прежний top1 воспроизведены: центр<0,5px,балл<0,001; исходные хеши неизменны.',met=None),dict(criterion='C: оракул среди3−top1, средний сдвиг>=2п.п. и нижняя95%повидеоДИ>0; достаточный запас для одной функциональной попытки, не доказательство работы селектора.',met=None)],
        summary='Подготовлена ограниченная проверка запаса кандидатов; результата пока нет.',done='',obtained='',interpretation='',decision='',cannot_claim='Оракул использует разметку только при оценке и не является алгоритмом применения. Нет HOTA, доказательства безопасного отказа, причинного диагноза или эффективности KeepTrack на FlyVis.',metrics=[],commits=[],files=names[:2],evidence=[dict(kind='file',ref='runs/z13_candidate_reserve_20261021/protocol.json',claim='Критерии и неизменные источники до прогона')]))
    write(cp,cat);journal('постановка запаса кандидатов','*Постановка.* '+p['action']+'\n\nОракул — только диагностический потолок. Признаки и выбор кандидатов не читают разметку поиска. Заранее: до1800с, один GPU, C/D/R1; запас C не менее2п.п. с положительной нижней границей. Итог не заменяет функциональную проверку и HOTA. Полное разрешение автора до07:40 МСК; независимое сохранение07:35, дерево закончить до07:39:30.')


def extract():
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tracking.gpu_lock import GpuLock
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_target_center import ridge_scores,observed_center_target
    from tracking.z13_readout import to_frame
    from tracking.z13_candidates import candidates
    p=read(RUN/'protocol.json');verify(p)
    fit=Episodes(p['fit']);data=Episodes(p[SPLIT]);vid=[];crops=[];frameids=[]
    for v,n in data.meta:
        vid.extend([v]*n)
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
            crops.extend(z['fr_crop']);frameids.extend(z['fr_frame'])
    crops=np.asarray(crops);frameids=np.asarray(frameids)
    by_ep={}
    for i,l in enumerate(data.labels):
        if not l[8]:by_ep.setdefault(l[0],[]).append(i)
    eps=sorted(by_ep)
    # GT is deliberately confined to this separate evaluation table.
    labels=[]
    for e,ids in by_ep.items():
        for i in ids:
            l=data.labels[i];r=dict(label=i,episode=e,pos=l[1],video=vid[e],kind=l[2],pred=np.asarray(l[6]).tolist())
            assert np.array_equal(crops[data.off[e]+4+l[1]],l[7])
            if l[2]==1:r.update(gt=np.asarray(l[4]).tolist(),scale=float(max(l[5][2]-l[5][0],l[5][3]-l[5][1])))
            labels.append(r)
    write(RUN/SPLIT/'labels.json',labels)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    with GpuLock('flyvis_fovea_gpu'):
        for ai,arm in enumerate(p['arms']):
            folder=RUN/SPLIT/arm
            if (folder/'done.json').exists():continue
            status('Нормировка прежних признаков',variant=arm,done=0,total=len(eps),arms_done=ai,arms_total=3)
            bank,geom,bb=old.build_arm(arm,torch,ck)
            if bb is not None and not str(bb.dev).startswith('cuda'):raise RuntimeError('CUDA required for frozen core extraction')
            stats=old.feature_stats(torch,arm,bb,geom,fit)
            lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            dev=bb.dev if bb is not None else 'cpu';folder.mkdir(parents=True,exist_ok=True)
            for left in range(0,len(eps),8):
                path=folder/f'cache_{left:05d}.npz'
                if path.exists():continue
                status('Кандидаты на всех кадрах эпизодов',variant=arm,done=left,total=len(eps),arms_done=ai,arms_total=3)
                ch=eps[left:left+8];mx=[max(data.labels[i][1] for i in by_ep[e]) for e in ch]
                hx,ms=data.batch(torch,dev,ch,mx);F=old.episode_features(torch,arm,bb,hx,geom.neighbors,set(range(3,hx.shape[1])))
                storage={k:[] for k in ('episode','t','frame','count','centers','features','scores','crop_width','seed_center','seed_features')}
                for j,e in enumerate(ch):
                    xe=(F[3][j].T-stats[0])/stats[1];mask=ms[j].cpu().numpy();seed=geom.points[mask].mean(0)
                    seedc=to_frame(seed,crops[data.off[e]+3],bank.eye_size);seedf=xe[mask].mean(0)
                    ts=list(range(3,5+mx[j]));xs=np.stack([(F[t][j].T-stats[0])/stats[1] for t in ts]);y=observed_center_target(mask,geom.points,geom.spacing)
                    maps=ridge_scores(xe,y,xs,[lam])[0]
                    for si,t in enumerate(ts):
                        r=candidates(maps[si],xs[si],geom);n=len(r['indices']);crop=crops[data.off[e]+t]
                        cs=np.zeros((3,2));fs=np.zeros((3,xe.shape[1]));ss=np.zeros(3)
                        cs[:n]=[to_frame(c,crop,bank.eye_size) for c in r['centers']];fs[:n]=r['features'];ss[:n]=r['scores']
                        for k,val in dict(episode=e,t=t,frame=frameids[data.off[e]+t],count=n,centers=cs,features=fs,scores=ss,crop_width=float(crop[2]-crop[0]),seed_center=seedc,seed_features=seedf).items():storage[k].append(val)
                arr={k:np.asarray(v,dtype=np.int32 if k in ('episode','t','frame','count') else np.float64 if k in ('centers','scores') else np.float32) for k,v in storage.items()}
                temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**arr);temp.replace(path)
                write(path.with_suffix('.json'),dict(sha256=sha(path),episodes=ch,frames=len(arr['t'])))
                del F,hx,ms
            write(folder/'done.json',dict(completed=now().isoformat(),batches=len(list(folder.glob('cache_*.npz'))),lambda_value=lam))
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    verify(p)
    if SPLIT=='evaluate':report()


def report():
    import numpy as np
    from scripts.z13_continue import paired,per_video_useful
    p=read(RUN/'protocol.json');verify(p);labels=read(RUN/'evaluate/labels.json');res={};repro={};outrows={}
    for arm in p['arms']:
        folder=RUN/'evaluate'/arm;assert (folder/'done.json').exists();cache={}
        for path in sorted(folder.glob('cache_*.npz')):
            assert sha(path)==read(path.with_suffix('.json'))['sha256']
            with np.load(path) as z:
                for i in range(len(z['t'])):cache[(int(z['episode'][i]),int(z['t'][i]))]=(z['centers'][i],z['scores'][i],int(z['count'][i]))
        saved={r['label']:r for r in read(ROOT/f'runs/z13_observed_memory_20261021/{arm}/single_rows.json')};top=[];oracle=[];dc=ds=0.
        for l in labels:
            cs,ss,n=cache[(l['episode'],l['pos']+4)];r={k:l[k] for k in ('video','kind','label','pos','pred')};r.update(center=cs[0].tolist(),p_present=float(ss[0]))
            old=saved[l['label']];dc=max(dc,float(np.abs(cs[0]-old['center']).max()));ds=max(ds,abs(float(ss[0])-old['p_present']))
            useful=np.zeros(n,bool)
            if l['kind']==1:
                err=np.linalg.norm(cs[:n]-l['gt'],axis=1);ep=np.linalg.norm(np.asarray(l['pred'])-l['gt']);useful=(err<=.25*l['scale'])&(err<ep)
                r.update(err=float(err[0]),err_pred=float(ep),scale=l['scale'],useful=bool(useful[0]))
            elif l['kind']==0:r['useful']=False
            top.append(r);o=dict(r)
            if l['kind']==1:
                best=int(np.argmin(err));o.update(center=cs[best].tolist(),p_present=float(ss[best]),err=float(err[best]),useful=bool(useful.any()))
            elif l['kind']==0:o['useful']=False
            oracle.append(o)
        assert len(top)==len(saved)==8326 and dc<.5 and ds<.001
        repro[arm]=dict(rows=len(top),max_center_px=dc,max_peak_score=ds)
        a=per_video_useful(top);b=per_video_useful(oracle);d=paired(b,a)
        res[arm]=dict(top1_mean=float(np.mean(list(a.values()))),oracle3_mean=float(np.mean(list(b.values()))),reserve=d,sufficient=d['mean_per_video']>=.02 and d['ci95'][0]>0)
        outrows[arm]=dict(top1=top,oracle3=oracle)
        write(folder/'top1_rows.json',top);write(folder/'oracle3_rows.json',oracle)
    write(RUN/'reproduction.json',repro);write(RUN/'report.json',dict(arms=res,C_reserve_sufficient=res['C']['sufficient'],limitations='Oracle only; repeated train26; no effective causal selector, no HOTA.'))


def control(split,budget):
    marker=RUN/(split+'_controller_result.json')
    if marker.exists():raise RuntimeError('Existing controller result; preserve and inspect')
    p=read(RUN/'protocol.json');verify(p)
    soft=min(now()+timedelta(seconds=budget),datetime.fromisoformat(p['save_at']));hard=min(soft+timedelta(seconds=20),datetime.fromisoformat(p['hard_stop']))
    if now()>=soft:raise RuntimeError('No work window')
    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--extract',split,'--budget',str(budget)]
    with (RUN/(split+'.stdout.log')).open('w',encoding='utf-8') as out,(RUN/(split+'.stderr.log')).open('w',encoding='utf-8') as err:
        worker=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        write(RUN/'controller.json',dict(worker_pid=worker.pid,command=cmd,started_at=now().isoformat(),save_stop_at=soft.isoformat(),hard_stop_at=hard.isoformat(),phase=split))
        killed=False
        while worker.poll() is None:
            if now()>=soft:(RUN/(split+'.stop')).touch(exist_ok=True)
            if now()>=hard:subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],check=True,capture_output=True);killed=True;break
            time.sleep(.5)
        code=worker.wait(timeout=15);write(marker,dict(worker_ended=True,exit_code=code,hard_timeout=killed,ended_at=now().isoformat()))
        if code:write(RUN/'status.json',dict(state='failed',phase=split,stage='Сохранены готовые пакеты; требуется разбор',updated=now().isoformat()))
    return code


def main():
    global SPLIT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--control',choices=['evaluate','fit']);a.add_argument('--extract',choices=['evaluate','fit']);a.add_argument('--budget',type=int,default=1800);x=a.parse_args()
    if x.prepare:prepare();return
    if x.control:return control(x.control,x.budget)
    if not x.extract:raise ValueError('Choose a phase')
    SPLIT=x.extract;LIMIT=x.budget
    os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:extract();write(RUN/'status.json',dict(state='completed',phase=SPLIT,stage='Кандидаты и результат сохранены',updated=now().isoformat()))
    except Exception as exc:
        write(RUN/'status.json',dict(state='failed',phase=SPLIT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':sys.exit(main())
