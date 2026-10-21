"""Fixed transfer of the preregistered map-only104 readout; no dev fitting.

Candidate extraction is the same numerical path as candidate_reserve.
Only the input episode directory and result directory differ on dev35.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_map_transfer104_20261021';SOURCE=ROOT/'runs/z13_candidate_data104_20261021'
DEV=ROOT/'runs/z13_target_center_dev_20261018';CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
IDENT='z13_map_transfer104_v1';ATTEMPT='technical';LIMIT=300;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')
def prepare():
    if RUN.exists():raise RuntimeError('Existing run; inspect without duplication')
    source=read(ROOT/'runs/z13_map_only104_20261021/report.json');assert source['primary_pass']
    base=read(SOURCE/'protocol.json');verify(base);videos=read(ROOT/'runs/train_cache/selection.json')['videos']['dev']
    assert len(videos)==35 and not set(videos)&set(base['train_full104']+base['evaluation26'])
    prior=read(DEV/'report.json');assert prior['summary']['C_center']['rows']==9314
    names=['scripts/z13_map_transfer104_20261021.py','scripts/z13_candidate_reserve_20261021.py','scripts/z13_candidate_identity_20261021.py','scripts/z13_candidate_ranker_20261021.py','tracking/z13_candidate_ranker.py','tracking/z13_candidates.py','runs/z13_map_only104_20261021/protocol.json','runs/z13_map_only104_20261021/report.json','runs/z13_candidate_data104_20261021/protocol.json','runs/z13_candidate_data104_20261021/report.json','runs/z13_target_center_dev_20261018/report.json','runs/train_cache/selection.json']
    for arm in ('C','D','R1'):
        for seed in (0,1,2):names.extend(f'runs/z13_candidate_data104_20261021/{arm}/geometry/seed{seed}/{n}' for n in ('model.pt','training.json'))
        for v in videos:names.extend([f'runs/z13_target_center_dev_20261018/{arm}/{v}/center_rows.json',f'runs/z13_target_center_dev_20261018/{arm}/{v}/done.json'])
    names.append('runs/z13_target_center_dev_20261018/protocol.json')
    names.extend(f'runs/z13_target_center_dev_20261018/data/{v}.npz' for v in videos)
    p=dict(id=IDENT,created=now().isoformat(),authorization='Explicit autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026.',
        action='Transfer the fixed map-only104 choice to all35dev videos; no setting/model/seed selection and no new training. C is primary before outcomes; D secondary cannot rescue C.',
        fit=base['fit'],videos=videos,arms=['C','D','R1'],model_mode='geometry',seeds=[0,1,2],
        extraction='Unchanged three-peak candidate definition, same train20 normalizer, same saved center lambda, same frozen C/D/R1 cores and episode history. Dev raw episodes already exist and are hash-frozen. No query labels enter feature extraction or model input.',
        primary='C_map-C_top1: mean>=0.02 and paired-video95%lower>0; C_map-R1_map and C_map-C_nearest: both lower>0; all3 C seed point gains versus top1 nonnegative. All conditions required.',
        secondary='D same contrasts; C/D map versus D prior; generic readout effect on R1 and Cgain-R1gain reported. No dev tuning or choose20vs104/bestseed. Reused dev35 is not blind.',
        reproduction='Before full transfer first saved train batch all3arms: candidate centers<0.5px, response heights<0.001, same lambda. All9314dev top1 rows must reproduce earlier CPU dev35 centers<0.5px, scores<0.001, same lambda. Any failure invalidates efficacy until explained, no tolerance relaxation.',
        counterargument='Positive train26 may be overfit or geometry-specific; train26 was reused and the generic module also improves R1. A different dataset partition can reverse the readout gain.',
        decision='If all primary conditions pass, freeze as transferable functional candidate before causal intervention/runtime gate; otherwise retain the original centered readout, no dev retuning. No HOTA claim.',
        technical_budget_seconds=300,budget_seconds=1800,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        prohibited='No core training, downloads, protected75, detector changes, HOTA or dev fitting.',sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: фиксированный перенос чтения карты104 на dev35',area='flyvis',state='ready',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future'],
        question='Сохраняется ли выигрыш фиксированного чтения пиков карты C на других35видео без подбора?',hypothesis='Выбор по карте может переноситься лучше простого максимума и ближайшего к прогнозу пика; перенос пока не измерен.',
        expected='Все35dev, прежние9314 строк, C/D/R1, те же3 geometry-модели104, никаких новых моделей и изменения λ/нормировки. Техническое воспроизведение до полного GPU-кэша, затем CPU-применение; до300+1800с.',
        acceptance=[dict(criterion=p['reproduction'],met=None),dict(criterion='C карта−C top1: >=2п.п., нижняя95%повидеоДИ>0; точки всех3 сидов неотрицательны.',met=None),dict(criterion='C карта лучше R1 карты и ближайшего к прогнозу пика C: обе нижние95%повидеоДИ>0.',met=None)],summary='Подготовлен фиксированный перенос без подбора на dev; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Ранее использованный dev35 не является слепым тестом. Карта содержит FlyVis даже без дополнительных векторов активности. Прирост общего считывателя отделяется R1; нет HOTA, калиброванного отказа и доказательства анатомии.',metrics=[],commits=[],files=['scripts/z13_map_transfer104_20261021.py','tracking/z13_candidate_ranker.py'],evidence=[dict(kind='file',ref='runs/z13_map_transfer104_20261021/protocol.json',claim='Условия до dev исхода'),dict(kind='file',ref='runs/z13_map_only104_20261021/report.json',claim='Основные train26 критерии выполнены')]))
    write(cp,cat);journal('постановка переноса фиксированной карты104','*Постановка.* '+p['action']+'\n\nКритерии до исходов: '+p['primary']+'\n\n'+p['reproduction']+'\n\n'+p['decision']+'\n\nГотовые35dev эпизодов и9 моделей104 найдены локально, проверены хешами. Сырые векторы в моделях обнулены; пики/высоты карты остаются. No-FlyVis контроль — R1 с тем же узлом. Бюджет300с технической проверки +1800с полного кэша, затем отдельный CPU-этап до120с; независимая остановка дерева и сохранение к сроку автора. Отложенные75 не используются.')


def dataset(videos,directory):
    import numpy as np
    from scripts import z13_train as train
    old=train.DATA
    try:train.DATA=directory;data=train.Episodes(videos)
    finally:train.DATA=old
    crops=[];frames=[];vid=[];base={};offset=0
    for v,n in data.meta:
        vid.extend([v]*n);base[v]=offset
        with np.load(directory/(v+'.npz')) as z:crops.extend(z['fr_crop']);frames.extend(z['fr_frame']);offset+=len(z['lab_kind'])
    by={};labels=[];crops=np.asarray(crops);frames=np.asarray(frames)
    for i,l in enumerate(data.labels):
        if l[8]:continue
        e=l[0];by.setdefault(e,[]).append(i);v=vid[e];r=dict(label=f'{v}:{i-base[v]}',episode=e,pos=l[1],video=v,kind=l[2],pred=np.asarray(l[6]).tolist())
        assert np.array_equal(crops[data.off[e]+4+l[1]],l[7])
        if l[2]==1:r.update(gt=np.asarray(l[4]).tolist(),scale=float(max(l[5][2]-l[5][0],l[5][3]-l[5][1])))
        labels.append(r)
    return data,crops,frames,by,labels


def batch(torch,arm,bb,geom,eye_size,stats,lam,data,crops,frameids,by,ch):
    import numpy as np
    from scripts import z13_target_model as old
    from tracking.z13_target_center import ridge_scores,observed_center_target
    from tracking.z13_readout import to_frame
    from tracking.z13_candidates import candidates
    mx=[max(data.labels[i][1] for i in by[e]) for e in ch];dev=bb.dev if bb is not None else 'cpu'
    hx,ms=data.batch(torch,dev,ch,mx);F=old.episode_features(torch,arm,bb,hx,geom.neighbors,set(range(3,hx.shape[1])))
    storage={k:[] for k in ('episode','t','frame','count','centers','features','scores','crop_width','seed_center','seed_features')}
    for j,e in enumerate(ch):
        xe=(F[3][j].T-stats[0])/stats[1];mask=ms[j].cpu().numpy();seed=geom.points[mask].mean(0)
        seedc=to_frame(seed,crops[data.off[e]+3],eye_size);seedf=xe[mask].mean(0);ts=list(range(3,5+mx[j]))
        xs=np.stack([(F[t][j].T-stats[0])/stats[1] for t in ts]);y=observed_center_target(mask,geom.points,geom.spacing);maps=ridge_scores(xe,y,xs,[lam])[0]
        for si,t in enumerate(ts):
            r=candidates(maps[si],xs[si],geom);n=len(r['indices']);crop=crops[data.off[e]+t];cs=np.zeros((3,2));fs=np.zeros((3,xe.shape[1]));ss=np.zeros(3)
            cs[:n]=[to_frame(c,crop,eye_size) for c in r['centers']];fs[:n]=r['features'];ss[:n]=r['scores']
            for k,val in dict(episode=e,t=t,frame=frameids[data.off[e]+t],count=n,centers=cs,features=fs,scores=ss,crop_width=float(crop[2]-crop[0]),seed_center=seedc,seed_features=seedf).items():storage[k].append(val)
    return {k:np.asarray(v,dtype=np.int32 if k in ('episode','t','frame','count') else np.float64 if k in ('centers','scores') else np.float32) for k,v in storage.items()}


def extract(technical=False):
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tracking.gpu_lock import GpuLock
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    p=read(RUN/'protocol.json');verify(p);fit=Episodes(p['fit']);original=read(CACHE/'protocol.json')
    videos=original['evaluate'] if technical else p['videos'];raw=ROOT/'runs/z13_data' if technical else DEV/'data'
    data,crops,frames,by,labels=dataset(videos,raw);eps=sorted(by);repro={}
    if not technical:
        end=read(RUN/'technical_controller_result.json');assert end['worker_ended'] and end['exit_code']==0
        assert read(RUN/'technical.json')['passed'];write(RUN/'evaluate/labels.json',labels);assert len(labels)==9314
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    with GpuLock('flyvis_fovea_gpu'):
        for ai,arm in enumerate(p['arms']):
            folder=RUN/'evaluate'/arm
            if not technical and (folder/'done.json').exists():continue
            status('Нормировка прежних20 train',variant=arm,done=0,total=len(eps),arms_done=ai,arms_total=3)
            bank,geom,bb=old.build_arm(arm,torch,ck)
            if bb is not None:assert str(bb.dev).startswith('cuda')
            stats=old.feature_stats(torch,arm,bb,geom,fit);lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            if technical:
                arr=batch(torch,arm,bb,geom,bank.eye_size,stats,lam,data,crops,frames,by,eps[:8]);oldpath=CACHE/'evaluate'/arm/'cache_00000.npz'
                with np.load(oldpath) as z:
                    assert np.array_equal(arr['episode'],z['episode']) and np.array_equal(arr['t'],z['t']) and np.array_equal(arr['count'],z['count'])
                    dc=float(np.abs(arr['centers']-z['centers']).max());ds=float(np.abs(arr['scores']-z['scores']).max())
                assert dc<.5 and ds<.001 and lam==read(CACHE/'evaluate'/arm/'done.json')['lambda_value']
                repro[arm]=dict(max_center_px=dc,max_score=ds,lambda_value=lam)
            else:
                for left in range(0,len(eps),8):
                    path=folder/f'cache_{left:05d}.npz'
                    if path.exists():assert sha(path)==read(path.with_suffix('.json'))['sha256'];continue
                    status('Кандидаты на35dev без подбора',variant=arm,done=left,total=len(eps),arms_done=ai,arms_total=3)
                    ch=eps[left:left+8];arr=batch(torch,arm,bb,geom,bank.eye_size,stats,lam,data,crops,frames,by,ch)
                    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arr);tmp.replace(path)
                    write(path.with_suffix('.json'),dict(sha256=sha(path),episodes=ch,frames=len(arr['t'])))
                write(folder/'done.json',dict(ended=now().isoformat(),lambda_value=lam,batches=len(list(folder.glob('cache_*.npz')))))
            del bb,bank;gc.collect();torch.cuda.empty_cache()
    verify(p)
    if technical:write(RUN/'technical.json',dict(passed=True,arms=repro))


def evaluate():
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(2)
    import scripts.z13_candidate_identity_20261021 as loader
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);end=read(RUN/'cache_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    labels=read(RUN/'evaluate/labels.json');results={};seeds={};manifest={};repro={}
    for ai,arm in enumerate(p['arms']):
        status('Применение фиксированных3 моделей104',variant=arm,done=ai,total=3)
        oldroot=loader.CACHE
        try:loader.CACHE=RUN;seq,m=loader.load('evaluate',arm)
        finally:loader.CACHE=oldroot
        manifest.update(m);arrays=pack(labels,seq);logits=[];seeds[arm]={}
        for seed in p['seeds']:
            model=make_ranker(torch,False);model.load_state_dict(torch.load(SOURCE/arm/'geometry'/f'seed{seed}'/'model.pt',map_location='cpu',weights_only=True))
            pred=predict(torch,model,arrays);logits.append(pred);seeds[arm][str(seed)]=per_video_useful(make_rows(labels,arrays,pred))
        results[arm+'_map']=make_rows(labels,arrays,np.mean(logits,axis=0))
        results[arm+'_top1']=loader.rows(labels,seq,mode='top1');results[arm+'_nearest']=loader.rows(labels,seq,mode='kalman')
        saved=[r for v in p['videos'] for r in read(DEV/arm/v/'center_rows.json')];ref={r['label']:r for r in saved};assert len(ref)==len(labels)==9314
        dc=max(float(np.abs(np.asarray(r['center'])-ref[r['label']]['center']).max()) for r in results[arm+'_top1']);ds=max(abs(r['p_present']-ref[r['label']]['p_present']) for r in results[arm+'_top1'])
        assert dc<.5 and ds<.001
        lam=read(RUN/'evaluate'/arm/'done.json')['lambda_value'];assert lam==read(DEV/'protocol.json')['choices'][arm]['center']['lambda_value']
        assert lam==read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
        repro[arm]=dict(rows=len(ref),max_center_px=dc,max_score=ds,lambda_value=lam)
        for mode in ('map','top1','nearest'):write(RUN/arm/(mode+'_rows.json'),results[arm+'_'+mode])
    results['D_prior']=[r for v in p['videos'] for r in read(DEV/'D'/v/'prior_rows.json')]
    pv={k:per_video_useful(v) for k,v in results.items()};pairs=[]
    for arm in ('C','D'):pairs.extend([(arm+'_map',arm+'_top1'),(arm+'_map','R1_map'),(arm+'_map',arm+'_nearest'),(arm+'_map','D_prior')])
    pairs.append(('R1_map','R1_top1'));comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    a={v:pv['C_map'][v]-pv['C_top1'][v] for v in pv['C_top1']};b={v:pv['R1_map'][v]-pv['R1_top1'][v] for v in pv['R1_top1']};comp['C_gain-R1_gain']=paired(a,b)
    sg={s:paired(v,pv['C_top1']) for s,v in seeds['C'].items()};d=comp['C_map-C_top1'];passed=d['mean_per_video']>=.02 and d['ci95'][0]>0 and all(v['mean_per_video']>=0 for v in sg.values()) and all(comp[k]['ci95'][0]>0 for k in ('C_map-R1_map','C_map-C_nearest'))
    verify(p);write(RUN/'cache_manifest.json',manifest);write(RUN/'reproduction.json',repro);write(RUN/'seed_metrics.json',seeds)
    write(RUN/'report.json',dict(primary_pass=passed,paired=comp,C_seeds_vs_top1=sg,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in results.items()},limitations='Fixed transfer; reused dev35, no dev fitting or HOTA. Generic effect on R1 reported separately; no calibrated refusal.'))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',choices=['technical','cache','evaluate'],default='technical');a.add_argument('--budget',type=int,default=300);x=a.parse_args()
    if x.prepare:prepare();return
    if not x.worker:raise ValueError('Choose prepare/worker')
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='' if ATTEMPT=='evaluate' else '0';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:
        if ATTEMPT=='evaluate':evaluate()
        else:extract(ATTEMPT=='technical')
        write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Этап фиксированного переноса сохранён',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
