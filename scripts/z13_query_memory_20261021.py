"""Separate observed ridge models, matched fixed/query-conditioned fusion."""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_query_memory_20261021';IDENT='z13_query_memory_v1'
ATTEMPT='full';LIMIT=2400;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Run exists; preserve and inspect')
    base=read(ROOT/'runs/z13_observed_memory_20261021/protocol.json');verify(base)
    names=['tracking/z13_query_memory.py','scripts/z13_query_memory_20261021.py','tests/test_z13_query_memory.py','scripts/z13_night_control_20261021.py','runs/z13_observed_memory_20261021/replay_report.json']
    names.extend(f'runs/z13_observed_memory_20261021/observations/{v}.npz' for v in base['fit']+base['evaluate'])
    p=dict(id=IDENT,created=now().isoformat(),authorization='Autonomous necessary runs/repairs/publication until07:40 MSK21.10.2026 explicitly authorized.',
        fit=base['fit'],evaluate=base['evaluate'],arms=['C','D','R1'],
        action='Replace one jointly fitted ridge memory by separate models for distinct observed frames1/2/3 and query-conditioned reading; matched uniform mixture of the identical expert maps is the key control.',
        method='Each distinct observed feature/mask map fits its own Gaussian-center ridge expert at the saved single-frame lambda. Query weight at each column: softmax over expert response standardized per expert map, divided by temperature. Output=(1-eta)*last_map+eta*weighted_expert_maps. Fixed-memory control substitutes uniform weights. No query update of models and no future GT.',
        fit_grid='Fixed: eta=[0,0.25,0.5,1]. Query: eta0 at temperature1, then eta=[0.25,0.5,1] x temperature=[0.25,1,4]. Same original20 train only, maximize per-video useful fraction; grid order breaks ties, baseline wins exact ties. Original scalar lambda unchanged. All fitted choices evaluated on full26; no retuning.',
        primary='C query-C last: mean>=0.02 and lower95%paired-videoCI>0. C query-C fixed and C query-R1 query: both lower>0. All criteria required. D same contrasts and C query-D query are secondary, not rescue.',
        reproduction='Last expert exactly original single observed map; each arm8326 rows, center<0.5px,score<0.001; same original lambda, scaler, temporal history. All source and restored observed-mask SHA256 checked.',
        counterargument='Earlier observed activities may be less compatible with current state; query selection can amplify a wrong expert. Separate fixed-mixture control separates query dependence from added model capacity.',
        decision='If main criteria fail, keep original centered readout; no claim all memory fails. No core training, downloads,dev,HOTA orheldout75. Previous three-peak oracle is not a ceiling for these new full response maps.',
        budget_seconds=2400,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/2104.00324v1 sections3.1-3.3: read historical representations conditioned on current query. Our ridge-expert mixture is not full STMTrack.',
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: чтение отдельных моделей памяти по текущему запросу',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','STMTrack','engineering-future'],
        question='Помогает ли зависимый от текущего запроса выбор между отдельными моделями прошлых наблюдений сверх последнего образца и равномерного смешивания тех же моделей?',
        hypothesis='Раздельное сохранение моделей и выбор по текущему отклику может использовать прошлые наблюдения иначе, чем испытанное совместное обучение одной модели.',
        expected='Те же C/D/R1 и fit20/train26. Отдельные модели1–3, фиксированное и зависимое от запроса смешивание; допуски прежние, ядро не обучается. GPU до2400с, по одному процессу.',
        acceptance=[dict(criterion='Контроль last всех3 вариантов воспроизведён на8326 строках: центр<0,5px,балл<0,001; исходники/наблюдаемые маски неизменны.',met=None),dict(criterion='C query−C last: средний сдвиг>=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='C query лучше C fixed и такого же R1 query: обе нижние95%повидеоДИ>0.',met=None)],
        summary='Зафиксирована одна версия чтения раздельных моделей памяти; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Не полная STMTrack, не HOTA и не безопасное вмешательство. Параметры выбираются только по fit20. Оракул трёх прежних пиков не задаёт потолок новых карт отклика.',metrics=[],commits=[],files=names[:2],evidence=[dict(kind='file',ref='runs/z13_query_memory_20261021/protocol.json',claim='Постановка до исходов'),dict(kind='file',ref='runs/z13_observed_memory_20261021/report.json',claim='Прежняя совместная модель памяти не показала прироста')]))
    write(cp,cat);journal('постановка чтения раздельных моделей памяти','*Постановка.* '+p['action']+'\n\n'+p['fit_grid']+'\n\nКритерии до исходов: '+p['primary']+'\n\nПервая память объединяла все наблюдения в одну регрессию; здесь меняется чтение отдельных моделей. Это отдельная проверка механизма, не подбор настроек неудачного селектора по train26. GPU до2400с, один процесс; ядро фиксировано, dev/HOTA/75 не используются.')


def extract(torch,arm,bb,bank,geom,data,stats,lam,split):
    import numpy as np
    from scripts.z13_target_model import episode_features
    from tracking.z13_query_memory import expert_maps
    by_ep={};videos=[];masks=[];frames=[]
    for v,n in data.meta:
        videos.extend([v]*n)
        with np.load(ROOT/f'runs/z13_observed_memory_20261021/observations/{v}.npz') as a:masks.extend(a['masks']);frames.extend(a['frames'])
    for i,l in enumerate(data.labels):
        if not l[8]:by_ep.setdefault(l[0],[]).append(i)
    eps=sorted(by_ep);folder=RUN/arm/split;folder.mkdir(parents=True,exist_ok=True)
    labels=[]
    for e,ids in by_ep.items():
        for i in ids:
            l=data.labels[i];r=dict(label=i,video=videos[e],kind=l[2],pos=l[1],crop=np.asarray(l[7]).tolist(),pred=np.asarray(l[6]).tolist())
            if l[2]==1:r.update(gt=np.asarray(l[4]).tolist(),scale=float(max(l[5][2]-l[5][0],l[5][3]-l[5][1])))
            labels.append(r)
    write(folder/'labels.json',labels);dev=bb.dev if bb is not None else 'cpu'
    for left in range(0,len(eps),8):
        path=folder/f'maps_{left:05d}.npz'
        if path.exists():continue
        status('Отдельные модели наблюдений: '+split,variant=arm,done=left,total=len(eps))
        ch=eps[left:left+8];mx=[max(data.labels[i][1] for i in by_ep[e]) for e in ch];hx,ms=data.batch(torch,dev,ch,mx)
        need={1,2,3}|{4+data.labels[i][1] for e in ch for i in by_ep[e]};F=episode_features(torch,arm,bb,hx,geom.neighbors,need)
        scores=[];counts=[];idxs=[]
        for j,e in enumerate(ch):
            observed=np.stack([np.zeros_like(F[1][j].T),*[F[t][j].T for t in (1,2,3)]]);observed=(observed-stats[0])/stats[1]
            ids=by_ep[e];xs=np.stack([(F[4+data.labels[i][1]][j].T-stats[0])/stats[1] for i in ids])
            assert np.array_equal(masks[e][3],ms[j].cpu().numpy())
            maps,slots=expert_maps(observed,np.asarray(masks[e]),xs,frames[e],geom.points,geom.spacing,lam)
            for si,i in enumerate(ids):
                z=np.zeros((3,721),np.float64);z[:len(slots)]=maps[:,si];scores.append(z);counts.append(len(slots));idxs.append(i)
        temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,scores=np.asarray(scores),counts=np.asarray(counts),labels=np.asarray(idxs));temp.replace(path)
        write(path.with_suffix('.json'),dict(sha256=sha(path),episodes=ch));del F,hx,ms
    write(folder/'done.json',dict(ended=now().isoformat()))


def load_maps(folder):
    import numpy as np
    out={};manifest={}
    for path in sorted(folder.glob('maps_*.npz')):
        digest=sha(path);assert digest==read(path.with_suffix('.json'))['sha256'];manifest[path.relative_to(ROOT).as_posix()]=digest
        with np.load(path) as a:s=a['scores'];n=a['counts'];ids=a['labels']
        for i,label in enumerate(ids):out[int(label)]=s[i,:n[i]].copy()
    return out,manifest


def evaluate(maps,labels,geom,eye_size,mode,eta=0.,temperature=1.):
    import numpy as np
    from tracking.z13_query_memory import fuse
    from scripts.z13_target_model import center_and_presence
    from tracking.z13_readout import to_frame
    out=[]
    for l in labels:
        check();score=fuse(maps[l['label']],mode,eta,temperature);ce,pres=center_and_presence(score,geom);c=np.asarray(to_frame(ce,l['crop'],eye_size))
        r={k:l[k] for k in ('label','video','kind','pos','pred')};r.update(center=c.tolist(),p_present=pres)
        if l['kind']==1:
            err=float(np.linalg.norm(c-l['gt']));ep=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']));r.update(err=err,err_pred=ep,scale=l['scale'],useful=bool(err<=.25*l['scale'] and err<ep))
        elif l['kind']==0:r['useful']=False
        out.append(r)
    return out


def worker():
    import gc
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tracking.gpu_lock import GpuLock
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);fit=Episodes(p['fit']);ev=Episodes(p['evaluate']);allrows={};choices={};manifest={}
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False)
    with GpuLock('flyvis_fovea_gpu'):
        for ai,arm in enumerate(p['arms']):
            folder=RUN/arm
            if (folder/'done.json').exists():
                choices[arm]=read(folder/'choice.json');manifest.update(read(folder/'cache_manifest.json'))
                for mode in ('last','fixed','query'):allrows[arm+'_'+mode]=read(folder/(mode+'_rows.json'))
                continue
            status('Нормировка прежних признаков',variant=arm,done=0,total=3,arms_done=ai,arms_total=3)
            bank,geom,bb=old.build_arm(arm,torch,ck)
            if bb is not None and not str(bb.dev).startswith('cuda'):raise RuntimeError('Frozen-core extraction requires CUDA')
            stats=old.feature_stats(torch,arm,bb,geom,fit);lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
            extract(torch,arm,bb,bank,geom,fit,stats,lam,'fit');maps,m=load_maps(folder/'fit');manifest.update(m);labels=read(folder/'fit/labels.json');choice={}
            for mode in ('fixed','query'):
                params=[(e,1.) for e in (0.,.25,.5,1.)] if mode=='fixed' else [(0.,1.)]+[(e,t) for e in (.25,.5,1.) for t in (.25,1.,4.)]
                grid=[]
                for i,(eta,temp) in enumerate(params):
                    status('Выбор чтения памяти на fit20',variant=arm+' '+mode,done=i,total=len(params))
                    rr=evaluate(maps,labels,geom,bank.eye_size,mode,eta,temp);score=float(np.mean(list(per_video_useful(rr).values())));grid.append(dict(eta=eta,temperature=temp,fit_score=score))
                choice[mode]=dict(best=max(grid,key=lambda x:x['fit_score']),grid=grid)
            write(folder/'choice.json',choice);del maps
            extract(torch,arm,bb,bank,geom,ev,stats,lam,'evaluate');maps,m=load_maps(folder/'evaluate');manifest.update(m);labels=read(folder/'evaluate/labels.json')
            for mode in ('last','fixed','query'):
                best=choice[mode]['best'] if mode in choice else dict(eta=0.,temperature=1.)
                rr=evaluate(maps,labels,geom,bank.eye_size,mode,best['eta'],best['temperature']);allrows[arm+'_'+mode]=rr;write(folder/(mode+'_rows.json'),rr)
            saved={r['label']:r for r in read(ROOT/f'runs/z13_observed_memory_20261021/{arm}/single_rows.json')};rr=allrows[arm+'_last'];assert len(rr)==len(saved)==8326
            dc=max(float(np.abs(np.asarray(r['center'])-saved[r['label']]['center']).max()) for r in rr);ds=max(abs(r['p_present']-saved[r['label']]['p_present']) for r in rr);assert dc<.5 and ds<.001
            write(folder/'reproduction.json',dict(rows=len(rr),max_center_px=dc,max_score=ds,lambda_value=lam));write(folder/'cache_manifest.json',{k:v for k,v in manifest.items() if f'/{arm}/' in k});write(folder/'done.json',dict(ended=now().isoformat()));choices[arm]=choice
            del bb,bank,maps;gc.collect();torch.cuda.empty_cache()
    pv={k:per_video_useful(v) for k,v in allrows.items()};pairs=[]
    for arm in ('C','D'):pairs.extend([(arm+'_query',arm+'_'+b) for b in ('last','fixed')]);pairs.append((arm+'_query','R1_query'))
    pairs.append(('D_query','C_query'));comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs};c=comp['C_query-C_last'];passed=c['mean_per_video']>=.02 and c['ci95'][0]>0 and comp['C_query-C_fixed']['ci95'][0]>0 and comp['C_query-R1_query']['ci95'][0]>0
    verify(p);write(RUN/'cache_manifest.json',manifest);write(RUN/'report.json',dict(primary_pass=passed,paired=comp,choices=choices,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in allrows.items()},limitations='Exploratory reused train26; fixed core and saved lambda; one fit20 grid; no HOTA, no calibrated refusal; old3-peak oracle not a ceiling for new maps.'))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',default='full');a.add_argument('--budget',type=int,default=2400);x=a.parse_args()
    if x.prepare:prepare();return
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    if not x.worker:raise ValueError('Choose prepare/worker')
    try:worker();write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Чтение моделей памяти сохранено',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
