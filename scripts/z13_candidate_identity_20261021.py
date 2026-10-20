"""One bounded causal identity-readout pilot over frozen candidate caches."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,sha,verify,write,now,MSK
RUN=ROOT/'runs/z13_candidate_identity_20261021';CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
IDENT='z13_candidate_identity_v1';T0=time.monotonic();LIMIT=1800;ATTEMPT='full'


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    p=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=p.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    p.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing pilot must be inspected')
    reserve=read(CACHE/'report.json');assert reserve['C_reserve_sufficient']
    p=dict(id=IDENT,created=now().isoformat(),authorization='Autonomous necessary runs, repairs and publication authorized by author until07:40 MSK21.10.2026.',
        question='Does causal propagation of candidate identities improve the centered FlyVis readout, beyond geometric matching and the same node on R1?',
        action='Replace tallest-peak choice by a target identity carried through one-to-one candidate assignments. Fixed core, same3 candidates and all available episode frames. Missing assignment returns no correction.',
        method='Handcrafted alpha*cosine-distance+(1-alpha)*frame-distance/crop-width. Assignment by Hungarian algorithm, per-object missing option. Seed from last observed box and activity. Last target retained for redetection; unmatched distractors expire. No future data. Not a KeepTrack GNN reproduction.',
        fit='Same20 train only. Appearance alpha=[0.25,0.5,0.75,1.0], missing_cost=[0.25,0.5,0.75]; geometry control alpha=0, same3 costs. Choose highest per-video mean useful fraction on fit20; deterministic grid order breaks ties. No parameter changes after train26.',
        primary='C association-C top1: point gain>=0.02 and paired-video95% lower>0. C association-C geometry and C association-R1 association: both lower>0. All three required; D secondary, not a rescue.',
        secondary='D vs its top1/geometry, D vs R1/C; nearest Kalman among the same3; selected known nonuseful and unknown separately at fixed1/2/5%. Confidence is matching margin, not probability. Rejection reduces coverage and is not automatically safety.',
        counterargument='Three-peak oracle reserve need not be accessible from appearance/history; identity can drift, persist on a wrong seed, or be lost through candidate disappearance.',
        decision='If main criteria fail, no deployment or dev tuning. Inspect which of one-to-one identity versus query-time candidate reading needs replacement, without a causal claim from failure alone.',
        budget_seconds=1800,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/2103.16556v2 sections3.1-3.7; our small matching implementation differs from learned GNN/Sinkhorn KeepTrack.',
        sha256={**read(CACHE/'protocol.json')['sha256'],**{r:sha(ROOT/r) for r in ['tracking/z13_candidate_identity.py','scripts/z13_candidate_identity_20261021.py','scripts/z13_night_control_20261021.py','tests/test_z13_candidate_identity.py','runs/z13_candidate_reserve_20261021/report.json']}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: покадровое сопоставление кандидатов без будущих данных',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','KeepTrack','engineering-future'],
        question=p['question'],hypothesis='Сохранение идентичности отдельных пиков может использовать часть измеренного запаса; перенос принципа KeepTrack требует проверки.',expected='Одна фиксированная версия по текущим и прошлым кадрам, C/D/R1 и геометрические контроли; параметры только по fit20, проверка на всех train26, CPU до1800с.',
        acceptance=[dict(criterion='Воспроизведение прежнего top1, хеши и одинаковые строки всех вариантов подтверждены.',met=None),dict(criterion='C сопоставление−C top1: прирост>=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='C сопоставление лучше C геометрии и R1 сопоставления: обе нижние95%повидеоДИ>0.',met=None)],
        summary='Зафиксирована одна версия сопоставления кандидатов, исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Нет HOTA и безопасного причинного правила вмешательства; верхняя доля по всей выборке — диагностика. Это не полная реализация KeepTrack и не доказательство анатомии FlyVis.',metrics=[],commits=[],files=['tracking/z13_candidate_identity.py','scripts/z13_candidate_identity_20261021.py'],evidence=[dict(kind='file',ref='runs/z13_candidate_identity_20261021/protocol.json',claim='Постановка до результатов'),dict(kind='file',ref='runs/z13_candidate_reserve_20261021/report.json',claim='Измеренный диагностический запас кандидатов')]))
    write(cp,cat);journal('постановка временного сопоставления кандидатов','*Постановка.* '+p['action']+'\n\n'+p['fit']+'\n\nКритерии до исходов: '+p['primary']+'\n\nНадёжность отказа и HOTA не следуют из этого пилота. Функциональный CPU-этап до1800с, без нового ядра и без настройки на dev.')


def load(split,arm):
    import numpy as np
    out={};manifest={}
    folder=CACHE/split/arm;assert (folder/'done.json').exists()
    for path in sorted(folder.glob('cache_*.npz')):
        digest=sha(path);assert digest==read(path.with_suffix('.json'))['sha256'];manifest[path.relative_to(ROOT).as_posix()]=digest
        with np.load(path) as archive:
            z={k:archive[k] for k in archive.files}
            for i,e in enumerate(z['episode']):
                n=int(z['count'][i]);out.setdefault(int(e),[]).append(dict(t=int(z['t'][i]),centers=z['centers'][i,:n].copy(),features=z['features'][i,:n].copy(),scores=z['scores'][i,:n].copy(),width=float(z['crop_width'][i]),seed_center=z['seed_center'][i].copy(),seed_features=z['seed_features'][i].copy()))
    for frames in out.values():frames.sort(key=lambda f:f['t']);assert [f['t'] for f in frames]==list(range(3,frames[-1]['t']+1))
    return out,manifest


def select(sequences,alpha,tau):
    from tracking.z13_candidate_identity import IdentityReadout
    answers={}
    for e,frames in sequences.items():
        check();seed=frames[0];r=IdentityReadout(seed['seed_center'],seed['seed_features'],alpha,tau)
        for f in frames:
            j,confidence=r.step(f['centers'],f['features'],f['width']);answers[(e,f['t'])]=(j,confidence)
    return answers


def rows(labels,sequences,answers=None,mode='association'):
    import numpy as np
    lookup={(e,f['t']):f for e,frames in sequences.items() for f in frames};out=[]
    for l in labels:
        f=lookup[(l['episode'],l['pos']+4)]
        if mode=='top1':j,present=0,float(f['scores'][0])
        elif mode=='kalman':j=int(np.argmin(np.linalg.norm(f['centers']-l['pred'],axis=1)));present=float(f['scores'][j])
        else:j,present=answers[(l['episode'],l['pos']+4)]
        r={k:l[k] for k in ('video','label','kind','pos','pred')};center=f['centers'][j] if j>=0 else np.asarray(l['pred'])
        r.update(center=center.tolist(),p_present=present if j>=0 else -1.,intervene=j>=0,candidate=j)
        if l['kind']==1:
            err=float(np.linalg.norm(center-l['gt']));ep=float(np.linalg.norm(np.asarray(l['pred'])-l['gt']))
            r.update(err=err,err_pred=ep,scale=l['scale'],useful=bool(j>=0 and err<=.25*l['scale'] and err<ep))
        elif l['kind']==0:r['useful']=False
        out.append(r)
    return out


def worker():
    import numpy as np
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);result={};choices={};manifests={}
    for ai,arm in enumerate(('C','D','R1')):
        folder=RUN/arm
        if (folder/'done.json').exists():
            choices[arm]=read(folder/'choice.json')
            manifests.update(read(folder/'cache_manifest.json'))
            for mode in ('association','geometry','top1','kalman'):result[arm+'_'+mode]=read(folder/(mode+'_rows.json'))
            continue
        sequences,m=load('fit',arm);manifests.update(m);labels=read(CACHE/'fit/labels.json');choice={}
        for mode,alphas in [('association',[.25,.5,.75,1.]),('geometry',[0.])]:
            grid=[]
            for alpha in alphas:
                for tau in (.25,.5,.75):
                    status('Подбор сопоставления на прежних20 train',variant=arm+' '+mode,done=len(grid),total=len(alphas)*3,arms_done=ai,arms_total=3)
                    answer=select(sequences,alpha,tau);rr=rows(labels,sequences,answer);score=float(np.mean(list(per_video_useful(rr).values())))
                    grid.append(dict(alpha=alpha,missing_cost=tau,fit_score=score))
            best=max(grid,key=lambda x:x['fit_score']);choice[mode]=dict(best=best,grid=grid)
        write(folder/'choice.json',choice);choices[arm]=choice
        sequences,m=load('evaluate',arm);manifests.update(m);labels=read(CACHE/'evaluate/labels.json')
        status('Оценка причинного выбора на всех26 train',variant=arm,done=0,total=4,arms_done=ai,arms_total=3)
        for mode in ('association','geometry','top1','kalman'):
            if mode in choice:
                best=choice[mode]['best'];answer=select(sequences,best['alpha'],best['missing_cost'])
            else:answer=None
            rr=rows(labels,sequences,answer,mode);result[arm+'_'+mode]=rr;write(folder/(mode+'_rows.json'),rr)
        saved=read(CACHE/f'evaluate/{arm}/top1_rows.json');old={r['label']:r for r in saved};rr=result[arm+'_top1']
        assert len(rr)==len(old)==8326
        dc=max(float(np.abs(np.asarray(r['center'])-old[r['label']]['center']).max()) for r in rr)
        ds=max(abs(r['p_present']-old[r['label']]['p_present']) for r in rr)
        assert dc<1e-12 and ds<1e-12
        write(folder/'reproduction.json',dict(rows=len(rr),max_center_px=dc,max_score=ds))
        write(folder/'cache_manifest.json',{k:v for k,v in manifests.items() if f'/{arm}/' in k})
        write(folder/'done.json',dict(ended=now().isoformat()))
    pv={name:per_video_useful(rr) for name,rr in result.items()}
    pairs=[('C_association','C_top1'),('C_association','C_geometry'),('C_association','R1_association'),('D_association','D_top1'),('D_association','D_geometry'),('D_association','R1_association'),('D_association','C_association'),('C_association','C_kalman'),('D_association','D_kalman')]
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    c=comp['C_association-C_top1'];primary=c['mean_per_video']>=.02 and c['ci95'][0]>0 and comp['C_association-C_geometry']['ci95'][0]>0 and comp['C_association-R1_association']['ci95'][0]>0
    verify(p);write(RUN/'cache_manifest.json',manifests)
    write(RUN/'report.json',dict(primary_pass=primary,paired=comp,choices=choices,summary={name:dict(rows=len(rr),useful_mean_video=float(np.mean(list(pv[name].values()))),interventions=sum(r['intervene'] for r in rr),coverage=coverage_table(rr)) for name,rr in result.items()},limitations='Exploratory reused train26; fixed candidate set, handcrafted association, no HOTA or safe intervention claim.'))


def main():
    global ATTEMPT,LIMIT
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');ap.add_argument('--attempt',default='full');ap.add_argument('--budget',type=int,default=1800);a=ap.parse_args()
    if a.prepare:prepare();return
    ATTEMPT=a.attempt;LIMIT=a.budget
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    if not a.worker:raise ValueError('Choose prepare/worker')
    try:worker();write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Причинный пилот сохранён',updated=now().isoformat()))
    except Exception as exc:
        write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
