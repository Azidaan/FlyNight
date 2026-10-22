"""Bounded sampling-only intervention. Original labels, frozen caches/core."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import gc
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha
MSK = timezone(timedelta(hours=3))
RUN = ROOT/'runs/z13_semantic_sampling_20261023'
CACHE = ROOT/'runs/z13_candidate_reserve_20261021'
OLD = ROOT/'runs/z13_candidate_data104_20261021'
IDENT = 'z13_semantic_sampling_v1'
ARMS = ('D', 'C', 'R1')
T0 = time.monotonic()


def now():
    return datetime.now(MSK)


def journal(title, text):
    path = ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md'
    before = path.read_bytes()
    nl = b'\r\n' if b'\r\n' in before else b'\n'
    marker = '## Журнал изменений'.encode()+nl+nl
    assert before.count(marker) == 1
    entry = (f'**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n\n').replace('\n', nl.decode()).encode()
    path.write_bytes(before.replace(marker, marker+entry, 1))


def status(stage, **kw):
    p = read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0 >= p['budget_seconds'] or now() >= datetime.fromisoformat(p['save_at']):
        raise TimeoutError('Original CPU budget or Moscow deadline reached')
    write(RUN/'status.json', dict(state='running', stage=stage, updated=now().isoformat(),
          elapsed_seconds=round(time.monotonic()-T0, 2), deadline=p['hard_stop'], **kw))


def verify_sources(p):
    for name, digest in p['sha256'].items():
        assert sha(ROOT/name) == digest, name


def prepare():
    assert not RUN.exists()
    from scripts.z13_localization_labels_20261023 import RUN as BASE
    base=read(BASE/'protocol.json');verify_sources(base)
    names=['tracking/z13_semantic_sampling.py','scripts/z13_semantic_sampling_20261023.py','scripts/z13_semantic_audit_20261023.py','scripts/z13_data.py','tests/test_z13_semantic_sampling.py','scripts/z13_semantic_sampling_console.ps1']
    old=read(OLD/'protocol.json')
    for video in base['train']:
        names.extend([f'runs/z13_data/{video}.npz',f'runs/train_cache/{video}/alignment.json',f'data/bdd100k/labels/box_track_20/train/{video}.json'])
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0);assert now()<deadline
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author: autonomous necessary experiments until07:40MSK23.10.2026.',device='cpu',budget_seconds=3600,save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),arms=list(ARMS),seeds=[0,1,2],train=base['train'],evaluate=base['evaluate'],
      action='Only training row sampling changes: half uniform, half semantic-hard rows. Original utility labels, class weights, architecture, random initialization,1150steps and145950row presentations remain fixed. Same sampled row stream for all arms. Old18models reused.',
      hard_definition='Training104 only. Candidate center uniquely covered by one annotated box of another identity but the same category as target; no overlapping boxes, unknown kind excluded, old useful positives excluded. Union of hard rows across D/C/R1 is the COMMON sampling pool. Current GT is used only for training sampling, never evaluation inference. No new negative labels or losses.',
      metadata='Video/local episode identity and frame from existing episode NPZ; original annotations via saved alignment. Every known-present target center must agree within0.001px. All new sources hashed. No interpolation or fabricated distractor IDs.',
      gates='Nonempty D hard rows and common pool, correct identity/frame binding, unchanged old labels; all18old models replayed;32old steps reproduced; conservative18model estimate fits3600s.',
      primary='Dnew activity-Dold activity >=0.02 AND paired-video95%lower>0; Dnew beats Dold geometry and R1new activity with lower95>0. All required; C cannot rescue.',
      secondary='All26videos,three individual seeds, geometry ablations, top1/Kalman, fixed1/2/5percent confidence quotas with unknown and strict worse separately. No efficacy-driven parameter or seed selection.',
      counterargument='Box containment is only a conservative semantic proxy: a peak can represent overlapping texture rather than an object. Rare hard rows can be oversampled excessively and displace useful easy rows. Fixed50percent is one prespecified test, not a tuned optimum. This changes data frequency, not representation capacity.',
      paper='https://arxiv.org/html/1808.06048v1 section3.2, semantic same-category negatives. Adaptation of sampling principle, not DaSiamRPN reproduction or transferred magnitude.',
      sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучение выборщика на похожих размеченных соседях',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','engineering-future','DaSiamRPN'],question='Поможет ли увеличение частоты похожих соседей при неизменном считывателе и бюджете?',hypothesis='Обычные отрицательные примеры могут недостаточно обучать различению экземпляров одного класса; проверяется только частота таких примеров.',expected='D/C/R1,18моделей,3сида,104train и26оценки;3600с CPU+20с завершения, до07:40. Один общий набор трудных строк; только выбор строк меняется.',acceptance=[dict(criterion='Мишень непустая, исходная разметка связана с кадрами/идентичностями; старые метки,18контролей,32шага и бюджет воспроизведены,хеши проверены.',met=None),dict(criterion='Dnew-Dold activity >=2п.п.,нижняя95%повидеоДИ>0.',met=None),dict(criterion='Dnew лучше Dold geometry и R1new: обе нижние95%ДИ>0.',met=None)],summary='Протокол до исходов: половина пакета из общего набора строк с похожими соседями.',done='',obtained='',interpretation='',decision='',cannot_claim='Поисковый train26, не HOTA и не безопасный отказ. Нахождение пика внутри рамки не доказывает, что его активность принадлежит соседу. Не полная DaSiamRPN; эффект частоты не доказывает новизну.',metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref='runs/z13_semantic_sampling_20261023/protocol.json',claim='Критерии, единый бюджет, мишень и хеши до обучения')]))
    write(cp,cat)
    journal('постановка проверки похожих отрицательных примеров','*Постановка.* По DaSiamRPN §3.2 меняется только частота строк обучения: половина пакета из общего для D/C/R1 набора с пиком внутри единственной рамки другого объекта того же класса. Неизвестные и прежние положительные метки исключены из мишени; метки вообще не меняются. Сначала исходная разметка/кадры/идентичности и непустой набор, затем18равнобюджетных моделей. Протокол `runs/z13_semantic_sampling_20261023/protocol.json`. Критерии до результатов: D против старого чтения >=2п.п. с положительным ДИ, также превосходство над картой D и входом глаза R1. CPU3600с, своё дерево до07:40. Нет GPU/ядра/dev/HOTA/75.')


def train_steps(torch, model, tensors, nref, seed, weight, maxsteps=None, progress=None, hard=None):
    from tracking.z13_matched_data_budget import matched_batches
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]; presented=0; model.train()
    from tracking.z13_semantic_sampling import semantic_batches
    batches=matched_batches(len(tensors[0]),nref,50,128,seed) if hard is None else semantic_batches(len(tensors[0]),nref,seed,hard)
    for step, indices in enumerate(batches):
        if maxsteps is not None and step>=maxsteps: break
        if progress is not None and step%23==0: progress(step)
        ix=torch.as_tensor(indices,device='cpu');x,s,g,mask,target=[t[ix] for t in tensors]
        score=model(x,s,g,mask)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(score,target,pos_weight=torch.tensor(weight,device='cpu'),reduction='none')
        loss=(loss*mask).sum()/mask.sum(); assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        losses.append(float(loss.detach()));presented+=len(indices)
    return losses,presented


def worker():
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(2)
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_semantic_audit_20261023 import metadata,audit as semantic_audit
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json'); verify_sources(p)
    labels20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');labels=labels20+extra;ev=read(CACHE/'evaluate/labels.json')
    assert {l['video'] for l in labels}<=set(p['train']) and {l['video'] for l in ev}<=set(p['evaluate'])
    ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in labels20)
    assert nref==2919 and len(ids)==20778
    base=read(OLD/'protocol.json');m1,h1,c1=metadata(base['fit'],labels20);m2,h2,c2=metadata(base['evaluate'],extra);meta=m1+m2
    arrays={};audit={};replay={};manifest={**h1,**h2};result={};seedpv={};hard_union=np.zeros(len(labels),bool)
    for ai,arm in enumerate(ARMS):
        status('Проверка похожих соседей и прежних моделей',variant=arm,done=ai,total=3,models_done=0,models_total=18)
        seq,m=reader.load('fit',arm);manifest.update(m);first=pack(labels20,seq)
        prev=reader.CACHE
        try:
            reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally: reader.CACHE=prev
        manifest.update(m);last=pack(extra,seq);ta=tuple(np.concatenate([a,b]) for a,b in zip(first,last))
        seq,m=reader.load('evaluate',arm);manifest.update(m);ea=pack(ev,seq)
        newy=ta[4];negative,counts=semantic_audit(labels,ta[5],ta[3],meta)
        assert not np.any(negative & (newy>0));hard_union|=negative.any(1)
        audit[arm]=counts
        arrays[arm]=(ta,ea,newy)
        for mode,activity in [('activity',True),('geometry',False)]:
            preds=[]
            for seed in p['seeds']:
                sd=OLD/arm/mode/f'seed{seed}';tr=read(sd/'training.json');assert sha(sd/'model.pt')==tr['model_sha256']
                model=make_ranker(torch,activity);model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True));preds.append(predict(torch,model,ea))
            fresh=make_rows(ev,ea,np.mean(preds,axis=0));saved=read(OLD/arm/mode/'ensemble_rows.json');assert len(fresh)==len(saved)==len(ev)
            delta=max(abs(a['p_present']-b['p_present']) for a,b in zip(fresh,saved))
            assert delta<=1e-5
            assert all(a['candidate']==b['candidate'] and a['center']==b['center'] and a.get('useful')==b.get('useful') and a['video']==b['video'] and a['label']==b['label'] and a['pos']==b['pos'] for a,b in zip(fresh,saved))
            replay[arm+'_'+mode]=dict(rows=len(saved),all_candidates_centers_outcomes_equal=True,max_abs_score=delta)
            result[arm+'_old_'+mode]=saved
        del seq,first,last;gc.collect()
    write(RUN/'semantic_audit.json',dict(arms=audit,gt_centers_checked=c1+c2,common_hard_rows=int(hard_union[ids].sum()),common_indices=np.flatnonzero(hard_union[ids]).tolist()));write(RUN/'reproduction.json',replay);write(RUN/'cache_manifest.json',manifest)
    assert audit['D']['hard_rows']>0, 'No semantic target for intervention'
    hard=np.flatnonzero(hard_union[ids]);assert len(hard)>0
    status('Проверка прежней траектории обучения и стоимости',variant='D activity seed0',done=0,total=32,models_done=0,models_total=18)
    oldta=arrays['D'][0];tensors=[torch.as_tensor(x[ids],device='cpu') for x in oldta[:5]]
    tr=read(OLD/'D/activity/seed0/training.json');torch.manual_seed(0);probe=make_ranker(torch,True);beg=time.monotonic()
    losses,_=train_steps(torch,probe,tensors,nref,0,tr['pos_weight'],maxsteps=32)
    seconds=time.monotonic()-beg; delta=max(abs(a-b) for a,b in zip(losses,tr['loss'][:32]));assert delta<=1e-6,delta
    projected=seconds/32*1150*18*2+120
    write(RUN/'timing.json',dict(probe_steps=32,seconds=seconds,old_loss_max_abs=delta,conservative_remaining_seconds=projected,budget_seconds=p['budget_seconds']))
    assert projected<p['budget_seconds']-(time.monotonic()-T0), 'Measured work does not fit approved bounded pilot'
    del tensors,probe;gc.collect(); models_done=0
    for arm in ARMS:
        ta,ea,newy=arrays[arm];newta=(*ta[:4],newy);tensors=[torch.as_tensor(x[ids],device='cpu') for x in newta]
        for mode,activity in [('activity',True),('geometry',False)]:
            preds=[];key=arm+'_new_'+mode;seedpv[key]={}
            for seed in p['seeds']:
                sd=RUN/arm/mode/f'seed{seed}';ref=read(OLD/arm/mode/f'seed{seed}/training.json')
                assert ref['steps']==1150 and ref['presented_rows']==145950 and ref['reference_known_rows']==nref
                torch.manual_seed(seed);model=make_ranker(torch,activity)
                if (sd/'done.json').exists():
                    assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                else:
                    def progress(step):
                        status('Обучение на похожих соседях при прежнем бюджете',variant=f'{arm} {mode} seed{seed}',done=step,total=1150,models_done=models_done,models_total=18)
                    losses,presented=train_steps(torch,model,tensors,nref,seed,ref['pos_weight'],progress=progress,hard=hard)
                    assert len(losses)==1150 and presented==145950
                    sd.mkdir(parents=True,exist_ok=True);tmp=sd/'model.tmp.pt';torch.save(model.state_dict(),tmp);tmp.replace(sd/'model.pt')
                    write(sd/'training.json',dict(seed=seed,steps=len(losses),presented_rows=presented,pos_weight=ref['pos_weight'],loss=losses,model_sha256=sha(sd/'model.pt')))
                    write(sd/'done.json',dict(ended=now().isoformat()))
                pred=predict(torch,model,ea);preds.append(pred);seedpv[key][str(seed)]=per_video_useful(make_rows(ev,ea,pred));models_done+=1
            rr=make_rows(ev,ea,np.mean(preds,axis=0));result[key]=rr;write(RUN/arm/mode/'ensemble_rows.json',rr)
        del tensors;gc.collect()
    status('Парные сравнения по всем26 видео',models_done=18,models_total=18)
    for arm in ARMS:
        for b in ('top1','kalman'):result[arm+'_'+b]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{b}_rows.json')
    pv={k:per_video_useful(v) for k,v in result.items()};pairs=[]
    for arm in ARMS:
        for mode in ('activity','geometry'):pairs.append((arm+'_new_'+mode,arm+'_old_'+mode))
        pairs.extend((arm+'_new_activity',arm+'_'+b) for b in ('old_geometry','new_geometry','top1','kalman'))
    pairs += [('D_new_activity','R1_new_activity'),('D_new_activity','C_new_activity'),('C_new_activity','R1_new_activity'),('D_new_activity','C_old_activity')]
    comparisons={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    c=comparisons['D_new_activity-D_old_activity'];criteria=dict(main=c['mean_per_video']>=.02 and c['ci95'][0]>0,vs_old_map=comparisons['D_new_activity-D_old_geometry']['ci95'][0]>0,vs_R1=comparisons['D_new_activity-R1_new_activity']['ci95'][0]>0)
    summary={}
    for key,rr in result.items():
        cov=coverage_table(rr)
        for q,v in cov.items():
            v['known_nonuseful']=v.pop('harmful')
            selected=sorted(rr,key=lambda r:-r['p_present'])[:v['selected']]
            v['strict_center_worse_than_prediction']=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected)
            v['known_unavailable']=sum(r['kind']==0 for r in selected)
        pos=[r for r in rr if r['kind']==1]
        summary[key]=dict(rows=len(rr),useful_mean_video=float(np.mean(list(pv[key].values()))),accurate_fraction_pooled=sum(r['err']<=.25*r['scale'] for r in pos)/len(pos),coverage=cov)
    oldseed=read(OLD/'seed_metrics.json')['per_video']
    sg={arm:{s:paired(values,oldseed[arm+'_activity'][s]) for s,values in seedpv[arm+'_new_activity'].items()} for arm in ARMS}
    verify_sources(p);assert all(sha(ROOT/k)==h for k,h in manifest.items())
    write(RUN/'seed_metrics.json',dict(per_video=seedpv,matched_gain=sg))
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comparisons,summary=summary,per_video=pv,seed_gain=sg,
        limitations='Only training row sampling changed; common semantic-hard pool, unchanged original labels. Fixed existing cache,old class weight,three-seed mean logits. Reused train26, exploratory. Unknown excluded from training; coverage includes unknown separately. No HOTA or calibrated refusal. Geometry still has FlyVis map.'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');x=a.parse_args()
    if x.prepare: prepare();return
    assert x.worker
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:
        worker();write(RUN/'status.json',dict(state='completed',stage='Результаты18 моделей сохранены',updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),models_done=18,models_total=18))
    except Exception as exc:
        write(RUN/'status.json',dict(state='failed',stage=str(exc),updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2)));raise


if __name__=='__main__':main()
