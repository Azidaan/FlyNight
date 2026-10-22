"""Bounded CPU label-only intervention. Frozen candidate caches and core."""
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
RUN = ROOT/'runs/z13_localization_labels_20261023'
CACHE = ROOT/'runs/z13_candidate_reserve_20261021'
OLD = ROOT/'runs/z13_candidate_data104_20261021'
IDENT = 'z13_localization_labels_v1'
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
    assert not RUN.exists(), 'Existing run must be inspected, not overwritten'
    base = read(OLD/'protocol.json')
    fit = base['train_full104']; evaluate = base['evaluation26']
    assert len(fit) == 104 and len(evaluate) == 26 and not set(fit)&set(evaluate)
    # These reserved splits are metadata only; no dev/held-out examples are loaded.
    dev = read(ROOT/'runs/z13_target_center_dev_20261018/protocol.json')['videos']
    heldout = read(ROOT/'runs/life_fly_history/confirm_videos.json')['usable_after_alignment']
    assert not set(fit)&(set(dev)|set(heldout))
    assert not set(evaluate)&(set(dev)|set(heldout))
    names = ['tracking/z13_localization_labels.py', 'scripts/z13_localization_labels_20261023.py',
             'tests/test_z13_localization_labels.py', 'scripts/z13_localization_labels_console.ps1', 'scripts/z13_bounded_recovery_controller_20261022.py',
             'scripts/z13_hota_deadline_guard.py', 'tracking/z13_matched_data_budget.py',
             'tracking/z13_candidate_ranker.py', 'scripts/z13_candidate_ranker_20261021.py',
             'scripts/z13_candidate_identity_20261021.py', 'scripts/z13_continue.py',
             'scripts/z13_candidate_data104_20261021.py', 'runs/z13_candidate_data104_20261021/protocol.json',
             'runs/z13_candidate_data104_20261021/report.json', 'runs/z13_candidate_data104_20261021/seed_metrics.json',
             'runs/z13_candidate_reserve_20261021/fit/labels.json',
             'runs/z13_candidate_reserve_20261021/evaluate/labels.json',
             'runs/z13_candidate_data104_20261021/evaluate/labels.json',
             'research_flyvis_readout_decision_2026-10-23/ПРОТОКОЛ_ПРЕДЛОЖЕНИЕ.md',
             'runs/z13_target_center_dev_20261018/protocol.json', 'runs/life_fly_history/confirm_videos.json']
    for arm in ARMS:
        for mode in ('activity', 'geometry'):
            names.append(f'runs/z13_candidate_data104_20261021/{arm}/{mode}/ensemble_rows.json')
            for seed in (0, 1, 2):
                names.extend(f'runs/z13_candidate_data104_20261021/{arm}/{mode}/seed{seed}/{f}' for f in ('model.pt', 'training.json', 'done.json'))
        for mode in ('top1', 'kalman'):
            names.append(f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    for folder in (CACHE/'fit', CACHE/'evaluate', OLD/'evaluate'):
        for arm in ARMS:
            assert (folder/arm/'done.json').exists()
            for file in sorted((folder/arm).glob('cache_*.npz')):
                assert sha(file) == read(file.with_suffix('.json'))['sha256']
                names.extend([file.relative_to(ROOT).as_posix(), file.with_suffix('.json').relative_to(ROOT).as_posix()])
    deadline = now().replace(hour=7, minute=40, second=0, microsecond=0)
    assert now().weekday() < 5 and now() < deadline, 'Preparation requires authorized weekday morning window'
    p = dict(id=IDENT, created=now().isoformat(), authorization='Author: Давай пробуем новые варианты. First proposed label-only CPU pilot; 23.10.2026.',
             device='cpu', budget_seconds=900, save_at=(deadline-timedelta(minutes=5)).isoformat(),
             hard_stop=(deadline-timedelta(seconds=30)).isoformat(), stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
             arms=list(ARMS), seeds=[0,1,2], train=fit, evaluate=evaluate,
             split_check=dict(train=104,evaluate=26,dev=len(dev),heldout=len(heldout),no_train_or_evaluation_overlap_with_reserved=True),
             action='Only target labels: near GT center vs near GT AND better than Kalman. Fixed cached candidates, inputs, normalizer, architecture, initialization, data order, updates and old class weights.',
             primary='D new activity - D old activity >=0.02 AND paired-video lower95>0; D new activity beats D old geometry and R1 new activity with lower95>0. All required; C/map branches cannot rescue.',
             secondary='Dnew-Cnew, old C, top1, nearest Kalman, 3 individual seeds, full per-video metrics, 1/2/5% quotas. No tuning or best-seed choice.',
             replay='All18 saved models on full train26: same selected candidate/center/useful; max absolute ensemble score deviation<=1e-5. First32 old-label training losses D/activity/seed0 match within1e-6.',
             gates='Audit changed labels on train104 only; replay all controls; time32 old-label steps, conservative projected full18 model compute must fit original900 seconds. No evaluation efficacy from timing probe.',
             counterargument='Removing Kalman advantage can improve localization while increasing unnecessary harmful corrections. Original utility remains outcome.',
             prohibitions='No core training,GPU,downloads,dev,HOTA,heldout75,source overwrite or revised acceptance thresholds.',
             research='research_flyvis_readout_decision_2026-10-23/ИССЛЕДОВАНИЕ.md',
             sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json', p)
    catpath=ROOT/'experiments/project_history.json'; cat=read(catpath)
    assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: точность центра вместо превосходства над Калманом в обучении выборщика',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',
        tags=['fly-problem-P8','engineering-future'],question='Поможет ли одна замена метки внутри обучаемого выбора кандидатов при прежней оценке полезности?',
        hypothesis='Отделение точности центра от относительной полезности может улучшить выбор, но может увеличить ненужные поправки.',
        expected='Те же104 train и26 оценки,18 CPU моделей: D/C/R1 × активность/карта ×3 сида. Максимум900с плюс20с завершения, без ядра/dev/HOTA. До обучения проверяются мишень, контроль и стоимость.',
        acceptance=[dict(criterion='Метки различаются;18 контролей воспроизведены; исходники,кэши,веса классов,сид,порядок,1150 шагов и145950 предъявлений сохранены; обучение и оценка раздельны.',met=None),
                    dict(criterion='Dnew activity−Dold activity >=2п.п. и нижняя95%повидеоДИ>0.',met=None),
                    dict(criterion='Dnew activity лучше Dold geometry и R1new activity: обе нижние95%повидеоДИ>0.',met=None)],
        summary='Автор разрешил первую проверку по исследованию; критерии зафиксированы до результата.',done='',obtained='',interpretation='',decision='',
        cannot_claim='Train26 многократно использован; нет HOTA, безопасного отказа или доказанной причины прежней неудачи. Карта без сырых векторов всё ещё содержит FlyVis. Новая метка не является сама по себе научной новизной.',
        metrics=[],commits=[],files=names[:3],evidence=[dict(kind='file',ref='runs/z13_localization_labels_20261023/protocol.json',claim='Критерии и хеши до вычислений'),dict(kind='file',ref=p['research'],claim='Ранжирование и источники')]))
    write(catpath,cat)
    journal('постановка проверки новой обучающей метки', '*Постановка.* Автор: «Давай пробуем новые варианты». Принимается первая ступень исследования: меняется только метка выбора центра. Протокол `runs/z13_localization_labels_20261023/protocol.json`; основные критерии D≥2п.п. и положительный повидеоДИ, превосходство над старым чтением карты и новым R1. Максимум900с CPU; предварительно мишень, воспроизведение18 моделей и стоимость32 шагов. Старые веса классов и бюджет сохраняются. Нет GPU, ядра, dev, HOTA и отложенных75. При неудаче основной критерий не заменяется исходом C.')


def train_steps(torch, model, tensors, nref, seed, weight, maxsteps=None, progress=None):
    from tracking.z13_matched_data_budget import matched_batches
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    losses=[]; presented=0; model.train()
    for step, indices in enumerate(matched_batches(len(tensors[0]),nref,50,128,seed)):
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
    from tracking.z13_localization_labels import localization_targets
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json'); verify_sources(p)
    labels20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');labels=labels20+extra;ev=read(CACHE/'evaluate/labels.json')
    assert {l['video'] for l in labels}<=set(p['train']) and {l['video'] for l in ev}<=set(p['evaluate'])
    ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in labels20)
    assert nref==2919 and len(ids)==20778
    arrays={};audit={};replay={};manifest={};result={};seedpv={}
    for ai,arm in enumerate(ARMS):
        status('Проверка кэшей, меток и прежних моделей',variant=arm,done=ai,total=3,models_done=0,models_total=18)
        seq,m=reader.load('fit',arm);manifest.update(m);first=pack(labels20,seq)
        prev=reader.CACHE
        try:
            reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally: reader.CACHE=prev
        manifest.update(m);last=pack(extra,seq);ta=tuple(np.concatenate([a,b]) for a,b in zip(first,last))
        seq,m=reader.load('evaluate',arm);manifest.update(m);ea=pack(ev,seq)
        newy=localization_targets(labels,ta[5],ta[3]);oldy=ta[4];assert np.all(newy>=oldy)
        changed=(newy!=oldy)&ta[3];counts={}
        for i,l in enumerate(labels):
            d=counts.setdefault(l['video'],dict(rows=0,changed_candidates=0,changed_rows=0,near_candidates=0,old_positive_candidates=0))
            d['rows']+=1;d['changed_candidates']+=int(changed[i].sum());d['changed_rows']+=int(changed[i].any());d['near_candidates']+=int(newy[i].sum());d['old_positive_candidates']+=int(oldy[i].sum())
        audit[arm]=dict(changed_candidates=int(changed.sum()),changed_rows=int(changed.any(1).sum()),near_candidates=int(newy.sum()),old_positive_candidates=int(oldy.sum()),per_video=counts)
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
    write(RUN/'label_audit.json',audit);write(RUN/'reproduction.json',replay);write(RUN/'cache_manifest.json',manifest)
    assert audit['D']['changed_candidates']>0, 'No target for intervention'
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
                        status('Обучение новой метки при прежнем бюджете',variant=f'{arm} {mode} seed{seed}',done=step,total=1150,models_done=models_done,models_total=18)
                    losses,presented=train_steps(torch,model,tensors,nref,seed,ref['pos_weight'],progress=progress)
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
        limitations='Only training labels changed. Fixed existing cache,old class weight,three-seed mean logits. Reused train26, exploratory. Unknown excluded from training; coverage includes unknown separately. No HOTA or calibrated refusal. Geometry still has FlyVis map.'))


def main():
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');x=a.parse_args()
    if x.prepare: prepare();return
    assert x.worker
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
    try:
        worker();write(RUN/'status.json',dict(state='completed',stage='Результаты18 моделей сохранены',updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),models_done=18,models_total=18))
    except Exception as exc:
        write(RUN/'status.json',dict(state='failed',stage=str(exc),updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2)));raise


if __name__=='__main__':main()
