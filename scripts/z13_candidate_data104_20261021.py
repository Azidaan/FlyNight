"""One-factor training-data expansion with fixed optimization/normalization.

The cache phase calls the unchanged original extractor in a separate process.
Only its output root and data-list globals are redirected; its mathematics,
normalizer train20, lambda, core and candidate definition are unchanged.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,verify,now,MSK
RUN=ROOT/'runs/z13_candidate_data104_20261021';OLD=ROOT/'runs/z13_candidate_ranker_20261021';CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
IDENT='z13_candidate_data104_v1';ATTEMPT='cache';LIMIT=3600;T0=time.monotonic()


def check():
    if time.monotonic()-T0>LIMIT or (RUN/(ATTEMPT+'.stop')).exists() or now()>=datetime(2026,10,21,7,35,tzinfo=MSK):raise TimeoutError('Budget/save deadline')
def status(stage,**kw):
    check();write(RUN/'status.json',dict(state='running',phase=ATTEMPT,stage=stage,updated=now().isoformat(),elapsed_seconds=int(time.monotonic()-T0),**kw))
def journal(title,text):
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';s=path.read_text(encoding='utf-8');a='## Журнал изменений\n';assert s.count(a)==1
    path.write_text(s.replace(a,a+f'\n**{now():%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n'+text+'\n',1),encoding='utf-8')


def prepare():
    if RUN.exists():raise RuntimeError('Existing run must be inspected')
    old=read(CACHE/'protocol.json');verify(old)
    full=read(ROOT/'runs/z13_spatial_v2_20261013/D/full/identity.json')['config']['job']['fit'];extra=sorted(set(full)-set(old['fit']))
    assert len(full)==104 and len(extra)==84 and not set(full)&set(old['evaluate']) and set(old['fit'])<=set(full)
    names=['tracking/z13_matched_data_budget.py','scripts/z13_candidate_data104_20261021.py','tests/test_z13_matched_data_budget.py','tracking/z13_candidate_ranker.py','scripts/z13_candidate_ranker_20261021.py','scripts/z13_candidate_identity_20261021.py','scripts/z13_night_control_20261021.py','runs/z13_candidate_ranker_20261021/report.json','runs/z13_candidate_ranker_20261021/seed_metrics.json','runs/z13_candidate_reserve_20261021/fit/labels.json','runs/z13_candidate_reserve_20261021/evaluate/labels.json','runs/flyvis/results/flow/0000/000/best_chkpt','runs/flyvis/results/flow/0000/000/_meta.yaml']
    names.extend(f'runs/z13_data/{v}.npz' for v in extra)
    for arm in ('C','D','R1'):
        names.extend(f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json' for mode in ('top1','kalman'))
        for mode in ('activity','geometry'):
            names.append(f'runs/z13_candidate_ranker_20261021/{arm}/{mode}/ensemble_rows.json')
            for seed in (0,1,2):names.append(f'runs/z13_candidate_ranker_20261021/{arm}/{mode}/seed{seed}/training.json')
    p=dict(id=IDENT,created=now().isoformat(),authorization='Explicit author autonomous authorization until07:40 MSK21.10.2026.',
        fit=old['fit'],evaluate=extra,arms=['C','D','R1'],train_full104=full,evaluation26=old['evaluate'],
        cache_field_note='evaluate is only the unchanged extractor field for the additional84 TRAIN videos; it is not an evaluation set. Train104 and evaluation26 are disjoint. Original train20 normalization is retained.',
        action='Increase only the training-video pool for the same small candidate readout from20 to104, preserving architecture, initial seeds, loss, class weights, optimizer, totalupdates and totalpresentations. Old20 models are reused, never retrained.',
        training='Same factory, fixed seeds0/1/2, BCE, AdamW lr0.001,weight_decay0.01,gradientclip10. For each arm use original known train20 row count as reference; same50 reference-epoch batch-size sequence (128 and original remainder), so updates and presented rows exactly equal old run. New data traversed by deterministic seeded permutations. Reuse original train20 positive class weight per arm, no retuning. Fixed200-example train20 feature normalizer, same saved target lambda and original3-peak extractor.',
        primary='C104 activity-C20 activity: mean gain>=0.02 and paired-video95%lower>0. C104 activity must also beat C104 geometry,R1_104 activity,old Ctop1,and nearest-prediction choice: all lower>0. All3 matched C-seed data-gain point estimates nonnegative. D and data-effect interaction C-R1 secondary, no rescue.',
        counterargument='More training observations may improve only geometric learning or R1, or may not compensate for the representation. New104 sees fewer effective passes per video at the same compute budget. No cause is assumed from old801-positive-frame limitation.',
        decision='No train26 hyperparameter selection and no best-seed choice. If primary fails, retain original centered readout; do not promote generic geometry improvement as a FlyVis result.',
        budget_cache_seconds=3600,budget_train_seconds=1800,save_at='2026-10-21T07:35:00+03:00',hard_stop='2026-10-21T07:39:30+03:00',
        paper='https://arxiv.org/html/2103.16556v2 section3.6: learning candidate association depends on sufficient informative training samples. This data-volume check is not a reproduction.',
        prohibited='No core training,downloads,dev,HOTA,heldout75,changing old data/code/thresholds.',
        sha256={**old['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p);cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(v['id']==IDENT for v in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: больше обучающих видео при том же бюджете считывателя',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','fly-problem-P10','KeepTrack','engineering-future'],
        question='Помогают ли дополнительные84 train-видео тому же узлу выбора пиков при неизменных нормировке, архитектуре, сидах, весах потери и числе обновлений?',
        hypothesis='Дополнительные обучающие ситуации могут уменьшить переобучение малого считывателя; недостаток данных пока является гипотезой, а не причиной, установленной прошлым результатом.',
        expected='Извлечь только недостающие84 train, сохранить старые20/26 кэши и модели. После завершения GPU-процесса отдельный CPU-процесс обучает те же18 моделей на104train с прежним числом шагов и наблюдений. Сравнение на тех же26 без подбора.',
        acceptance=[dict(criterion='Train104 и оценка26 не пересекаются; исходный extractor неизменен; нормировка только прежних20; шаги,объём наблюдений,веса BCE,архитектура и сиды совпадают с контролем20.',met=None),dict(criterion='C104−C20 activity: прирост>=2п.п. и нижняя95%повидеоДИ>0; точки3 совпадающих сидов неотрицательны.',met=None),dict(criterion='C104 activity лучше C104 geometry,R1_104 activity,старого Ctop1 и ближайшего к прогнозу кандидата: все4 нижние95%повидеоДИ>0.',met=None)],
        summary='Зафиксирована однофакторная проверка объёма данных считывателя; исход не получен.',done='',obtained='',interpretation='',decision='',cannot_claim='Нехватка801 положительного кадра не установлена как причина прошлой неудачи. Данные train26 многократно использованы; это поисковое сравнение, не HOTA, не безопасный отказ и не доказательство экономии данных. Ядро не обучается.',metrics=[],commits=[],files=names[:2],evidence=[dict(kind='file',ref='runs/z13_candidate_data104_20261021/protocol.json',claim='Критерии и совпадающий бюджет до исходов'),dict(kind='file',ref='runs/z13_candidate_ranker_20261021/report.json',claim='Ранее узел на20 train не прошёл основные критерии')]))
    write(cp,cat);journal('постановка однофакторного расширения данных считывателя','*Постановка.* '+p['action']+'\n\n'+p['training']+'\n\n'+p['primary']+'\n\nДоступны104 train и26 непересекающихся видео оценки;84 дополнительных уже лежат в runs/z13_data, загрузок нет. Риск нехватки801 положительного кадра был записан до прежнего прогона, но причина не объявляется доказанной. GPU-кэш до3600с; затем новый отдельный CPU-процесс до1800с. Независимые сохранение07:35 и конец дерева до07:39:30; законченные старые модели не повторяются.')


def cache():
    p=read(RUN/'protocol.json');verify(p)
    # Redirect only in this service process; the frozen source file is unchanged.
    import scripts.z13_candidate_reserve_20261021 as extractor
    extractor.RUN=RUN;extractor.SPLIT='evaluate';extractor.T0=T0;extractor.LIMIT=LIMIT;extractor.status=status
    def cache_report():
        manifest={}
        for arm in p['arms']:
            assert (RUN/'evaluate'/arm/'done.json').exists()
            for path in (RUN/'evaluate'/arm).glob('cache_*.npz'):
                record=read(path.with_suffix('.json'));assert sha(path)==record['sha256'];manifest[path.relative_to(ROOT).as_posix()]=record
        write(RUN/'additional84_cache_manifest.json',manifest)
        write(RUN/'cache_complete.json',dict(training_videos=84,normalizer_videos=20,no_evaluation26_added=True,extractor_sha256=sha(ROOT/'scripts/z13_candidate_reserve_20261021.py')))
    extractor.report=cache_report;extractor.extract();verify(p)


def train():
    import gc
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_default_dtype(torch.float32);torch.set_num_threads(2)
    from tracking.z13_matched_data_budget import matched_batches
    from tracking.z13_candidate_ranker import make_ranker
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_candidate_ranker_20261021 import pack,predict,make_rows
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');verify(p);end=read(RUN/'cache_controller_result.json');assert end['worker_ended'] and end['exit_code']==0 and not end['hard_timeout']
    oldlabels=read(CACHE/'fit/labels.json');extra_labels=read(RUN/'evaluate/labels.json');evlabels=read(CACHE/'evaluate/labels.json')
    labels=oldlabels+extra_labels;ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);nref=sum(l['kind']>=0 for l in oldlabels)
    oldids=np.asarray([i for i,l in enumerate(oldlabels) if l['kind']>=0]);assert nref==len(oldids)
    results={};seedpv={};manifest={};budget={}
    for ai,arm in enumerate(p['arms']):
        status('Проверка фиксированных кэшей20+84 train',variant=arm,done=ai,total=3)
        a,m=reader.load('fit',arm);manifest.update(m);first=pack(oldlabels,a)
        oldroot=reader.CACHE
        try:reader.CACHE=RUN;a,m=reader.load('evaluate',arm)
        finally:reader.CACHE=oldroot
        manifest.update(m);extra=pack(extra_labels,a);ta=tuple(np.concatenate([x,y]) for x,y in zip(first,extra));del first,extra,a
        a,m=reader.load('evaluate',arm);manifest.update(m);ea=pack(evlabels,a);del a
        tensors=[torch.as_tensor(x[ids],device='cpu') for x in ta[:5]]
        for mode,activity in [('activity',True),('geometry',False)]:
            directory=RUN/arm/mode;preds=[];seedpv[arm+'_'+mode]={}
            for seed in (0,1,2):
                sd=directory/f'seed{seed}';reference=read(OLD/arm/mode/f'seed{seed}/training.json');assert reference['known_rows']==nref and reference['epochs']==50
                torch.manual_seed(seed);model=make_ranker(torch,activity);weight=reference['pos_weight']
                if (sd/'done.json').exists():
                    assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                else:
                    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);losses=[];presented=0;steps=50*((nref+127)//128)
                    for step,indices in enumerate(matched_batches(len(ids),nref,50,128,seed)):
                        if step%23==0:status('Обучение104 train при прежнем числе шагов',variant=arm+' '+mode+' seed'+str(seed),done=step,total=steps)
                        check();ix=torch.as_tensor(indices,device='cpu');x,s,g,mask,target=[t[ix] for t in tensors];score=model(x,s,g,mask)
                        assert score.device.type=='cpu' and next(model.parameters()).device.type=='cpu'
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(score,target,pos_weight=torch.tensor(weight,device='cpu'),reduction='none');loss=(loss*mask).sum()/mask.sum();assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();losses.append(float(loss.detach()));presented+=len(indices)
                    assert len(losses)==steps and presented==50*nref
                    sd.mkdir(parents=True,exist_ok=True);tmp=sd/'model.tmp.pt';torch.save(model.state_dict(),tmp);tmp.replace(sd/'model.pt')
                    write(sd/'training.json',dict(seed=seed,steps=steps,reference_epochs=50,reference_known_rows=nref,known_train104_rows=len(ids),presented_rows=presented,pos_weight=weight,reference_pos_weight=reference['pos_weight'],loss_mean_first50=float(np.mean(losses[:50])),loss_mean_last50=float(np.mean(losses[-50:])),loss=losses,model_sha256=sha(sd/'model.pt')))
                    write(sd/'done.json',dict(ended=now().isoformat()))
                pred=predict(torch,model,ea);preds.append(pred);rr=make_rows(evlabels,ea,pred);seedpv[arm+'_'+mode][str(seed)]=per_video_useful(rr)
            rr=make_rows(evlabels,ea,np.mean(preds,axis=0));results[arm+'104_'+mode]=rr;write(directory/'ensemble_rows.json',rr)
            results[arm+'20_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for baseline in ('top1','kalman'):results[arm+'_'+baseline]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{baseline}_rows.json')
        budget[arm]=dict(known104=len(ids),reference_known20=nref,steps=steps,presented=50*nref)
        del ta,ea,tensors;gc.collect()
    pv={k:per_video_useful(v) for k,v in results.items()};pairs=[]
    for arm in ('C','D'):
        pairs.extend([(arm+'104_activity',arm+'20_activity'),(arm+'104_activity',arm+'104_geometry'),(arm+'104_activity','R1104_activity'),(arm+'104_activity',arm+'_top1'),(arm+'104_activity',arm+'_kalman')])
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    seedsold=read(OLD/'seed_metrics.json')['per_video'];seedgain={arm:{s:paired(v,seedsold[arm+'_activity'][s]) for s,v in seedpv[arm+'_activity'].items()} for arm in ('C','D','R1')}
    cg={v:pv['C104_activity'][v]-pv['C20_activity'][v] for v in pv['C20_activity']};rg={v:pv['R1104_activity'][v]-pv['R120_activity'][v] for v in pv['R120_activity']};comp['C_data_effect-R1_data_effect']=paired(cg,rg)
    c=comp['C104_activity-C20_activity'];others=['C104_activity-C104_geometry','C104_activity-R1104_activity','C104_activity-C_top1','C104_activity-C_kalman'];passed=c['mean_per_video']>=.02 and c['ci95'][0]>0 and all(comp[k]['ci95'][0]>0 for k in others) and all(v['mean_per_video']>=0 for v in seedgain['C'].values())
    verify(p);write(RUN/'cache_manifest.json',manifest);write(RUN/'training_budget.json',budget);write(RUN/'seed_metrics.json',dict(per_video=seedpv,data_gain=seedgain))
    write(RUN/'report.json',dict(primary_pass=passed,paired=comp,seed_data_gain=seedgain,summary={k:dict(rows=len(v),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=coverage_table(v)) for k,v in results.items()},limitations='Only train-data pool changed; optimizer updates,observed sample counts,weights,normalizer,architecture,seeds fixed. Reused train26, exploratory; no HOTA or safe intervention.'))


def main():
    global ATTEMPT,LIMIT
    a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',action='store_true');a.add_argument('--attempt',choices=['cache','train'],default='cache');a.add_argument('--budget',type=int,default=3600);x=a.parse_args()
    if x.prepare:prepare();return
    ATTEMPT=x.attempt;LIMIT=x.budget;os.environ['CUDA_VISIBLE_DEVICES']='0' if ATTEMPT=='cache' else '';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
    if not x.worker:raise ValueError('Choose prepare/worker')
    try:
        if ATTEMPT=='cache':cache()
        else:train()
        write(RUN/'status.json',dict(state='completed',phase=ATTEMPT,stage='Этап данных/обучения сохранён',updated=now().isoformat()))
    except Exception as exc:write(RUN/'status.json',dict(state='failed',phase=ATTEMPT,stage=str(exc),updated=now().isoformat()));raise


if __name__=='__main__':main()
