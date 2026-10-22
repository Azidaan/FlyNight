"""Matched two-frame attention pilot, CPU, frozen FlyVis and candidate caches."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts.z13_candidate_reserve_20261021 import read,write,sha
from scripts import z13_localization_labels_20261023 as prior
RUN=ROOT/'runs/z13_temporal_candidates_20261023'
OLD=prior.OLD; CACHE=prior.CACHE; IDENT='z13_temporal_candidates_v1'
MSK=timezone(timedelta(hours=3));T0=time.monotonic()
ARMS=('D','C','R1');MODES=('previous_activity','current_activity','previous_geometry','current_geometry')


def now():return datetime.now(MSK)


def journal(title,text):
    old=prior.IDENT
    try:prior.IDENT=IDENT;prior.journal(title,text)
    finally:prior.IDENT=old


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):
        raise TimeoutError('Original budget / Moscow stop reached')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not RUN.exists()
    source=read(prior.RUN/'protocol.json');prior.verify_sources(source)
    assert read(prior.RUN/'finish_recorded.json')['state']=='completed_negative'
    names=['tracking/z13_temporal_candidates.py','tests/test_z13_temporal_candidates.py',
           'scripts/z13_temporal_candidates_20261023.py','scripts/z13_temporal_candidates_console.ps1',
           'runs/z13_localization_labels_20261023/report.json']
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0)
    assert now()<deadline
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=5400,
           save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),
           stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
           authorization='Author explicitly authorized necessary autonomous runs, repairs, commits and push until07:40MSK23.10.2026.',
           action='Inside the FlyVis candidate readout add learned soft cross-frame context over the three previous candidates and a null context. Change candidate scores and selected center.',
           method='Original68-16 embedding and88-32-1 ranker, plus pair network35-16-1, null logit and residual107-32-1. Residual final layer starts zero: exact old model replay. Shared embedding for current/previous. No persistent identities, one-to-one matching or supervised pair identities; this is not full KeepTrack.',
           matched_control='Identical architecture/parameters/initial state/optimizer/data order/updates, but context from current frame rather than previous frame. Both start from same saved old104 model and receive1150 further updates. Additional same architecture geometry ablation and R1 eye features. Geometry still contains FlyVis response maps, so R1 is required.',
           inputs='Current candidate features, exemplar, geometry; previous features/centers/scores at t-1 normalized against CURRENT pred,width and last observed center. Previous target label/GT, current GT and any future frame excluded from feature API. Initial search uses cached t3 context. All cached frames available, including unlabeled. Context is soft attention, null context does NOT mean calibrated target absence.',
           training='Original useful-center binary labels restored, not rejected localization-only labels. Known kind>=0 train104 rows; kind-1 excluded. Class weights from original104 branch. AdamW.001 wd.01 clip10; matched_batches(nref2919,50,128),1150updates145950presentations. Seeds0,1,2 fixed; mean logits, no selection. All parameters of readout trained, core frozen.',
           primary='D previous_activity-current_activity >=0.02 and paired-video95%lower>0. D previous_activity beats D old_activity, D previous_geometry, R1 previous_activity, and D old_geometry: all lower95>0. All required; C cannot rescue.',
           secondary='Every seed, C/R1 matched history comparisons, top1/Kalman,1/2/5% outcomes, full per-video metrics. No tuning after train26. No HOTA claim.',
           gates='Unit tests causal/permutation/padding/ablation;18 saved controls and zero-residual clones reproduce8326rows.32-step CPU timing must fit5400seconds; final exact source/cache hashes.',
           counterargument='Previous peaks may contain the same ambiguity and errors. Apparent gain may be extra capacity or extra training; identical current-context control separates these effects. Sparse GT pairs do not justify invented identities.',
           source='https://arxiv.org/html/2103.16556v2 sections3.4-3.6: learned intra/inter-frame context. Our outcome-supervised residual differs from association-supervised graph/Sinkhorn KeepTrack.',
           train=source['train'],evaluate=source['evaluate'],seeds=[0,1,2],arms=list(ARMS),modes=list(MODES),
           sha256={**source['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    labels=read(CACHE/'fit/labels.json')+read(OLD/'evaluate/labels.json')
    lookup={(l['video'],l['episode'],l['pos']):l for l in labels}
    pairs={str(lag):sum(l['kind']>=0 and lookup.get((l['video'],l['episode'],l['pos']-lag),{}).get('kind')==1 for l in labels) for lag in (1,2,3,6)}
    write(RUN/'pair_annotation_audit.json',dict(known_current_previous_positive=pairs,source=['runs/z13_candidate_reserve_20261021/fit/labels.json','runs/z13_candidate_data104_20261021/evaluate/labels.json'],interpretation='These are annotated row pairs, not all available frames. We do not fabricate target identities for unlabeled previous frames. Current-outcome supervision uses all known current rows.'))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучаемый контекст кандидатов предыдущего кадра',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','KeepTrack','engineering-future'],
        question='Помогает ли предыдущий набор кандидатов сверх той же модели с текущим контекстом?',
        hypothesis='Обучаемое сопоставление активности и координат может уменьшить путаницу пиков; связь с прошлым обучается по исходу текущего кадра, не по выдуманным идентичностям.',
        expected='36CPUмоделей: D/C/R1, текущий/предыдущий контекст, активность/карта,3сида;104train,26оценки,1150дополнительных шагов каждому. До5400с+20с завершения, всё дерево до07:40.',
        acceptance=[dict(criterion='Воспроизведение18старыхмоделей, причинность входов, одинаковый бюджет и окончание36моделей, хеши проверены.',met=None),dict(criterion='D предыдущий−D текущий контекст >=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='D предыдущий лучше прежней D, своего варианта без активности, R1 предыдущего и прежней карты D: все нижние95%повидеоДИ>0.',met=None)],
        summary='Постановка до обучения: один новый временной узел, сопоставимый контроль дополнительной ёмкости и обучения.',done='',obtained='',interpretation='',decision='',
        cannot_claim='Не полная KeepTrack, не перенос идентичностей и не калиброванный отказ. Train26 многократно использован; HOTA не измерена. Модель получает один предыдущий набор, а не всю историю.',metrics=[],commits=[],files=names[:3],
        evidence=[dict(kind='file',ref='runs/z13_temporal_candidates_20261023/protocol.json',claim='Критерии и контроль до исходов'),dict(kind='file',ref='runs/z13_temporal_candidates_20261023/pair_annotation_audit.json',claim='Проверка доступности размеченных пар')]))
    write(cp,cat)
    journal('постановка обучаемого контекста предыдущего кадра','*Постановка.* После отрицательного опыта метки проверяется временной узел считывания. Протокол `runs/z13_temporal_candidates_20261023/protocol.json`. Принцип межкадрового внимания из KeepTrack §3.4; это не полная реализация сопоставления идентичностей. Предыдущая активность и координаты доступны без разметки; loss (ошибка обучения) использует только известный исход текущего кадра. Контроль той же ёмкости читает текущий набор. Все36моделей начинают с прежних104весов и получают одинаковые1150дополнительных шагов; старые метки полезности. Основной прирост D против контроля >=2п.п. и положительный повидеоДИ, дополнительные контроли обязательны. CPU до5400с, без ядра/GPU/dev/HOTA/75. Архитектура draw.io обновится одной вехой после этого связанного блока и отрицательного опыта метки.')


def tensors_for(torch,arrays,mode,ids=None):
    selected=tuple(a if ids is None else a[ids] for a in arrays)
    mem=selected[6:9] if mode.startswith('previous') else (selected[0],selected[2],selected[3])
    return [torch.as_tensor(a,device='cpu') for a in (*selected[:4],*mem,selected[4])]


def train(torch,model,tensors,seed,weight,maxsteps=None,progress=None):
    from tracking.z13_matched_data_budget import matched_batches
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];presented=0
    model.train()
    for step,ix in enumerate(matched_batches(len(tensors[0]),2919,50,128,seed)):
        if maxsteps is not None and step>=maxsteps:break
        if progress and step%23==0:progress(step)
        ix=torch.as_tensor(ix,device='cpu');batch=[a[ix] for a in tensors];score=model(*batch[:-1])
        loss=torch.nn.functional.binary_cross_entropy_with_logits(score,batch[-1],pos_weight=torch.tensor(weight),reduction='none')
        loss=(loss*batch[3]).sum()/batch[3].sum();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
        history.append(float(loss.detach()));presented+=len(ix)
    return history,presented


def predict(torch,model,arrays,mode):
    import numpy as np
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(arrays[0]),512):
            t=tensors_for(torch,tuple(a[start:start+512] for a in arrays),mode)
            result.append(model(*t[:-1]).cpu().numpy())
    return np.concatenate(result)


def worker():
    import gc
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_temporal_candidates import previous_inputs,make_temporal
    from scripts.z13_candidate_ranker_20261021 import pack,make_rows,predict as old_predict
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');prior.verify_sources(p)
    l20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');labels=l20+extra;ev=read(CACHE/'evaluate/labels.json')
    ids=np.asarray([i for i,l in enumerate(labels) if l['kind']>=0]);assert len(ids)==20778
    manifest={};arrays={};replays={};results={};seedpv={};models_done=0
    def packed(ll,seq):
        return (*pack(ll,seq),*previous_inputs([(l['episode'],l['pos'],l['pred']) for l in ll],seq))
    for ai,arm in enumerate(ARMS):
        status('Проверка причинных входов и воспроизведения',variant=arm,done=ai,total=3,models_done=0,models_total=36)
        seq,m=reader.load('fit',arm);manifest.update(m);first=packed(l20,seq)
        previous=reader.CACHE
        try:reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally:reader.CACHE=previous
        manifest.update(m);last=packed(extra,seq);ta=tuple(np.concatenate([a,b]) for a,b in zip(first,last))
        seq,m=reader.load('evaluate',arm);manifest.update(m);ea=packed(ev,seq);arrays[arm]=(ta,ea)
        for mode in ('activity','geometry'):
            preds=[]
            for seed in (0,1,2):
                tr=read(OLD/arm/mode/f'seed{seed}/training.json');assert sha(OLD/arm/mode/f'seed{seed}/model.pt')==tr['model_sha256']
                torch.manual_seed(seed);model=make_temporal(torch,mode=='activity');model.base.load_state_dict(torch.load(OLD/arm/mode/f'seed{seed}/model.pt',map_location='cpu',weights_only=True))
                base=old_predict(torch,model.base,ea)
                for memory in ('previous','current'):
                    clone=predict(torch,model,ea,memory+'_'+mode);assert np.array_equal(base,clone)
                preds.append(base)
            fresh=make_rows(ev,ea,np.mean(preds,axis=0));saved=read(OLD/arm/mode/'ensemble_rows.json')
            delta=max(abs(a['p_present']-b['p_present']) for a,b in zip(fresh,saved));assert len(fresh)==len(saved)==8326 and delta<=1e-5
            assert all(a['candidate']==b['candidate'] and a['center']==b['center'] and a.get('useful')==b.get('useful') for a,b in zip(fresh,saved))
            replays[arm+'_'+mode]=dict(rows=len(saved),old_max_score_delta=delta,zero_residual_both_contexts_exact=True)
            results[arm+'_old_'+mode]=saved
        del seq,first,last;gc.collect()
    write(RUN/'reproduction.json',replays);write(RUN/'cache_manifest.json',manifest)
    ta=arrays['D'][0];mode='previous_activity';torch.manual_seed(0);model=make_temporal(torch)
    model.base.load_state_dict(torch.load(OLD/'D/activity/seed0/model.pt',map_location='cpu',weights_only=True))
    t=tensors_for(torch,ta,mode,ids);beg=time.monotonic();loss,_=train(torch,model,t,0,read(OLD/'D/activity/seed0/training.json')['pos_weight'],maxsteps=32)
    elapsed=time.monotonic()-beg;estimate=elapsed/32*1150*36*2+120
    write(RUN/'timing.json',dict(steps=32,seconds=elapsed,conservative_remaining_seconds=estimate,budget_seconds=p['budget_seconds'],loss=loss))
    assert estimate<p['budget_seconds']-(time.monotonic()-T0),'Timing gate: not enough original budget'
    del model,t;gc.collect()
    for arm in ARMS:
        ta,ea=arrays[arm]
        for mode in MODES:
            activity=mode.endswith('activity');reference='activity' if activity else 'geometry'
            t=tensors_for(torch,ta,mode,ids);preds=[];key=arm+'_'+mode;seedpv[key]={}
            for seed in (0,1,2):
                sd=RUN/arm/mode/f'seed{seed}';ref=read(OLD/arm/reference/f'seed{seed}/training.json')
                torch.manual_seed(seed);model=make_temporal(torch,activity)
                model.base.load_state_dict(torch.load(OLD/arm/reference/f'seed{seed}/model.pt',map_location='cpu',weights_only=True))
                if (sd/'done.json').exists():
                    assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                else:
                    def progress(step):status('Обучение временного узла и равного контроля',variant=f'{arm} {mode} seed{seed}',done=step,total=1150,models_done=models_done,models_total=36)
                    loss,presented=train(torch,model,t,seed,ref['pos_weight'],progress=progress)
                    assert len(loss)==1150 and presented==145950
                    sd.mkdir(parents=True,exist_ok=True);temp=sd/'model.tmp.pt';torch.save(model.state_dict(),temp);temp.replace(sd/'model.pt')
                    write(sd/'training.json',dict(seed=seed,steps=len(loss),presented_rows=presented,pos_weight=ref['pos_weight'],loss=loss,parameters=sum(x.numel() for x in model.parameters()),initial_base_sha256=ref['model_sha256'],model_sha256=sha(sd/'model.pt')))
                    write(sd/'done.json',dict(ended=now().isoformat()))
                pred=predict(torch,model,ea,mode);preds.append(pred);seedpv[key][str(seed)]=per_video_useful(make_rows(ev,ea,pred));models_done+=1
            rr=make_rows(ev,ea,np.mean(preds,axis=0));results[key]=rr;write(RUN/arm/mode/'ensemble_rows.json',rr)
            del t;gc.collect()
    status('Итоговые парные сравнения26видео',models_done=36,models_total=36)
    for arm in ARMS:
        for mode in ('top1','kalman'):results[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    pv={k:per_video_useful(v) for k,v in results.items()};assert all(len(v)==26 for v in pv.values());pairs=[]
    for arm in ARMS:
        pairs.extend((arm+'_previous_activity',arm+'_'+b) for b in ('current_activity','previous_geometry','old_activity','old_geometry','top1','kalman'))
        pairs.append((arm+'_previous_geometry',arm+'_current_geometry'))
    pairs.extend((a+'_previous_activity',b+'_previous_activity') for a,b in [('D','R1'),('C','R1'),('D','C')])
    comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs};main=comp['D_previous_activity-D_current_activity']
    criteria=dict(history=main['mean_per_video']>=.02 and main['ci95'][0]>0,
        controls=all(comp['D_previous_activity-'+b]['ci95'][0]>0 for b in ('D_old_activity','D_previous_geometry','R1_previous_activity','D_old_geometry')))
    summary={}
    for k,rr in results.items():
        cov=coverage_table(rr)
        for quota,v in cov.items():
            v['known_nonuseful']=v.pop('harmful');selected=sorted(rr,key=lambda r:-r['p_present'])[:v['selected']]
            v['strict_center_worse_than_prediction']=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected)
            v['known_unavailable']=sum(r['kind']==0 for r in selected)
        summary[k]=dict(rows=len(rr),useful_mean_video=float(np.mean(list(pv[k].values()))),coverage=cov)
    seeds={a:{s:paired(v,seedpv[a+'_current_activity'][s]) for s,v in seedpv[a+'_previous_activity'].items()} for a in ARMS}
    prior.verify_sources(p);assert all(sha(ROOT/n)==h for n,h in manifest.items())
    write(RUN/'seed_metrics.json',dict(per_video=seedpv,matched_history_gain=seeds))
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comp,summary=summary,per_video=pv,seeds=seeds,
        limitations='One previous frame, learned soft context supervised only by current utility. No explicit IDs or calibrated refusal, no HOTA. Fixed three seeds and all104train. Reused train26 exploratory. Map geometry is not no-FlyVis control; R1 comparison required.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    elif a.worker:
        try:
            worker();write(RUN/'status.json',dict(state='completed',stage='36моделей и все сравнения сохранены',updated=now().isoformat(),models_done=36,models_total=36,elapsed_seconds=round(time.monotonic()-T0,2)))
        except Exception as e:
            write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2)));raise
    else:ap.error('Choose prepare or worker')
