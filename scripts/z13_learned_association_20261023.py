"""Partial target correspondence plus synthetic candidate identity training."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts.z13_candidate_reserve_20261021 import read,write,sha
from scripts import z13_localization_labels_20261023 as prior
RUN=ROOT/'runs/z13_learned_association_20261023';OLD=prior.OLD;CACHE=prior.CACHE
IDENT='z13_learned_association_v1';ARMS=('D','C','R1');MSK=timezone(timedelta(hours=3));T0=time.monotonic()


def now():return datetime.now(MSK)


def journal(title,text):
    old=prior.IDENT
    try:prior.IDENT=IDENT;prior.journal(title,text)
    finally:prior.IDENT=old


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original budget or Moscow deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not RUN.exists();base=read(prior.RUN/'protocol.json');prior.verify_sources(base)
    assert read(ROOT/'runs/z13_temporal_candidates_20261023/finish_recorded.json')['state']=='completed_negative'
    names=['tracking/z13_learned_association.py','scripts/z13_learned_association_20261023.py','tests/test_z13_learned_association.py','scripts/z13_learned_association_console.ps1',
        'runs/z13_temporal_candidates_20261023/report.json','tracking/z13_candidate_identity.py','runs/z13_candidate_identity_20261021/report.json']
    for arm in ARMS:
        names.extend(f'runs/z13_candidate_identity_20261021/{arm}/{n}' for n in ['choice.json','association_rows.json','geometry_rows.json'])
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0);assert now()<deadline
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=7200,save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),
        authorization='Author: necessary nonstop experiments, independent decisions/repairs/commits/push until07:40MSK23.10.2026.',
        action='Replace fixed cosine/geometric association cost inside causal FlyVis candidate identity readout by a learned partial assignment matrix with explicit unmatched row/column. Same cached peaks, causal object store and initial target selection.',
        architecture='Shared LayerNorm68-linear16-tanh plus geometry3-16-16; two self/cross attention pairs of width16; projected dot product/4; learned dustbin score;10 log-Sinkhorn iterations. Positions in same current seed-origin/crop-width; score divided by max absolute observed score of both sets. Geometry ablation zeros raw features in identical network.',
        partial_training='Train104 only. Every labeled pair in same video/episode at positive gap1..6, both outcomes known. Canonical target candidate is closest to GT within existing0.25scale tolerance; kind0 or no close peak maps to dustbin. Unknown excluded, both-dustbin ignored. Only single target correspondence supervised; other real distractor matches unknown, never fabricated. Gap distribution reported. Most real pairs are6frames apart; application remains30Hz, a stated transfer risk.',
        synthetic_training='Same-frame candidate sets with known original indices; independently retain each candidate with probability0.8, ensure at least one per side. Right features additive Gaussian0.02 times within-candidate feature std (floor1e-6); positions independent Gaussian0.02 normalizedwidth; normalized score Gaussian0.03. No core/image augmentation. Matches/dustbins from retained indices; average NLL over known links. Synthetic deletion is not a model of every occlusion.',
        training='18models D/C/R1 x activity/geometry x seeds0,1,2. Random initialization,1150AdamW updates lr.001 wd.01 clip10. Each batch64 real+64 synthetic, fixed RNG per seed, equal averaging of real/synthetic per-pair losses. No class weights or checkpoint/seed selection. Mean log-assignment ensemble BEFORE identity decisions, plus individual seed results.',
        inference='Same30Hz cached frames and object store as old fixed association: observed t3 target nearest peak if distance/width<=0.25, otherwise original exemplar retained. Hungarian assignment with per-row dummy costs; accept link only if log real probability>log unmatched. Missing target retains last observed representation; unmatched distractors get new IDs. Confidence=log(real/unmatched), not calibrated probability. Unknown query labels unavailable to model.',
        primary='D learned activity - D old fixed association >=0.02 and paired-video lower95>0. D learned activity beats D learned geometry, R1 learned activity and strong D old104 geometry selector, each lower95>0. All required; C cannot rescue.',
        secondary='All three seeds; D/C vs old104activity/top1/Kalman; D-C; C-R1; 1/2/5% quotas with unknown, strict worse, actual interventions; no efficacy-driven threshold tuning.',
        gates='Causal/permutation/padding/null and batch-vs-sequential tests; replay original fixed association on all26video; actual nonempty matched/missing target pairs and unchanged caches;32training steps and one full timing inference estimate must fit7200seconds. Timing uses untrained/random model, no efficacy decision. Final hash checks.',
        counterargument='Three peaks need not represent distinct physical objects; canonical nearest target is a training convention. Sparse30Hz annotations and synthetic features may not teach real adjacent-frame identity. Missing target can retain a stale exemplar. Strong map/R1 controls prevent attributing generic geometry gains to FlyVis.',
        paper='https://arxiv.org/html/2103.16556v2 sections3.4-3.7, eq2-3 andtable2; https://arxiv.org/html/1911.11763v1. Adapted tiny model, cached feature augmentation and simpler retained-target policy, not full KeepTrack reproduction.',
        train=base['train'],evaluate=base['evaluate'],arms=list(ARMS),seeds=[0,1,2],sha256={**base['sha256'],**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучаемые соответствия пиков с отсутствующей парой',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','KeepTrack','engineering-future'],
        question='Улучшит ли явное обучение соответствий причинное ведение кандидата цели сверх фиксированного сопоставления, карты и входа глаза?',
        hypothesis='Частичные истинные пары цели и искусственные пары кандидатов могут обучить различению соседей и отсутствующей пары; обычный временной контекст этого не проверял.',
        expected='18 CPU моделей на104train,3сида;18 сетей,1150шагов,64реальных+64искусственныхпары в пакете. Оценка всех26train. До7200с+20с завершения, своё дерево до07:40; без ядра/GPU/dev/HOTA/75.',
        acceptance=[dict(criterion='Тесты причинности/пустой пары/масок, воспроизведение фиксированного контроля, состав пар,18моделей и хеши проверены.',met=None),dict(criterion='D learned−D fixed >=2п.п. и нижняя95%повидеоДИ>0.',met=None),dict(criterion='D learned лучше D без активности, R1 learned и прежней карты104D: все нижние95%повидеоДИ>0.',met=None)],
        summary='Протокол до результатов: явные частичные соответствия и пустая пара, без выдуманных идентичностей соседей.',done='',obtained='',interpretation='',decision='',
        cannot_claim='Не полная KeepTrack; соседние пики могут принадлежать одному объекту. Синтетическое удаление не доказывает работу при реальной окклюзии; пары с разрывом до6кадров отличаются от30Гц применения. Train26 поисковый, отказ не калиброван, HOTA не измерена.',metrics=[],commits=[],files=names[:3],evidence=[dict(kind='file',ref='runs/z13_learned_association_20261023/protocol.json',claim='Протокол, контроли и критерии до прогона'),dict(kind='file',ref='runs/z13_temporal_candidates_20261023/report.json',claim='Предыдущий мягкий контекст не улучшил равный контроль')]))
    write(cp,cat)
    journal('постановка явных соответствий кандидатов','*Постановка.* Меняется фиксированная стоимость сопоставления внутри пути мухи, вместо очередного изменения оценки готового центра. Протокол `runs/z13_learned_association_20261023/protocol.json`; KeepTrack §3.4–3.7: частичные пары цели и искусственные пары кандидатов с удалением. У неизвестных реальных соседей соответствия не приписываются. Доступны разрывы1–6кадров, применение30Гц остаётся риском переноса. D/C/R1, одинаковая архитектура с активностью и без,3сида,18моделей; независимый предел7200с CPU, всё до07:40. Главные критерии D против фиксированного сопоставления >=2п.п. и положительный ДИ, плюс карта104D и R1. Нет нового ядра/dev/HOTA. Предварительно проверяются реальные пары, старый контроль и стоимость.')


def build_training(labels,sequences):
    import numpy as np
    from tracking.z13_learned_association import pair_inputs,target_index,pack_pairs
    frames={(e,f['t']):f for e,fs in sequences.items() for f in fs}
    lookup={(l['video'],l['episode'],l['pos']):l for l in labels};pairs=[];targets=[];counts={};self_pairs=[]
    for e,fs in sequences.items():
        for f in fs:self_pairs.append(pair_inputs(f,f,f['width'],np.asarray(f['seed_center'])))
    for current in labels:
        if current['kind']<0:continue
        f=frames[(current['episode'],current['pos']+4)]
        for gap in range(1,7):
            previous=lookup.get((current['video'],current['episode'],current['pos']-gap))
            if previous is None or previous['kind']<0:continue
            g=frames[(previous['episode'],previous['pos']+4)]
            a=target_index(previous,g);b=target_index(current,f)
            if a<0 and b<0:continue
            pairs.append(pair_inputs(g,f,f['width'],np.asarray(f['seed_center'])));targets.append((a,b))
            key=f'gap{gap}_'+('matched' if a>=0 and b>=0 else 'missing' if b<0 else 'appeared');counts[key]=counts.get(key,0)+1
    assert pairs and self_pairs
    arrays=pack_pairs(pairs);size=arrays[0].shape[1];y=np.zeros((len(pairs),size+1,size+1),np.float32)
    for i,(a,b) in enumerate(targets):y[i,size if a<0 else a,size if b<0 else b]=1.
    return arrays,y,self_pairs,counts


def synthetic_batch(np,rng,pool,count):
    from tracking.z13_learned_association import pack_pairs
    pairs=[];links=[]
    for index in rng.integers(0,len(pool),size=count):
        x,g,_,_=pool[index];n=len(x);left=np.flatnonzero(rng.random(n)<.8);right=np.flatnonzero(rng.random(n)<.8)
        if not len(left):left=np.array([rng.integers(n)])
        if not len(right):right=np.array([rng.integers(n)])
        y=x[right].copy();h=g[right].copy()
        y+=rng.normal(size=y.shape).astype(np.float32)*.02*np.maximum(y.std(1,keepdims=True),1e-6)
        h[:,:2]+=rng.normal(0,.02,size=h[:,:2].shape);h[:,2]+=rng.normal(0,.03,size=len(h))
        pairs.append((x[left],g[left],y,h));local=[]
        for i,identity in enumerate(left):
            found=np.flatnonzero(right==identity);local.append((i,int(found[0]) if len(found) else -1))
        for j,identity in enumerate(right):
            if identity not in left:local.append((-1,j))
        links.append(local)
    arrays=pack_pairs(pairs)
    # Fixed size3 ensures identical label indexing when joining real and synthetic.
    for i,a in enumerate(arrays):
        if a.shape[1]<3:
            pad=[(0,0),(0,3-a.shape[1])]+([(0,0)] if a.ndim==3 else []);arrays[i]=np.pad(a,pad)
    y=np.zeros((count,4,4),np.float32)
    for row,ll in enumerate(links):
        for a,b in ll:y[row,3 if a<0 else a,3 if b<0 else b]=1/len(ll)
    return arrays,y


def train(torch,model,arrays,y,pool,seed,maxsteps=1150,progress=None):
    import numpy as np
    rng=np.random.default_rng(seed);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);losses=[]
    assert arrays[0].shape[1]==3
    model.train()
    for step in range(maxsteps):
        if progress and step%23==0:progress(step)
        ix=rng.integers(0,len(y),size=64);sa,sy=synthetic_batch(np,rng,pool,64)
        t=[torch.as_tensor(np.concatenate([a[ix],b]),device='cpu') for a,b in zip(arrays,sa)]
        target=torch.as_tensor(np.concatenate([y[ix],sy]),device='cpu');z=model(*t)
        loss=-(z*target).sum((1,2)).mean();assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step();losses.append(float(loss.detach()))
    return losses


def worker():
    import gc
    import numpy as np
    import torch
    torch.set_default_device('cpu');torch.set_num_threads(2)
    from tracking.z13_learned_association import make_association,select_sequences
    from tracking.z13_candidate_identity import IdentityReadout
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_continue import paired,per_video_useful,coverage_table
    p=read(RUN/'protocol.json');prior.verify_sources(p);l20=read(CACHE/'fit/labels.json');extra=read(OLD/'evaluate/labels.json');ev=read(CACHE/'evaluate/labels.json')
    data={};evals={};manifest={};audits={};replay={};results={}
    for ai,arm in enumerate(ARMS):
        status('Подготовка реальных и искусственных пар',variant=arm,done=ai,total=3,models_done=0,models_total=18)
        seq,m=reader.load('fit',arm);manifest.update(m);a,y,pool,counts=build_training(l20,seq)
        previous=reader.CACHE
        try:reader.CACHE=OLD;seq,m=reader.load('evaluate',arm)
        finally:reader.CACHE=previous
        manifest.update(m);b,yy,pool2,counts2=build_training(extra,seq)
        arrays=[np.concatenate([x,z]) for x,z in zip(a,b)];labels=np.concatenate([y,yy]);pool+=pool2
        for k,v in counts2.items():counts[k]=counts.get(k,0)+v
        assert sum(v for k,v in counts.items() if k.endswith('_matched'))>0
        data[arm]=(arrays,labels,pool);audits[arm]=dict(real_pairs=len(labels),synthetic_source_frames=len(pool),counts=counts)
        seq,m=reader.load('evaluate',arm);manifest.update(m);evals[arm]=seq;choice=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/choice.json')['association']['best'];answers={}
        for e,frames in seq.items():
            f=frames[0];r=IdentityReadout(f['seed_center'],f['seed_features'],choice['alpha'],choice['missing_cost'])
            for f in frames:answers[(e,f['t'])]=r.step(f['centers'],f['features'],f['width'])
        rr=reader.rows(ev,seq,answers);saved=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/association_rows.json')
        assert len(rr)==len(saved)==8326;assert all(a==b for a,b in zip(rr,saved))
        replay[arm]=dict(rows=len(rr),all_fields_equal=True);results[arm+'_fixed']=saved
        del a,b,y,yy,seq;gc.collect()
    write(RUN/'pair_audit.json',audits);write(RUN/'cache_manifest.json',manifest);write(RUN/'reproduction.json',replay)
    status('Замер32шагов и полного чтения без оценки эффективности',models_done=0,models_total=18)
    torch.manual_seed(0);model=make_association(torch);beg=time.monotonic();train(torch,model,*data['D'],0,maxsteps=32);seconds=time.monotonic()-beg
    # Separate random network for timing inference: no pilot efficacy decision.
    torch.manual_seed(97);probe=make_association(torch);beg=time.monotonic();select_sequences(torch,[probe],evals['D']);infer=time.monotonic()-beg
    estimate=seconds/32*1150*18*2+infer*36*2+120
    write(RUN/'timing.json',dict(train32_seconds=seconds,full_inference_seconds=infer,conservative_remaining_seconds=estimate,budget_seconds=p['budget_seconds']))
    assert estimate<p['budget_seconds']-(time.monotonic()-T0),'Work exceeds original resource budget'
    del model,probe;gc.collect();models_done=0;seedpv={}
    for arm in ARMS:
        for mode in ('activity','geometry'):
            models=[];seedpv[arm+'_'+mode]={}
            for seed in (0,1,2):
                sd=RUN/arm/mode/f'seed{seed}';torch.manual_seed(seed);model=make_association(torch,mode=='activity')
                if (sd/'done.json').exists():
                    assert sha(sd/'model.pt')==read(sd/'training.json')['model_sha256'];model.load_state_dict(torch.load(sd/'model.pt',map_location='cpu',weights_only=True))
                else:
                    def progress(step):status('Обучение частичных соответствий и отсутствующей пары',variant=f'{arm} {mode} seed{seed}',done=step,total=1150,models_done=models_done,models_total=18)
                    losses=train(torch,model,*data[arm],seed,progress=progress);sd.mkdir(parents=True,exist_ok=True);tmp=sd/'model.tmp.pt';torch.save(model.state_dict(),tmp);tmp.replace(sd/'model.pt')
                    write(sd/'training.json',dict(seed=seed,steps=len(losses),real_presentations=1150*64,synthetic_presentations=1150*64,parameters=sum(x.numel() for x in model.parameters()),loss=losses,model_sha256=sha(sd/'model.pt')));write(sd/'done.json',dict(ended=now().isoformat()))
                def progress(t):status('Причинное ведение кандидатов на26видео',variant=f'{arm} {mode} seed{seed}',done=t-3,total=31,models_done=models_done,models_total=18)
                answers=select_sequences(torch,[model],evals[arm],progress=progress);rr=reader.rows(ev,evals[arm],answers)
                write(sd/'rows.json',rr);seedpv[arm+'_'+mode][str(seed)]=per_video_useful(rr);models.append(model);models_done+=1
            status('Ансамбль соответствий до выбора идентичностей',variant=arm+' '+mode,models_done=models_done,models_total=18)
            answers=select_sequences(torch,models,evals[arm],progress=lambda t:status('Ансамбль: причинное ведение кандидатов',variant=arm+' '+mode,done=t-3,total=31,models_done=models_done,models_total=18))
            rr=reader.rows(ev,evals[arm],answers);results[arm+'_'+mode]=rr;write(RUN/arm/mode/'ensemble_rows.json',rr);del models;gc.collect()
        for mode in ('activity','geometry'):results[arm+'_old_'+mode]=read(OLD/arm/mode/'ensemble_rows.json')
        for mode in ('top1','kalman'):results[arm+'_'+mode]=read(ROOT/f'runs/z13_candidate_identity_20261021/{arm}/{mode}_rows.json')
    status('Полные парные итоги',models_done=18,models_total=18)
    pv={k:per_video_useful(v) for k,v in results.items()};assert all(len(v)==26 for v in pv.values());pairs=[]
    for arm in ARMS:pairs.extend((arm+'_activity',arm+'_'+b) for b in ('fixed','geometry','old_activity','old_geometry','top1','kalman'))
    pairs.extend((a+'_activity',b+'_activity') for a,b in [('D','R1'),('C','R1'),('D','C')]);comp={a+'-'+b:paired(pv[a],pv[b]) for a,b in pairs}
    main=comp['D_activity-D_fixed'];criteria=dict(main=main['mean_per_video']>=.02 and main['ci95'][0]>0,controls=all(comp['D_activity-'+b]['ci95'][0]>0 for b in ('D_geometry','R1_activity','D_old_geometry')))
    summary={}
    for key,rr in results.items():
        cov=coverage_table(rr)
        for q,v in cov.items():
            v['known_nonuseful']=v.pop('harmful');selected=sorted(rr,key=lambda r:-r['p_present'])[:v['selected']]
            v['actual_interventions']=sum(r['intervene'] for r in selected);v['strict_center_worse_than_prediction']=sum(r['kind']==1 and r['err']>r['err_pred'] for r in selected);v['known_unavailable']=sum(r['kind']==0 for r in selected)
        summary[key]=dict(rows=len(rr),interventions=sum(r['intervene'] for r in rr),useful_mean_video=float(np.mean(list(pv[key].values()))),coverage=cov)
    prior.verify_sources(p);assert all(sha(ROOT/n)==h for n,h in manifest.items())
    seedgain={a:{s:paired(v,pv[a+'_fixed']) for s,v in seedpv[a+'_activity'].items()} for a in ARMS}
    write(RUN/'seed_metrics.json',dict(per_video=seedpv,against_fixed=seedgain))
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired=comp,per_video=pv,summary=summary,seeds=seedgain,limitations=p['counterargument']+' No HOTA, calibrated refusal or full KeepTrack reproduction; reused train26.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    elif a.worker:
        try:worker();write(RUN/'status.json',dict(state='completed',stage='18моделей и все причинные сравнения сохранены',updated=now().isoformat(),models_done=18,models_total=18,elapsed_seconds=round(time.monotonic()-T0,2)))
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2)));raise
    else:ap.error('Choose prepare or worker')
