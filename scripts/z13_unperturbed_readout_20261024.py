"""Runtime-like training crop sampling, with a size-matched mixed-data control."""
from pathlib import Path
from datetime import datetime,timedelta
import hashlib
import json
import math
import os
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_unperturbed_readout_20261024'
CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
EXTRA=ROOT/'runs/z13_candidate_data104_20261021'
W=ROOT/'runs/z13_weekend_20261024'
IDENT='z13_unperturbed_readout_v1';ARMS=('D','C','R1');T0=time.monotonic()


def journal(title,text):
    from scripts import z13_val20_common_20261024 as S
    old=S.IDENT
    try:S.IDENT=IDENT;S.journal(title,text)
    finally:S.IDENT=old


def flags(labels,videos):
    """Bind every label to its original local episode; no outcome-derived sampling."""
    result=np.zeros(len(labels),bool);offset=0
    for v in videos:
        with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
            pert=z['ep_pert'];ep=z['lab_ep'];pos=z['lab_pos'];kind=z['lab_kind'];pred=z['lab_pred']
        local={(int(e),int(t)):i for i,(e,t) in enumerate(zip(ep,pos))}
        for i,l in enumerate(labels):
            if l['video']!=v:continue
            e=l['episode']-offset;j=local[(e,l['pos'])]
            assert int(kind[j])==l['kind'] and np.array_equal(pred[j],l['pred'])
            result[i]=bool(pert[e])
        offset+=len(pert)
    assert {l['video'] for l in labels}<=set(videos)
    return result


def select_rows(labels,perturbed):
    groups={}
    for i,l in enumerate(labels):
        if l['kind']>=0:groups.setdefault((l['video'],l['kind']),[]).append(i)
    clean=[];mixed=[];details={}
    for key,indices in sorted(groups.items()):
        u=[i for i in indices if not perturbed[i]]
        order=sorted(indices,key=lambda i:hashlib.sha256(('mixed-crop42:'+labels[i]['video']+':'+str(labels[i]['episode'])+':'+str(labels[i]['pos'])).encode()).hexdigest())
        m=order[:len(u)];clean+=u;mixed+=m
        details[str(key)]=dict(clean=len(u),mixed=len(m),mixed_perturbed=sum(bool(perturbed[i]) for i in m))
    assert len(clean)==len(mixed)>0 and not any(perturbed[i] for i in clean)
    return dict(clean=np.asarray(clean,np.int64),mixed=np.asarray(mixed,np.int64)),details


def measure(rows):
    groups={}
    for row in rows:groups.setdefault(row['video'],[]).append(row)
    yield_values={};center_values={};details={}
    for video,items in groups.items():
        k=max(1,math.ceil(.05*len(items)));selected=sorted(items,key=lambda x:-x['p_present'])[:k]
        useful=sum(x['kind']>=0 and bool(x.get('useful',False)) for x in selected)
        unknown=sum(x['kind']<0 for x in selected)
        yield_values[video]=useful/k
        positives=[x for x in items if x['kind']==1]
        if positives:center_values[video]=sum(bool(x['useful']) for x in positives)/len(positives)
        details[video]=dict(all_rows=len(items),selected=k,confirmed_useful=useful,unknown=unknown,
            confirmed_yield_lower=useful/k,possible_yield_upper=(useful+unknown)/k,
            known_precision=useful/(k-unknown) if k>unknown else None,
            strict_center_worse=sum(x['kind']==1 and x['err']>x['err_pred'] for x in selected))
    return yield_values,center_values,details


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or now()>=datetime.fromisoformat(p['save_at']) or time.monotonic()-T0>p['budget_seconds']:raise TimeoutError('Original pilot budget/deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),models_total=36,**kw))


def prepare():
    assert not RUN.exists()
    base=read(EXTRA/'protocol.json');a=read(W/'authorization.json')
    names=[Path(__file__).relative_to(ROOT).as_posix(),'scripts/z13_unperturbed_readout_console.ps1',
        'scripts/z13_publish_readout_sampling_20261024.py','tests/test_z13_unperturbed_readout.py',
        'scripts/z13_candidate_ranker_20261021.py','tracking/z13_candidate_ranker.py',
        'tracking/z13_matched_data_budget.py','scripts/z13_localization_labels_20261023.py',
        'scripts/z13_data.py','scripts/z13_bounded_stage_20261022.py',
        'runs/z13_candidate_data104_20261021/protocol.json','runs/z13_candidate_reserve_20261021/protocol.json',
        'runs/z13_candidate_reserve_20261021/fit/labels.json','runs/z13_candidate_data104_20261021/evaluate/labels.json',
        'runs/z13_candidate_reserve_20261021/evaluate/labels.json']
    for v in base['train_full104']+base['evaluation26']:names.append(f'runs/z13_data/{v}.npz')
    for arm in ARMS:
        for folder in (CACHE/'fit'/arm,CACHE/'evaluate'/arm,EXTRA/'evaluate'/arm):
            names += [x.relative_to(ROOT).as_posix() for x in folder.iterdir() if x.suffix in ('.npz','.json')]
        for mode in ('activity','geometry'):
            names.append(f'runs/z13_candidate_data104_20261021/{arm}/{mode}/ensemble_rows.json')
            for seed in (0,1,2):
                names += [f'runs/z13_candidate_data104_20261021/{arm}/{mode}/seed{seed}/{n}' for n in ('model.pt','training.json')]
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=1800,
        save_at=min(now()+timedelta(seconds=1800),datetime.fromisoformat(a['save_at'])).isoformat(),
        hard_stop=min(now()+timedelta(seconds=1820),datetime.fromisoformat(a['hard_stop'])).isoformat(),
        action='Change only fitting-sample distribution inside the learned FlyVis candidate readout:unperturbed episodes versus equally sized mixed episodes. No new features,loss,core weights,candidates or architecture.',
        train104=base['train_full104'],fit20=base['fit'],extra84=base['evaluate'],evaluate26=base['evaluation26'],
        sampling='Use original ep_pert flag. For each train-video and physical kind>=0 use all unperturbed rows;control has exactly that count from all original rows sorted by SHA256(mixed-crop42:video:episode:pos). Same counts per video/kind;zero-count strata omitted in both. Fixed selection shared by all arms,seeds and ablations. This is conditional subsampling,not paired regeneration of identical scenes.',
        primary='Evaluate only originally unperturbed episodes of all26reused train videos. Select ceil(0.05*N) from ALL eligible rows per video before reading outcomes. Confirmed yield=known-useful/ALL selected;unknown cases separately give an upper bound (useful+unknown)/selected,not observed errors. Dclean-Dmixed confirmed-yield mean>=0.05 and paired95%lower>0. Dclean must beat cleanDgeometry and cleanR1activity with both lower95>0. Useful-center mean on kind1 must not fall below mixedD. All26 must have eligible and positive rows. All required;otherwise no expensive transfer.',
        training='Same BCE candidate useful labels,original saved positive weight for each arm/mode/seed,AdamW lr0.001 weight_decay0.01,clip10,50referenceepochs,nref2919,batch128,1150updates/145950row presentations,initial seeds0/1/2. Three-seed mean logits,never pick best seed. Old18full-data models independently replayed as secondary reference. Same current/past information;unperturbed flag used only for sampling/evaluation grouping,not as network input.',
        controls='C/D/R1,activity and geometry only,clean and size-matched mixed,3seeds=36newCPUmodels. Same all26unperturbed evaluation rows/quotas. Report full mixed-evaluation secondary as well;unknown bounds and worse-center counts. No change to previous failed candidate/null criterion or its historical result.',
        counterargument='Artificial crop shifts may improve robustness;removing them can reduce diversity and overfit. Current fit samples are naturally different episodes,not counterfactual copies;even matched video/kind/count leaves object/age/size differences. Unperturbed baseline episodes are only closer to runtime,not actual corrected closed-loop state. A confirmed-yield gain can reflect unknown-label composition,not real precision;bounds reported. This exploratory repeatedly used26train pilot cannot establish cause,HOTA or seed independence of the core.',
        sources=[dict(url='https://arxiv.org/html/2105.11595v1',section='3.4,5.2/Table2',support='Distribution of tracking training triplets matters for false continuation. No claim SiamMOT tests removing crop perturbations or predicts our effect.')],
        evidence_of_difference='scripts/z13_data.py:PERTURB_MAX=0.25,JITTER=0.05 and ep_pert;tracking/z13_ridge_localizer.py:runtime crop from current prediction without random offset. Documentation establishes a regime difference,not that it is harmful.',
        sha256={n:sha(ROOT/n) for n in names})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Подготовка парного CPU-сравнения выборок',updated=now().isoformat(),models_done=0,models_total=36))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(x['id']==IDENT for x in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: обучение считывателя на эпизодах без искусственного сдвига окна',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8'],
        question='Поможет ли согласование режима обучающих вырезок с применением при равном размере и бюджете?',
        hypothesis='Удаление искусственных сдвигов может улучшить выбор полезных ответов;польза и причинность не установлены.',
        expected=p['action']+' '+p['training'],acceptance=[dict(criterion=p['primary'],met=None),dict(criterion='36моделей,3сида,полное воспроизведение18старых моделей,старой32шаговойтраектории и неизменность источников.',met=None)],
        summary='Отдельная замена обучающей выборки,критерии до исходов.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument'],metrics=[],commits=[],files=[names[0]],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новый режим выборки и все контроли до обучения')]))
    write(cp,cat)
    gp=ROOT/'experiments/metrics_glossary.json';g=read(gp)
    g['metrics']['confirmed_useful_selected_fraction']=dict(title='Подтверждённая доля полезных среди всех выбранных',ru='Число выбранных ответов с подтверждённой полезностью делится на число всех выбранных,включая неизвестные исходы. Это наблюдаемая нижняя граница доли полезных,а не точность среди известных.',better='higher',unit='доля',range=[0,1],tool='scripts/z13_unperturbed_readout_20261024.py:measure',caveat='Неизвестные не объявляются ошибочными:верхняя граница=(полезные+неизвестные)/все выбранные. НеHOTA и не гарантия риска.',source='scripts/z13_unperturbed_readout_20261024.py:measure')
    write(gp,g)
    journal('постановка согласования обучающих вырезок с применением',
        '*Действие.* '+p['action']+'\n\n*Критерии до прогона.* '+p['primary']+'\n\n'
        '*Контроли.* '+p['sampling']+' '+p['controls']+'\n\n*Возражение.* '+p['counterargument']+'\n\n'
        'Из кода подтверждено различие случайного сдвига/дрожания,а не его вред. SiamMOT§5.2,'
        'https://arxiv.org/html/2105.11595v1 показывает зависимость от состава обучающих троек,не эту нашу модификацию. '
        'CPU1800с,сначала точное воспроизведение/32шага и консервативная цена36моделей;безGPU,ядра,dev,val20подбора,75и загрузок. '
        'Прежние отрицательные критерии не меняются. Новый показатель подтверждённой доли описан в словаре до использования.')


def load_data(arm):
    import scripts.z13_candidate_identity_20261021 as reader
    from scripts.z13_candidate_ranker_20261021 import pack
    l20=read(CACHE/'fit/labels.json');l84=read(EXTRA/'evaluate/labels.json');ev=read(CACHE/'evaluate/labels.json')
    seq,m1=reader.load('fit',arm);a=pack(l20,seq)
    prior=reader.CACHE
    try:reader.CACHE=EXTRA;seq,m2=reader.load('evaluate',arm)
    finally:reader.CACHE=prior
    b=pack(l84,seq);ta=tuple(np.concatenate([x,y]) for x,y in zip(a,b))
    seq,m3=reader.load('evaluate',arm);ea=pack(ev,seq)
    return ta,ea,{**m1,**m2,**m3}


def worker():
    import gc,torch
    from scripts.z13_localization_labels_20261023 import verify_sources,train_steps
    from scripts.z13_candidate_ranker_20261021 import predict,make_rows
    from tracking.z13_candidate_ranker import make_ranker
    from scripts.z13_continue import paired
    torch.set_num_threads(2);torch.set_default_device('cpu')
    q=torch.tensor(2.,requires_grad=True);(q*q).backward();assert q.grad.item()==4 and q.device.type=='cpu' and not torch.cuda.is_initialized()
    p=read(RUN/'protocol.json');verify_sources(p)
    l20=read(CACHE/'fit/labels.json');l84=read(EXTRA/'evaluate/labels.json');labels=l20+l84;ev=read(CACHE/'evaluate/labels.json')
    status('Сверка признаков сдвига эпизодов',models_done=0)
    pert=np.concatenate([flags(l20,p['fit20']),flags(l84,p['extra84'])]);evpert=flags(ev,p['evaluate26'])
    choices,detail=select_rows(labels,pert);keep=np.flatnonzero(~evpert)
    assert {ev[i]['video'] for i in keep}==set(p['evaluate26'])
    assert {ev[i]['video'] for i in keep if ev[i]['kind']==1}==set(p['evaluate26'])
    write(RUN/'sampling.json',dict(strata=detail,clean=choices['clean'].tolist(),mixed=choices['mixed'].tolist(),eval_unperturbed_indices=keep.tolist(),fit_clean_rows=len(choices['clean']),fit_mixed_rows=len(choices['mixed']),eval_rows=len(keep)))
    manifest={};repro={};seedrows={};rows={};secondary={};done=0
    for arm in ARMS:
        status('Загрузка неизменных кандидатов',variant=arm,models_done=done)
        ta,ea,m=load_data(arm);manifest.update(m)
        for mode,activity in (('activity',True),('geometry',False)):
            oldpred=[]
            for seed in (0,1,2):
                model=make_ranker(torch,activity);model.load_state_dict(torch.load(EXTRA/arm/mode/f'seed{seed}/model.pt',map_location='cpu',weights_only=True));oldpred.append(predict(torch,model,ea))
            original=make_rows(ev,ea,np.mean(oldpred,0));saved=read(EXTRA/arm/mode/'ensemble_rows.json')
            assert len(original)==len(saved)==8326
            delta=max(abs(a['p_present']-b['p_present']) for a,b in zip(original,saved))
            assert delta<=1e-5 and all(a['center']==b['center'] and a.get('useful')==b.get('useful') for a,b in zip(original,saved))
            repro[arm+'_'+mode]=dict(rows=8326,max_abs_score=delta,centers_exact=True)
            rows['oldfull_'+arm+'_'+mode]=[original[i] for i in keep]
            yy,cc,dd=measure(original);secondary['oldfull_'+arm+'_'+mode]=dict(yield_per_video=yy,centers_per_video=cc,quota_details=dd)
            if arm=='D' and mode=='activity':
                ref=read(EXTRA/'D/activity/seed0/training.json');ids=np.flatnonzero([l['kind']>=0 for l in labels])
                torch.manual_seed(0);model=make_ranker(torch,True);tensors=[torch.as_tensor(a[ids],device='cpu') for a in ta[:5]]
                loss,_=train_steps(torch,model,tensors,2919,0,ref['pos_weight'],maxsteps=32)
                assert max(abs(a-b) for a,b in zip(loss,ref['loss'][:32]))<=1e-6
                torch.manual_seed(0);model=make_ranker(torch,True);tensors=[torch.as_tensor(a[choices['mixed']],device='cpu') for a in ta[:5]]
                began=time.monotonic();train_steps(torch,model,tensors,2919,0,ref['pos_weight'],maxsteps=32);seconds=time.monotonic()-began
                estimate=2*seconds/32*1150*36+180
                write(RUN/'price.json',dict(steps32_seconds=seconds,conservative_training_seconds=estimate,available_seconds=p['budget_seconds']-(time.monotonic()-T0)))
                assert estimate<p['budget_seconds']-(time.monotonic()-T0),'Price exceeds fixed pilot budget'
            for regime in ('mixed','clean'):
                predictions=[];key=regime+'_'+arm+'_'+mode
                tensors=[torch.as_tensor(a[choices[regime]],device='cpu') for a in ta[:5]]
                for seed in (0,1,2):
                    ref=read(EXTRA/arm/mode/f'seed{seed}/training.json');torch.manual_seed(seed);model=make_ranker(torch,activity)
                    folder=RUN/key/f'seed{seed}';status('Обучение считывателя',variant=key+' seed'+str(seed),models_done=done,done=0,total=1150)
                    loss,n=train_steps(torch,model,tensors,2919,seed,ref['pos_weight'],progress=lambda step:status('Обучение считывателя',variant=key+' seed'+str(seed),models_done=done,done=step,total=1150))
                    assert len(loss)==1150 and n==145950
                    folder.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),folder/'model.pt')
                    write(folder/'training.json',dict(seed=seed,steps=len(loss),presented=n,pos_weight=ref['pos_weight'],loss=loss,model_sha256=sha(folder/'model.pt')))
                    prediction=predict(torch,model,ea);predictions.append(prediction)
                    rr=make_rows(ev,ea,prediction);selected=[rr[i] for i in keep];a,b,d=measure(selected)
                    seedrows[key+'_'+str(seed)]=dict(yield_per_video=a,centers_per_video=b)
                    write(folder/'done.json',dict(at=now().isoformat(),prediction_finite=bool(np.isfinite(prediction).all())))
                    done+=1
                rr=make_rows(ev,ea,np.mean(predictions,0));rows[key]=[rr[i] for i in keep]
                yy,cc,dd=measure(rr);secondary[key]=dict(yield_per_video=yy,centers_per_video=cc,quota_details=dd)
                write(RUN/key/'ensemble_rows.json',rows[key]);write(RUN/key/'all_rows.json',rr)
        del ta,ea;gc.collect()
    write(RUN/'reproduction.json',repro);write(RUN/'cache_manifest.json',manifest);write(RUN/'seed_metrics.json',seedrows)
    yields={};centers={};details={}
    for k,rr in rows.items():yields[k],centers[k],details[k]=measure(rr)
    contrasts={};cc={}
    for arm in ARMS:
        a='clean_'+arm+'_activity'
        for b in ('mixed_'+arm+'_activity','clean_'+arm+'_geometry','oldfull_'+arm+'_activity'):
            contrasts[a+'-'+b]=paired(yields[a],yields[b]);cc[a+'-'+b]=paired(centers[a],centers[b])
    a='clean_D_activity';b='clean_R1_activity';contrasts[a+'-'+b]=paired(yields[a],yields[b]);cc[a+'-'+b]=paired(centers[a],centers[b])
    main=contrasts['clean_D_activity-mixed_D_activity'];control=[contrasts['clean_D_activity-clean_D_geometry'],contrasts['clean_D_activity-clean_R1_activity']]
    criteria=dict(complete=all(x['n']==26 for x in [main,*control]) and done==36,
        main=main['mean_per_video']>=.05 and main['ci95'][0]>0,
        controls=all(x['ci95'][0]>0 for x in control),centers_not_lower=cc['clean_D_activity-mixed_D_activity']['mean_per_video']>=0)
    verify_sources(p);assert all(sha(ROOT/n)==h for n,h in manifest.items())
    write(RUN/'report.json',dict(primary_pass=all(criteria.values()),criteria=criteria,paired_confirmed_yield=contrasts,paired_centers=cc,
        summary={k:dict(confirmed_yield=float(np.mean(list(v.values()))),useful_centers=float(np.mean(list(centers[k].values())))) for k,v in yields.items()},
        yield_per_video=yields,centers_per_video=centers,quota_details=details,models_done=done,three_seed_ensembles=True,
        original18models_replayed=True,original32step_trajectory_reproduced=True,secondary_all_original_evaluation_rows=secondary,
        eval_regime='Originally unperturbed train26 episodes only;no val20 tuning',not_hota=True))
    write(RUN/'status.json',dict(state='completed',stage='36моделей готовы;проверка итогов',updated=now().isoformat(),models_done=36,models_total=36))


def record():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p);e=read(RUN/'controller_result.json')
    assert e['exit_code']==0 and e['all_owned_ended'] and e['worker_ended'] and not e['remaining']
    ps=WindowsProcesses();assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items())
    for n,h in read(RUN/'cache_manifest.json').items():assert sha(ROOT/n)==h
    dirs=list(RUN.glob('*/seed*/done.json'));assert len(dirs)==36
    for f in dirs:assert sha(f.parent/'model.pt')==read(f.parent/'training.json')['model_sha256']
    r=read(RUN/'report.json')
    for name,c in r['paired_confirmed_yield'].items():
        a,b=name.split('-');assert paired(r['yield_per_video'][a],r['yield_per_video'][b])==c
    for path in RUN.glob('*/ensemble_rows.json'):
        y,c,d=measure(read(path));key=path.parent.name
        assert y==r['yield_per_video'][key] and c==r['centers_per_video'][key] and d==r['quota_details'][key]
    cp=ROOT/'experiments/project_history.json';cat=read(cp);card=next(c for c in cat['experiments'] if c['id']==IDENT)
    source=(RUN/'report.json').relative_to(ROOT).as_posix();head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='Согласование выборки '+('прошло все поисковые критерии;перенос ещё не измерен.' if r['primary_pass'] else 'не прошло все поисковые критерии;в стенд не переносится.'),
        obtained=source+' → paired_confirmed_yield/paired_centers/summary: '+json.dumps(dict(yield_contrasts=r['paired_confirmed_yield'],centers=r['paired_centers']),ensure_ascii=False),
        interpretation='Проверена конкретная условная выборка,не причинная роль аугментации вообще. Подтверждённая доля — нижняя граница с отдельно сохранёнными неизвестными,не HOTA.',
        decision='Только при всех пройденных критериях отдельно зарегистрировать фиксированное причинное применение/перенос;лучший сид и порог по26не выбирать.' if r['primary_pass'] else 'Не продвигать эту версию к дорогому переносу;сохранить все3сида и исходы,выбрать другой узел по измерениям.')
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Полный парный CPU-итог выборки.'))
        card[k]=v
    card.update(state='completed_positive' if r['primary_pass'] else 'completed_negative',done=now().strftime('%Y-%m-%d'))
    card['acceptance'][0].update(met=r['primary_pass'],evidence=source)
    card['acceptance'][1].update(met=True,evidence='controller_result.json;reproduction.json;cache_manifest.json;36done.json')
    card['confirmed_results']=[]
    if r['primary_pass']:
        card['confirmed_results'].append(dict(id=IDENT+'_confirmed_yield',title='Больше подтверждённых полезных ответов при5%отборе;не HOTA и не полная точность',
            category='readout',metric='confirmed_useful_selected_fraction',unit='fraction',comparison='clean_D_activity-mixed_D_activity',
            subset='Только исходно несдвинутые эпизоды повторно использованных26train',conditions=p['sampling']+' '+p['training'],
            limitations=p['counterargument'],evidence=dict(source=source,sha256=sha(RUN/'report.json'),locator=['paired_confirmed_yield','clean_D_activity-mixed_D_activity'])))
    write(cp,cat);journal('итог обучения без искусственного сдвига окна','*Получено.* '+fields['obtained']+'\n\n*Вывод.* '+fields['interpretation']+'\n\n*Решение.* '+fields['decision']+'\n\nК HOTA проценты не пересчитываются;доля потолка стенда неизвестна.')
    write(RUN/'finish_recorded.json',dict(at=now().isoformat(),report_sha256=sha(RUN/'report.json'),models_verified=36,all_owned_ended=True,primary_pass=r['primary_pass']))


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    ps=WindowsProcesses();write(RUN/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat()))
    while not (RUN/'controller_result.json').exists():time.sleep(3)
    e=read(RUN/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended']:
        write(RUN/'monitor_failure.json',dict(at=now().isoformat(),result=e));return
    # Startup verification may run concurrently with fitting. Keep catalog/plan
    # stable during its tests; never race completion recording against them.
    while not (W/'readout_sampling_start_publication.json').exists():
        if now()>=datetime.fromisoformat(read(RUN/'protocol.json')['hard_stop']):
            write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error='Startup publication not verified before original deadline;full outputs retained.'));return
        time.sleep(3)
    record()
    from scripts.z13_publish_readout_sampling_20261024 import checks_and_publish
    checks_and_publish('finish')


if __name__=='__main__':
    if sys.argv[1]=='--prepare':prepare()
    elif sys.argv[1]=='--worker':
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif sys.argv[1]=='--monitor':
        try:monitor()
        except Exception as error:write(RUN/'monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif sys.argv[1]=='--record':record()
    else:raise ValueError(sys.argv[1])
