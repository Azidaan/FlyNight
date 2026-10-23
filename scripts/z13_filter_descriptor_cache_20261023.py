"""Bounded extraction of observed-only filter coefficients; no query frames."""
from pathlib import Path
from datetime import datetime,timedelta
import os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts import z13_localization_labels_20261023 as prior
from scripts.z13_candidate_reserve_20261021 import read,write,sha
RUN=ROOT/'runs/z13_filter_descriptor_cache_20261023';IDENT='z13_filter_descriptor_v1';T0=time.monotonic();now=prior.now
SPLITS={'fit20':(prior.CACHE,'fit'),'fit84':(prior.OLD,'evaluate'),'train26':(prior.CACHE,'evaluate')}


def journal(title,text):
    old=prior.IDENT
    try:prior.IDENT=IDENT;prior.journal(title,text)
    finally:prior.IDENT=old


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('Original budget/Moscow deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not RUN.exists();base=read(prior.RUN/'protocol.json');prior.verify_sources(base)
    names=['tracking/z13_filter_descriptor.py','tests/test_z13_filter_descriptor.py','scripts/z13_filter_descriptor_cache_20261023.py','scripts/z13_filter_descriptor_20261023.py','scripts/z13_filter_descriptor_console.ps1','scripts/z13_target_model.py','tracking/z13_target_center.py','tracking/z13_candidates.py']
    videos={};cachehash={}
    for split,(root,part) in SPLITS.items():
        videos[split]=read(root/'protocol.json')[part]
        for arm in prior.ARMS:
            for path in (root/part/arm).glob('cache_*.npz'):
                assert sha(path)==read(path.with_suffix('.json'))['sha256'];cachehash[path.relative_to(ROOT).as_posix()]=sha(path)
    assert len(videos['fit20'])==20 and len(videos['fit84'])==84 and len(videos['train26'])==26
    assert len(set(sum(videos.values(),[])))==130
    names.extend(f'runs/z13_data/{v}.npz' for v in sum(videos.values(),[]))
    for arm in prior.ARMS:names.extend([f'runs/z13_target_center_dev_20261018/{arm}/normalization.json',f'runs/z13_target_center_20261018/{arm}/choice.json'])
    deadline=now().replace(hour=7,minute=40,second=0,microsecond=0);assert now()<deadline
    p=dict(id=IDENT,created=now().isoformat(),authorization='Author requested continued autonomous necessary runs; preparation resumed24.10.2026. Conservative end07:40MSK24.10; Saturday access also permitted by AGENTS9.1.',device='cpu',budget_seconds=5400,save_at=(deadline-timedelta(minutes=5)).isoformat(),hard_stop=(deadline-timedelta(seconds=30)).isoformat(),stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),videos=videos,
      action='Replace mean foreground descriptor supplied to candidate ranker with68coefficients of the existing observed-center ridge target filter. Same3peaks/queryfeatures/geometry. Foreground+background affect filter fitting; no query labels enter descriptor.',
      model='Matched dual embeddings: separate68-LayerNorm-linear16-tanh for query and descriptor, existing88-32-ReLU-1scorehead. Both mean and ridge conditions share initialization, capacity, optimizer and sample order. Do not compare only against the smaller shared-embedding old ranker.',
      training='36models:D/C/R1 x ridge/mean x activity/geometry x3seeds. Train104 original labels/classweights,1150updates145950presentations;ensemblemeanlogits. No search-conditioned prediction, semantic sampling or future memory.',
      primary='D ridge-D mean >=0.02 and lower95%paired-videoCI>0. D ridge also beats old D activity,old D geometry,R1 ridge and D ridge geometry, all lower95>0. All required; C cannot rescue.',
      extraction='Only input frames0..3 and observed ex_mask, no query frame or labels read into feature API. Existing fixed normalizer and lambda. CPU frozen core in original groups of8; save each completed group. Existing candidates are unchanged.',
      parity='Before any efficacy: descriptor projection equals original ridge_scores on observed features within1e-10; reconstructed seed features maxabs<=0.001; observed-frame3 candidate count equal, centers<0.5pixel,scores<0.001. These are technical transfer tolerances, never loosen after failure. Check every episode. Mean control uses exact old cached mean, not reconstructed one.',
      timing='First8D episodes retained; conservative2x total groups cost+180s must fit remaining5400s. Training separate CPU process at most1800s after complete extraction. No completed group repeated.',
      counterargument='Current response map already uses this filter; coefficients may add no information. Coefficients are not feature means, hence separate descriptor embedding in both matched arms. This test does not diagnose intrinsic representational limits.',
      paper='https://arxiv.org/html/1904.07220v2 section3/3.1: target model as filter weights incorporating foreground and background. Supplying these weights to a peak ranker is our hypothesis, not a reported DiMP component or guaranteed gain.',
      sha256={**base['sha256'],**cachehash,**{n:sha(ROOT/n) for n in names}})
    write(RUN/'protocol.json',p)
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: коэффициенты модели цели вместо среднего образца',area='flyvis',state='planned',started=f'{now():%Y-%m-%d}',tags=['fly-problem-P8','DiMP','engineering-future'],question='Помогают ли коэффициенты уже построенного различающего фильтра при выборе пика сверх среднего образца и карты?',hypothesis='Среднее образца не передаёт различие цели и наблюдаемого фона; коэффициенты фильтра могут передать его. Нехватка этой информации пока не доказана.',expected='Наблюдаемые4кадра,фиксированные нормы/lambda,36равныхCPUмоделей,104train/26оценки,3сида; извлечение5400с и обучение1800с, независимые контроллеры до07:40.',acceptance=[dict(criterion='Все наблюдаемые признаки и рамки воспроизведены в заранее заданных допусках, исходные18моделей и32шага сверены;36моделей и хеши проверены.',met=None),dict(criterion='D ridge-D mean >=2п.п., нижняя95%повидеоДИ>0.',met=None),dict(criterion='D ridge лучше старой Dсактивностью,старой картыD,R1 ridge и D ridge безактивности; все нижние95%ДИ>0.',met=None)],summary='Постановка до исходов; первая стадия — воспроизведение и стоимость.',done='',obtained='',interpretation='',decision='',cannot_claim='Это авторская адаптация фильтра как описания цели, не воспроизведение DiMP. Карта уже зависит от фильтра. Поисковый train26, нет HOTA/отказа/доказательства анатомии.',metrics=[],commits=[],files=names[:5],evidence=[dict(kind='file',ref='runs/z13_filter_descriptor_cache_20261023/protocol.json',claim='Механизм, равный контроль, допуски и критерии до исходов')]))
    write(cp,cat);journal('постановка описания цели коэффициентами фильтра','*Постановка.* '+p['action']+'\n\n'+p['counterargument']+'\n\nПротокол `runs/z13_filter_descriptor_cache_20261023/protocol.json`. Основание DiMP §3/3.1: модель цели как веса различающего фильтра. Новое здесь — подать уже существующие веса выборщику вместо среднего; сам фильтр, карта и ядро прежние. Равные отдельные кодировщики в обеих версиях,36моделей. До исходов: прирост D>=2п.п. и положительный интервал, также сильные контроли. Сначала8эпизодов только для стоимости/воспроизведения; CPU5400с+20, затем отдельное обучение1800с, до07:40. Текущую D на стенде не заменять.')


def worker():
    import gc,numpy as np,torch
    torch.set_num_threads(2);torch.set_default_device('cpu');assert not torch.cuda.is_available()
    from scripts import z13_target_model as old
    from tracking.z13_filter_descriptor import fit_descriptor
    from tracking.z13_target_center import ridge_scores,observed_center_target
    from tracking.z13_candidates import candidates
    from tracking.z13_readout import to_frame
    p=read(RUN/'protocol.json');prior.verify_sources(p);inputs={};sources={};groups_total=0
    for split,(root,part) in SPLITS.items():
        status('Чтение только наблюдаемых входов',variant=split)
        with np.load(next((root/part/'D').glob('cache_*.npz'))) as z:assert 3 in z['t']
        reference={}
        for arm in prior.ARMS:
            ref={}
            for path in sorted((root/part/arm).glob('cache_*.npz')):
                with np.load(path) as z:
                    for i in np.flatnonzero(z['t']==3):ref[int(z['episode'][i])]={k:z[k][i].copy() for k in ('seed_features','centers','scores','count')}
            reference[arm]=ref
        eps=sorted(reference['D']);assert all(sorted(reference[a])==eps for a in prior.ARMS)
        data={};base=0
        for v in p['videos'][split]:
            with np.load(ROOT/f'runs/z13_data/{v}.npz') as z:
                off=z['ep_offset'];needed=[e for e in eps if base<=e<base+len(off)-1]
                if needed:
                    hx=z['fr_hex'];mask=z['ex_mask'];crop=z['fr_crop']
                    for e in needed:
                        k=e-base;data[e]=(hx[off[k]:off[k]+4].astype(np.float32),mask[k].copy(),crop[off[k]+3].copy())
                base+=len(off)-1
        assert set(data)==set(eps);inputs[split]=(data,reference,eps);groups_total+=3*((len(eps)+7)//8)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False);done=0;summary={}
    for arm in prior.ARMS:
        status('Замороженное ядро только на4наблюдаемых кадрах',variant=arm,done=done,total=groups_total)
        bank,geom,bb=old.build_arm(arm,torch,ck);assert bb is None or str(bb.dev)=='cpu'
        normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{arm}/normalization.json');mu=np.asarray(normal['mean']);sd=np.asarray(normal['sd']);lam=read(ROOT/f'runs/z13_target_center_20261018/{arm}/choice.json')['center']['lambda_value']
        for split,(data,refs,eps) in inputs.items():
            descriptors={};checks=[]
            for left in range(0,len(eps),8):
                status('Коэффициенты из образца; сверка старых признаков',variant=arm+' '+split,done=done,total=groups_total)
                path=RUN/arm/split/f'chunk_{left:05d}.json';ch=eps[left:left+8]
                if path.exists():record=read(path)
                else:
                    start=time.monotonic();hx=torch.as_tensor(np.stack([data[e][0] for e in ch]),device='cpu');f=old.episode_features(torch,arm,bb,hx,geom.neighbors,{3})[3];record=dict(descriptors={},checks=[])
                    for j,e in enumerate(ch):
                        x=(f[j].T-mu)/sd;mask=data[e][1];y=observed_center_target(mask,geom.points,geom.spacing);w,m,b=fit_descriptor(x,y,lam);score=(x-m)@w+b
                        exact=float(np.abs(score-ridge_scores(x,y,x,[lam])[0]).max());r=candidates(score,x,geom);n=len(r['indices']);ref=refs[arm][e]
                        seed_error=float(np.abs(x[mask].mean(0)-ref['seed_features']).max());centers=np.asarray([to_frame(c,data[e][2],bank.eye_size) for c in r['centers']])
                        if n!=int(ref['count']):
                            write(RUN/'parity_failure.json',dict(arm=arm,split=split,episode=e,new_count=n,old_count=int(ref['count'])));raise AssertionError('Observed candidate count differs')
                        center_error=float(np.abs(centers-ref['centers'][:n]).max());score_error=float(np.abs(r['scores']-ref['scores'][:n]).max())
                        check=dict(episode=e,projection_max_abs=exact,seed_max_abs=seed_error,center_max_px=center_error,score_max_abs=score_error)
                        if not (exact<=1e-10 and seed_error<=.001 and center_error<.5 and score_error<.001):
                            write(RUN/'parity_failure.json',dict(arm=arm,split=split,**check));raise AssertionError(check)
                        record['descriptors'][str(e)]=w.tolist();record['checks'].append(check)
                    record['seconds']=time.monotonic()-start;write(path,record)
                descriptors.update(record['descriptors']);checks.extend(record['checks']);done+=1
                if not (RUN/'timing.json').exists():
                    estimate=record['seconds']*groups_total*2+180;available=min(p['budget_seconds']-(time.monotonic()-T0),(datetime.fromisoformat(p['save_at'])-now()).total_seconds())
                    write(RUN/'timing.json',dict(first8_seconds=record['seconds'],groups=groups_total,conservative_seconds=estimate,available_seconds=available,proceed=estimate<available));assert estimate<available,'Full extraction exceeds measured budget'
            write(RUN/arm/split/'descriptors.json',descriptors);write(RUN/arm/split/'reproduction.json',checks);summary[arm+'_'+split]=dict(episodes=len(eps),max_seed=max(v['seed_max_abs'] for v in checks),max_center=max(v['center_max_px'] for v in checks),max_score=max(v['score_max_abs'] for v in checks))
        del bb,bank;gc.collect()
    prior.verify_sources(p);write(RUN/'report.json',dict(parity_pass=True,summary=summary,observed_frames_only=True,no_efficacy_claim=True));write(RUN/'status.json',dict(state='completed',stage='Коэффициенты и воспроизведение сохранены',updated=now().isoformat(),done=done,total=groups_total))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=str(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
