"""Bounded functional prototype of observed target-background discrimination."""
from pathlib import Path
from collections import Counter
from datetime import timedelta
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
RUN=ROOT/'runs/z13_observed_contrast_20261025'
CACHE=ROOT/'runs/z13_candidate_reserve_20261021'
OLD=ROOT/'runs/z13_assignment_effect_20261024/recovery1'
CAUSE=ROOT/'runs/z13_refusal_causes_20261025'
IDENT='z13_observed_contrast_v1'
ARMS=('D','C','R1')


def journal(title,text):
    from scripts import z13_val20_common_20261024 as m
    saved=m.IDENT
    try:m.IDENT=IDENT;m.journal(title,text)
    finally:m.IDENT=saved


def prepare():
    assert not RUN.exists()
    old=read(OLD/'protocol.json')
    assert not read(CAUSE/'normalization/report.json')['primary_pass']
    names=[Path(__file__).relative_to(ROOT).as_posix(),'tracking/z13_observed_contrast.py',
        'tests/test_z13_observed_contrast.py','scripts/z13_observed_contrast_console.ps1',
        'scripts/z13_bounded_stage_20261022.py','tracking/z13_readout.py',
        'runs/z13_local_peak_20261021/geometry.json',(CACHE/'evaluate/labels.json').relative_to(ROOT).as_posix(),
        (OLD/'protocol.json').relative_to(ROOT).as_posix(),(OLD/'report.json').relative_to(ROOT).as_posix(),
        (CAUSE/'report.json').relative_to(ROOT).as_posix(),(CAUSE/'rows.json').relative_to(ROOT).as_posix(),
        'runs/z13_absolute_confidence_20261022/report.json','runs/z13_candidate_identity_20261021/report.json',
        'runs/z13_semantic_sampling_20261023/report.json','tracking/z13_candidate_identity.py',
        'tracking/z13_semantic_sampling.py','tracking/z13_search_conditioned.py']
    for video in old['evaluate26']:
        names += [f'runs/z13_data/{video}.npz',(OLD/'videos'/(video+'.json')).relative_to(ROOT).as_posix()]
    for arm in ARMS:
        names += [x.relative_to(ROOT).as_posix() for x in (CACHE/'evaluate'/arm).glob('cache_*.npz')]
    p=dict(id=IDENT,created=now().isoformat(),device='cpu',budget_seconds=600,
        save_at=(now()+timedelta(seconds=600)).isoformat(),hard_stop=(now()+timedelta(seconds=620)).isoformat(),
        authorization='Standing author authorization for autonomous necessary alternatives until26.10 07:40MSK; author specifically requests causes and repair of mistimed intervention.',
        question='Does explicit comparison with specific previously observed background descriptors improve refusal beyond target-only cosine and the same node on receptors?',
        action='Replace the FlyVis readout confidence by target-versus-background similarity while retaining canonical top1 centers. Observed background prototypes are fixed before search; null score0. This is a concrete inference module prototype,not a fourth diagnostic of the rejected normalization replacement.',
        method='Cached68-dimensional candidate features and last-observed target mean. Background descriptors only from t=3 observed-frame response peaks with their entire lattice neighborhood outside the observed target mask. No query labels or future frames choose negatives. L2-unit descriptors; contrast=cos(query,target)-mean cos(query,observed backgrounds),coefficient1 fixed before results. If no background,exact target-only fallback. Same fixed maximum1% quota,nonpositive margins abstain; no threshold tuning. All8326rows/26reusedtrain perD/C/R1; original score decisions reproduce exactly. No model trained.',
        primary='All three paired-video95%lower bounds strictly>0 on net selected beneficial-minus-harmful actions over all known queries: Dcontrast-Dtargetonly;Dcontrast-DoriginalZ;Dcontrast-R1contrast. Any failure forbids implementation/HOTA promotion of this version. Report unknowns and actual coverage for every policy; no precision or safe-risk claim from known-only outcomes.',
        scope='Internal readout prototype on frozen features,not external tracker guard. Cstock same node reported;R1 same readout noFly control. Geometry and detector/ByteTrack/MASA unchanged. Not isolated proof of anatomical advantage; capacities differ.',
        difference_from_tried='Semantic sampling changed frequency of training examples but did not retain scene-specific negative descriptors during inference. Candidate identity propagated and replaced descriptors through time; this prototype freezes a target/background contrast from the observed frame and makes no temporal assignment or update. Search-conditioned weight predictor and center-utility training are not repeated. Old2%assignment-supervision training gate stays failed;there is no new supervised training here.',
        counterargument='Existing learned readouts already consume target and query features. Simple cosine contrast may add no usable information,mislabel target-like background,or abstain at every useful case. Even a positive cached result may fail with closed-loop states. No causal claim about all representations from failure.',
        alternatives=[dict(node='Fixed observed contrast',status='chosen cheap zero-training prototype',cost='existing caches only'),
            dict(node='Online distractor memory with reliability update',status='later only if observed contrast has signal;old identity update differs',cost='new controlled temporal module'),
            dict(node='Detection-aligned candidate decoder',status='alternative if current point proposals cannot express useful assignment;earlier center-oracle test remains inconclusive',cost='own cheap assignment gate before any extraction')],
        paper=dict(url='https://arxiv.org/html/1808.06048v1',title='Distractor-aware Siamese Networks for Visual Object Tracking',section='3.3,equations2-4',
            transfer='Only target-minus-distractor comparison is borrowed. We use normalizedFlyVis68-vectors,fixed observed prototypes,mean weights1,no incremental update,no learnedSiamRPN,not its reproduction or numerical gains.'),
        limits='One-step cached baseline-state effects only,labels on sparse annotated times,reusedtrain26. Same top1 proposal means this version cannot solve absent useful candidates. NoGPU,newfeatures,weights,dev,20valtuning,HOTA,downloadsor75.',
        sha256={n:sha(ROOT/n) for n in sorted(set(names))})
    write(RUN/'protocol.json',p);write(RUN/'status.json',dict(state='ready',stage='Новый узел:образец цели против наблюдавшегося фона',done=0,total=3,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp);assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: внутреннее считывание цели с фиксированными фоновыми образцами',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8','DaSiamRPN'],
        question=p['question'],hypothesis='Сравнение с конкретным наблюдавшимся фоном может отличать похожий отклик от принадлежности цели лучше одной формы карты;это не установлено.',
        expected=p['method'],acceptance=[dict(criterion='Все8326строк/26видео и исходный отбор84 точно воспроизведены;фон только наблюдаемый;источникинеизменны.',met=None),dict(criterion=p['primary'],met=None)],
        summary='Зарегистрирован один новый исполняемый узел считывания,без обучения.',obtained='',interpretation='',decision='',cannot_claim=p['limits'],metrics=[],commits=[],files=names[:3],
        evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Новый механизм,отличие от прежних вариантов и критерии до исходов')]))
    write(cp,cat)
    journal('новая рабочая версия:образец цели минус наблюдавшийся фон',
        '*Действие.* '+p['action']+'\n\n*Метод.* '+p['method']+'\n\n*Критерии до исходов.* '+p['primary']+
        '\n\n*Отличие от испытанного.* '+p['difference_from_tried']+'\n\n*Контраргумент.* '+p['counterargument']+
        '\n\nОпора:DaSiamRPN,https://arxiv.org/html/1808.06048v1,§3.3,формулы2–4. Переноситсяпринцип,неархитектураиприростстатьи. '
        'CPUдо600с,свойконтроллер,безGPU/обучения/новойHOTA/отложенных75. Прежниенеудачныекритериинеменяются.')


def worker():
    from tracking.z13_readout import HexGeometry,to_eye
    from tracking.z13_observed_contrast import contrast_scores,background_indices,quota_decisions
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    p=read(RUN/'protocol.json');verify_sources(p)
    labels=read(CACHE/'evaluate/labels.json');bound=read(CAUSE/'rows.json')
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    observed={};base=0
    for video in read(OLD/'protocol.json')['evaluate26']:
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:
            masks=z['ex_mask'];offset=z['ep_offset'];crop=z['fr_crop'];pert=z['ep_pert']
        for e in {l['episode'] for l in labels if l['video']==video}:
            k=e-base;assert not pert[k]
            observed[e]=dict(mask=masks[k].copy(),crop=crop[int(offset[k])+3].copy())
        base+=len(masks)
    result={};stats={};chosen={}
    for ai,arm in enumerate(ARMS):
        if (RUN/'stop.request').exists() or now().isoformat()>=p['save_at']:raise TimeoutError('Original prototype deadline')
        write(RUN/'status.json',dict(state='running',stage='Образец против конкретных фоновых образцов;без обучения',variant=arm,done=ai,total=3,updated=now().isoformat()))
        cache={}
        for path in sorted((CACHE/'evaluate'/arm).glob('cache_*.npz')):
            with np.load(path) as z:
                data={k:z[k] for k in ('episode','t','count','centers','features','scores','seed_features')}
            for i,e in enumerate(data['episode']):
                n=int(data['count'][i]);cache[int(e),int(data['t'][i])]=dict(features=data['features'][i,:n],centers=data['centers'][i,:n],score=float(data['scores'][i,0]),seed=data['seed_features'][i])
        prototypes={};observed_stats=[]
        for e,obs in observed.items():
            f=cache[e,3];points=[to_eye(c,obs['crop'],g['eye_size']) for c in f['centers']]
            ids=background_indices(points,obs['mask'],geom)
            prototypes[e]=(f['seed'],f['features'][ids])
            observed_stats.append(dict(episode=e,background_count=len(ids),background_indices=ids))
        outcome={r['label']:r for r in bound if r['arm']==arm}
        source=[];positive=[];contrast=[]
        for label in labels:
            f=cache[label['episode'],label['pos']+4];r=outcome[label['label']]
            assert abs(f['score']-r['score'])<.001
            a,b=contrast_scores(f['features'][:1],*prototypes[label['episode']])
            source.append(r);positive.append(float(a[0]));contrast.append(float(b[0]))
        policies=dict(targetonly=quota_decisions(positive),contrast=quota_decisions(contrast),originalZ=np.asarray([r['selected'] for r in source]))
        assert int(policies['originalZ'].sum())==84
        result[arm]={};chosen[arm]={}
        for mode,selected in policies.items():
            counters=Counter();pv={v:dict(known=0,beneficial=0,harmful=0) for v in sorted({r['video'] for r in source})}
            for yes,row in zip(selected,source):
                if row['kind']>=0:pv[row['video']]['known']+=1
                if not yes:continue
                counters['selected']+=1
                if row['kind']<0:counters['unknown']+=1
                else:
                    counters['known']+=1;counters[row['outcome']]+=1
                    pv[row['video']]['beneficial']+=row['outcome']=='beneficial';pv[row['video']]['harmful']+=row['outcome']=='harmful'
            net={v:(q['beneficial']-q['harmful'])/q['known'] for v,q in pv.items()}
            result[arm][mode]=dict(counts=dict(counters),per_video=pv,net_frequency=net)
            chosen[arm][mode]=[r['label'] for yes,r in zip(selected,source) if yes]
        stats[arm]=dict(episodes=len(prototypes),episodes_with_background=sum(len(bg)>0 for _,bg in prototypes.values()),observed=observed_stats)
        write(RUN/(arm+'_scores.json'),[dict(label=l['label'],video=l['video'],target_score=a,contrast_score=b) for l,a,b in zip(labels,positive,contrast)])
        del cache
    contrasts={}
    for arm in ARMS:
        for baseline in ('targetonly','originalZ'):
            contrasts[arm+'_contrast-'+arm+'_'+baseline]=paired(result[arm]['contrast']['net_frequency'],result[arm][baseline]['net_frequency'])
    contrasts['D_contrast-R1_contrast']=paired(result['D']['contrast']['net_frequency'],result['R1']['contrast']['net_frequency'])
    contrasts['D_contrast-C_contrast']=paired(result['D']['contrast']['net_frequency'],result['C']['contrast']['net_frequency'])
    passed=all(contrasts[k]['ci95'][0]>0 for k in ('D_contrast-D_targetonly','D_contrast-D_originalZ','D_contrast-R1_contrast'))
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=bool(passed),results=result,paired=contrasts,chosen_labels=chosen,prototype_statistics=stats,
        sources_unchanged=True,not_hota=True,core_trained=False,limits=p['limits']))
    write(RUN/'status.json',dict(state='completed',stage='Рабочая версия считывания проверена на сохранённых случаях',done=3,total=3,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise ValueError('Choose --prepare or --worker')
