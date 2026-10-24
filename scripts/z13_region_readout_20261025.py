"""A bounded zero-training prototype: response regions -> assignment costs."""
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
import inspect
import hashlib
import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
RUN = ROOT / 'runs/z13_region_readout_20261025'
OLD = ROOT / 'runs/z13_assignment_effect_20261024/recovery1'
MAPS = ROOT / 'runs/z13_query_memory_20261021'
W = ROOT / 'runs/z13_weekend_20261024'
IDENT = 'z13_region_readout_v1'
ARMS = ('D','C','R1','geometry')
ALPHAS = (0., .25, .5, 1.)


def journal(title, text):
    from scripts import z13_val20_common_20261024 as source
    old = source.IDENT
    try:
        source.IDENT = IDENT
        source.journal(title, text)
    finally:
        source.IDENT = old


def replay_source():
    from scripts.z13_assignment_effect_20261024 import replay_video
    source = inspect.getsource(replay_video)
    old = '''                        for arm in ARMS:
                            alts=[]
                            for center in centers[q['uid']][arm]:
                                t.mean[:2]=center;altered=costs.copy();altered[ri]=self.get_dists([t],detections)[0]
                                alts.append(assignment_vector(altered,self.args.match_thresh))
                            event['alternatives'][arm]=alts'''
    new = '''                        for arm, candidates in new_cost_rows(q,costs[ri],detections).items():
                            alts=[]
                            for row_cost in candidates:
                                altered=costs.copy();altered[ri]=row_cost
                                alts.append(assignment_vector(altered,self.args.match_thresh))
                                if len(alts)==1:assert np.array_equal(alts[0],base)
                            event['alternatives'][arm]=alts'''
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = '''                    for j,outcome in enumerate(rr['arms'][arm]):
                        l=q['label']
                        if l['kind']==1:
                            error=np.linalg.norm(centers[q['uid']][arm][j]-l['gt']);prior=np.linalg.norm(np.asarray(l['pred'])-l['gt'])
                            outcome['old_center_useful']=bool(error<=.25*l['scale'] and error<prior)
                        else:outcome['old_center_useful']=False if l['kind']==0 else None'''
    assert source.count(old) == 1
    source = source.replace(old, '')
    return '\n'.join(line.rstrip() for line in source.splitlines())+'\n'


def prepare():
    assert not RUN.exists()
    base = read(OLD / 'protocol.json')
    fit, evaluate = base['fit20'], base['evaluate26']
    names = [Path(__file__).relative_to(ROOT).as_posix(), 'tracking/z13_region_readout.py',
             'tests/test_z13_region_readout.py', 'scripts/z13_region_readout_console.ps1',
             'scripts/z13_bounded_stage_20261022.py', 'scripts/z13_assignment_effect_20261024.py',
             'scripts/z13_query_memory_20261021.py', 'tracking/z13_readout.py',
             'tracking/z13_ridge_localizer.py', 'runs/z13_local_peak_20261021/geometry.json']
    for arm in ('D','C','R1'):
        for split in ('fit','evaluate'):
            names += [x.relative_to(ROOT).as_posix() for x in (MAPS/arm/split).glob('maps_*.npz')]
            names += [x.relative_to(ROOT).as_posix() for x in (MAPS/arm/split).glob('maps_*.json')]
            names.append((MAPS/arm/split/'labels.json').relative_to(ROOT).as_posix())
    authorization = read(W / 'authorization.json')
    began = now()
    p = dict(id=IDENT, created=began.isoformat(), device='cpu', budget_seconds=1200,
        save_at=min(began+timedelta(seconds=1200),datetime.fromisoformat(authorization['save_at'])).isoformat(),
        hard_stop=min(began+timedelta(seconds=1220),datetime.fromisoformat(authorization['hard_stop'])).isoformat(),
        fit20=fit, evaluate26=evaluate, videos=fit+evaluate, alphas=list(ALPHAS),
        action='Replace the internal readout output: use the existing target-response field to score actual detection regions, rather than overwrite the predicted box center with one peak. Test this zero-training prototype on exact one-step assignments before any closed-loop run.',
        method='All high-score detections are projected to the unchanged query retinal crop. A region is supported only if its center is in the existing hex field and its rectangle contains retinal columns. Its logit is the mean standardized target response inside the rectangle; NULL logit0; joint softmax over supported regions+NULL. New fused cost=(1-alpha)*old_cost+alpha*(1-det_conf*region_probability); unsupported costs unchanged. Exact alpha0 baseline. Same maps, support, calibration grid and association rules for C/D/R1. Geometry control uses only a fixed negative squared-distance field around the original prediction, replacing the response field in the identical decoder.',
        calibration='For each arm independently, choose alpha from[0,.25,.5,1] on20fit only by mean-video(beneficial-harmful)/known, original alpha0 wins ties. Freeze choices before opening evaluate26 outcome rows. All four grids reported. No training of core or readout weights, no best seed or quota search.',
        primary='On all26 reused training evaluation videos, D net action frequency versus unchanged baseline must have mean>=0.02 and lower paired-video95%CI>0. D must also beat geometry-only, R1 and C region decoders with all lower95%CI>0. All conditions required; other comparisons cannot rescue. Unknown outcomes reported separately and never assigned invented labels.',
        technical='All46 baseline outputs float32-exact. Every query bound to the saved tracker prediction/identity and its original label index. Before use, last-expert map top1 center must match canonical candidate within0.5px and score within0.001 on all fit/evaluate rows. Alpha0 whole assignments exactly equal baseline. Full tracker states/costs unchanged after every shadow evaluation. All source hashes unchanged. First2fit videos:2*mean_seconds*remaining44+120 must fit remaining1200s, else save partial and stop.',
        difference='The earlier detection3 gate replaced centers and evaluated center error, and learned-association/rankers selected existing response peaks. Neither tested this detection-region output or its effect on the complete assignment. The failed peak-center assignment reserve>=2% is preserved for that action; a cost-vector action has a different expressible output and is tested directly here. This is not a new detector or a claim that the surrounding matching algorithm improved.',
        counterargument='Cost calibration and proposal geometry may explain the entire effect without FlyVis. Region means may dilute the target, share response between overlapping boxes, or merely reject all detections. The no-FlyVis geometry/R1 controls are mandatory. Individual shadow actions do not capture simultaneous interactions, feedback or later matching stages, so a positive result is only a gate for separate closed-loop verification.',
        literature=dict(url='https://arxiv.org/html/2105.11595v1', sections='2.2,3.1,3.4',
            support='SiamMOT verifies region-based motion reading for association. This specific pooling/cost-vector interface is our prototype; SiamMOT does not provide its formula or gains.'),
        next='Only passing all criteria permits a separate frozen-policy closed-loop pilot with equal raw inputs, matched controls, own cost/deadline gate and HOTA. Otherwise preserve prototype and stop this implementation; no alpha retuning on26.',
        limits='Repeated26train, sparse original labels, baseline-state one-step diagnostic. Not HOTA, safe refusal, final-val performance or proven intrinsic core improvement. No GPU/new video/features/core training/dev/20val tuning/75/downloads.',
        executed_source_sha256=hashlib.sha256(replay_source().encode()).hexdigest(),
        sha256={**base['sha256'],**{n:sha(ROOT/n) for n in sorted(set(names))}})
    # Existing sources are retained as dependencies, not modified by this prototype.
    write(RUN/'protocol.json',p)
    write(RUN/'executed_replay_source.txt',replay_source())
    write(RUN/'status.json',dict(state='ready',stage='Считывание отклика по рамкам, без сдвига центра',done=0,total=46,updated=now().isoformat()))
    cp=ROOT/'experiments/project_history.json';cat=read(cp)
    assert not any(c['id']==IDENT for c in cat['experiments'])
    cat['experiments'].append(dict(id=IDENT,title='З13: считывание карты по областям детекций прямо в сопоставление',area='flyvis',state='ready',started=now().strftime('%Y-%m-%d'),tags=['fly-problem-P8','SiamMOT'],
        question='Помогает ли внутреннее чтение отклика по реальным рамкам сверх того же интерфейса без мухи, при неизменном центре трекера?',
        hypothesis='Геометрическое улучшение центра не гарантирует верного назначения; иной выход считывателя может использовать уже имеющийся отклик без разрушения рамки. Это проверяемая гипотеза, не установленное исправление.',
        expected=p['action']+' '+p['method'],acceptance=[dict(criterion=p['technical'],met=None),dict(criterion=p['primary'],met=None)],
        summary='Исполняемый внутренний декодер и одинаковые контроли зарегистрированы до результата.',obtained='',interpretation='',decision='',cannot_claim=p['counterargument']+' '+p['limits'],metrics=[],commits=[],files=names[:4],evidence=[dict(kind='file',ref=(RUN/'protocol.json').relative_to(ROOT).as_posix(),claim='Механизм, контроли и критерии до расчёта')]))
    write(cp,cat)
    journal('постановка нового выхода считывателя: отклик по рамкам',
        '*Действие.* '+p['action']+'\n\n*Метод.* '+p['method']+'\n\n*Калибровка и критерии до результата.* '+p['calibration']+' '+p['primary']+
        '\n\n*Отличие от испытанного.* '+p['difference']+'\n\n*Контраргумент.* '+p['counterargument']+
        '\n\nSiamMOT: '+p['literature']['url']+'; переносится мотив регионального движения, не формула или результат. CPU до1200с, независимый контроллер и видимая консоль; научные данные, прежние критерии и пороги применения не меняются.')


def worker():
    from scripts import z13_assignment_effect_20261024 as original
    from scripts.z13_query_memory_20261021 import load_maps
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_continue import paired
    from tracking.z13_readout import HexGeometry,to_eye,to_frame
    from tracking.z13_ridge_localizer import map_center
    from tracking.z13_region_readout import region_probabilities,cost_row
    p=read(RUN/'protocol.json');verify_sources(p)
    g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json')
    geom=HexGeometry(np.asarray(g['points'],np.float32),g['eye_size'])
    queries,centers=original.load_queries(read(OLD/'protocol.json'))
    maps={}; reproduction={}
    for split in ('fit','evaluate'):
        reference=ROOT/'runs/z13_candidate_reserve_20261021'/split
        for arm in ('D','C','R1'):
            write(RUN/'status.json',dict(state='running',stage='Воспроизведение исходных полных карт',variant=split+'/'+arm,done=len(reproduction),total=6,updated=now().isoformat()))
            data,_=load_maps(MAPS/arm/split)
            labels={x['label']:x for x in read(MAPS/arm/split/'labels.json')}
            canonical={}
            for path in sorted((reference/arm).glob('cache_*.npz')):
                with np.load(path) as z:
                    ep,ts,cs,sc=z['episode'],z['t'],z['centers'],z['scores']
                for i in range(len(ep)):
                    canonical[int(ep[i]),int(ts[i])]=(cs[i,0],float(sc[i,0]))
            dc=ds=0.
            oldlabels=read(reference/'labels.json')
            for label in oldlabels:
                idx=label['label'];s=data[idx][-1]
                c,score=map_center(s,geom);point=np.asarray(to_frame(c,label['crop'],g['eye_size']))
                other,oldscore=canonical[label['episode'],label['pos']+4]
                assert labels[idx]['video']==label['video'] and labels[idx]['kind']==label['kind']
                dc=max(dc,float(np.abs(point-other).max()));ds=max(ds,abs(score-oldscore))
            assert dc<.5 and ds<.001,(split,arm,dc,ds)
            reproduction[split+'_'+arm]=dict(rows=len(oldlabels),max_center_px=dc,max_score=ds)
            maps[split,arm]={idx:value[-1] for idx,value in data.items()}
    write(RUN/'reproduction.json',reproduction)
    support_counts=Counter()
    def new_cost_rows(q,old,detections):
        split='fit' if q['split']=='fit20' else 'evaluate'
        l=q['label'];crop=l['crop']
        boxes=np.asarray([d.xyxy for d in detections],float).reshape(-1,4)
        projected=[];supported=[]
        for box in boxes:
            a=to_eye(box[:2],crop,g['eye_size']);b=to_eye(box[2:],crop,g['eye_size'])
            projected.append([*a,*b])
            c=(np.asarray(a)+b)/2
            supported.append(bool(np.all(c>=0) and np.all(c<=g['eye_size']) and np.linalg.norm(geom.points[geom.nearest(c)]-c)<=.6*geom.spacing))
        projected=np.asarray(projected,float).reshape(-1,4)
        confidence=np.asarray([d.score for d in detections],float)
        prediction=np.asarray(to_eye(l['pred'],crop,g['eye_size']))
        geometry=-np.sum(((geom.points-prediction)/g['eye_size'])**2,axis=1)
        answer={}
        for arm in ARMS:
            field=geometry if arm=='geometry' else maps[split,arm][l['label']]
            probabilities,valid=region_probabilities(field,geom.points,projected,supported)
            answer[arm]=[cost_row(old,probabilities,confidence,valid,a) for a in ALPHAS]
            assert np.array_equal(answer[arm][0],old)
        support_counts['queries']+=1;support_counts['supported_regions']+=int(valid.sum())
        return answer
    source=replay_source();assert hashlib.sha256(source.encode()).hexdigest()==p['executed_source_sha256']
    namespace={**vars(original),'RUN':RUN,'new_cost_rows':new_cost_rows,'T0':time.monotonic()}
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    replay=namespace['replay_video'];times=[];choices=None
    def aggregate(videos,selection=None):
        result={a:{j:{} for j in range(len(ALPHAS))} for a in ARMS}
        counts={a:[Counter() for _ in ALPHAS] for a in ARMS}
        for video in videos:
            rows=read(RUN/'videos'/(video+'.json'))['rows'];known=[r for r in rows if r['kind']>=0]
            assert known
            for arm in ARMS:
                for j in range(len(ALPHAS)):
                    if selection is not None and j!=selection[arm]:continue
                    result[arm][j][video]=float(np.mean([int(r['arms'][arm][j]['beneficial'])-int(r['arms'][arm][j]['harmful']) for r in known]))
                    for row in rows:
                        c=counts[arm][j];c['rows']+=1
                        if row['kind']<0:c['unknown']+=1
                        else:
                            c['known']+=1;o=row['arms'][arm][j]
                            c['beneficial']+=o['beneficial'];c['harmful']+=o['harmful']
                            assert j!=0 or not(o['beneficial'] or o['harmful'])
        return result,counts
    for vi,video in enumerate(p['videos']):
        if vi==len(p['fit20']):
            fit,_=aggregate(p['fit20'])
            choices={a:max(range(len(ALPHAS)),key=lambda j:np.mean(list(fit[a][j].values()))) for a in ARMS}
            write(RUN/'choice.json',dict(indices=choices,alphas={a:ALPHAS[j] for a,j in choices.items()},fit_per_video=fit,frozen_before_evaluate26=True))
        path=RUN/'videos'/(video+'.json');assert not path.exists()
        began=time.monotonic();rows,checks=replay(video,queries[video],centers,p)
        write(path,dict(video=video,rows=rows,checks=checks,seconds=time.monotonic()-began));times.append(time.monotonic()-began)
        write(RUN/'status.json',dict(state='running',stage='Отклик по областям и полное сопоставление',variant=video,done=vi+1,total=46,updated=now().isoformat()))
        if vi==1:
            left=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
            estimate=2*np.mean(times)*44+120
            write(RUN/'price.json',dict(estimated=float(estimate),remaining=left,passed=bool(estimate<left)))
            assert estimate<left,'Conservative cost exceeds original budget'
    values,counts=aggregate(p['evaluate26'],choices)
    chosen={a:values[a][j] for a,j in choices.items()}
    contrasts={'D-baseline':paired(chosen['D'],{v:0. for v in p['evaluate26']})}
    for arm in ('C','R1','geometry'):contrasts['D-'+arm]=paired(chosen['D'],chosen[arm])
    passed=contrasts['D-baseline']['mean_per_video']>=.02 and all(v['ci95'][0]>0 for v in contrasts.values())
    verify_sources(p)
    write(RUN/'report.json',dict(complete=True,primary_pass=bool(passed),choices={a:ALPHAS[j] for a,j in choices.items()},
        paired=contrasts,per_video=chosen,counts={a:dict(counts[a][j]) for a,j in choices.items()},
        support_counts=dict(support_counts),sources_unchanged=True,exact_baseline_videos=46,
        video_sha256={v:sha(RUN/'videos'/(v+'.json')) for v in p['videos']},not_hota=True,limits=p['limits']))
    write(RUN/'status.json',dict(state='completed',stage='Региональный выход считывателя проверен',done=46,total=46,updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(RUN/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    else:raise ValueError('Choose --prepare or --worker')
