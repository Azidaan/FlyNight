"""Bounded train26 saved-feature check; no FlyVis forward pass or core training."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha

RUN = ROOT / 'runs/z13_pairwise_utility_20261021'
OLD = ROOT / 'runs/z13_map_utility_20261018'
MSK = timezone(timedelta(hours=3))
IDENT = 'z13_pairwise_utility_v1'


def journal(title, text):
    path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan = path.read_text(encoding='utf-8')
    anchor = '## Журнал изменений\n'
    assert anchor in plan
    now = datetime.now(MSK)
    path.write_text(plan.replace(anchor, anchor + f'\n**{now:%d.%m.%Y, %H:%M} MSK** — {title} `{IDENT}`\n\n' + text + '\n', 1), encoding='utf-8')


def cutoff(now):
    for delta in range(8):
        day = now.date() + timedelta(days=delta)
        stamp = datetime.combine(day, datetime.min.time(), MSK).replace(hour=7, minute=40)
        if day.weekday() < 5 and stamp > now:
            return stamp
    raise AssertionError('Missing weekday cutoff')


def verify(p):
    for rel, digest in p['sha256'].items():
        path = (ROOT / rel).resolve()
        path.relative_to(ROOT.resolve())
        assert sha(path) == digest, rel


def state(stage, **values):
    write(RUN / 'status.json', dict(state='running', stage=stage,
          updated=datetime.now(MSK).isoformat(), **values))


def prepare():
    assert not RUN.exists(), 'Inspect existing experiment; do not duplicate'
    previous = read(OLD / 'protocol.json')
    videos = previous['videos']
    names = ['tracking/z13_pairwise_utility.py', 'tracking/z13_utility.py',
             'tracking/z13_checkpoint.py', 'scripts/z13_pairwise_utility_20261021.py',
             'scripts/z13_pairwise_utility_progress.ps1', 'tests/test_z13_pairwise_utility.py',
             'scripts/z13_parallel_next.py', 'scripts/z13_continue.py', 'scripts/z13_score_swap.py',
             'runs/z13_map_utility_20261018/protocol.json', 'runs/z13_map_utility_20261018/report.json']
    for video in videos:
        names.append(f'runs/z13_data/{video}.npz')
    for arm in ('D', 'C', 'R1'):
        names += [f'runs/z13_target_center_20261018/{arm}/center_rows.json',
                  f'runs/z13_map_utility_20261018/{arm}/predicted_scores.json',
                  f'runs/z13_map_utility_20261018/{arm}/feature_index.json',
                  f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json']
        names += [f'runs/z13_map_utility_20261018/{arm}/{v}.json' for v in videos]
    now = datetime.now(MSK)
    deadline = cutoff(now)
    p = dict(id=IDENT, created_at=now.isoformat(), timezone='Europe/Moscow (UTC+03:00)',
        authorization='Author requested continued work; a bounded cheap CPU pilot under AGENTS section7. No plan revision or expensive core training.',
        action='Add one fixed pairwise ranking term to a linear utility objective for unchanged D/C/R1 centers and saved14 features.',
        question='Does training known useful examples to outrank known nonuseful ones improve the primary1% selection?',
        objective='mean pointwise logistic loss + strength*mean pairwise logistic loss + ||w||^2/(2*n_known). strength0 vs1; same scipy L-BFGS-B max500 gtol1e-8 ftol1e-12; intercept unpenalized. No hyperparameter search.',
        pairs='Uniform without replacement from all known positive-negative training pairs; max4096, NumPy seed0 per fold. Global ranking matches global selection, not per-video deployment.',
        estimator='StandardScaler fitted only on known outcomes of25 other train videos. Leave one entire video out. Fit two objectives on identical14 map+geometry features, plus same objectives on6 geometry features.',
        primary='D_pair_map-D_point_map, D_pair_map-D_old_map, D_pair_map-D_pair_geometry, D_pair_map-R1_pair_map: all four lower95% paired video bootstrap intervals strictly above0 at fixed global1% coverage. Primary uses all26 videos.2%/5% secondary cannot rescue failure.',
        reproduction='Before any new objective fitting, reproduce original sklearn C1 max500 seed0 scores for all3 arms and both14/6 features, max absolute difference<=1e-10, identical selected keys at1/2/5%. Preserve exact row keys and outcomes.',
        counterargument='The14 observables may not distinguish identity or center accuracy; a pairwise loss does not guarantee precision at the extreme top, and global held-out ranking is not a causal online threshold.',
        alternatives='A reliable-observation memory readout (STMTrack) and candidate-identity memory (KeepTrack) change internal information but need new forward passes. Choose this single cached-data gate first for lower cost, not because the alternatives are disproved.',
        literature=[dict(url='https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf', section='3, equations1-3', supports='Pairwise logistic ranking loss; article uses web search, not FlyVis or tracking. Mixed point/pair objective is our adaptation.')],
        limitations='Exploratory reused train26, one fixed seed; unknown outcomes excluded from fitting but retained in selection denominator. This is an application of modernized FlyVis, not new core modernization or evidence of HOTA. No dev, heldout75, GPU, core training, video decoding or downloads.',
        decision_value='Only if all primary criteria pass prepare a separately specified causal application policy. Otherwise stop this fixed loss implementation; do not tune on dev or close FlyVis.',
        videos=videos, arms=['D','C','R1'], budget_seconds=600, shutdown_seconds=15,
        schedule_hard_stop=deadline.isoformat(), cuda=False, sha256={name:sha(ROOT/name) for name in names})
    RUN.mkdir()
    write(RUN / 'protocol.json', p)
    write(RUN / 'status.json', dict(state='ready', stage='Критерии записаны до прогона', updated=now.isoformat(), done=0,total=26))
    path = ROOT / 'experiments/project_history.json'
    catalog = read(path)
    assert not any(x['id'] == IDENT for x in catalog['experiments'])
    catalog['experiments'].append(dict(id=IDENT, title='З13: парное обучение оценки полезности фиксированных центров', area='flyvis', state='ready', started=now.date().isoformat(),
        question=p['question'], hypothesis='Парная ошибка может улучшить порядок полезных поправок на неизменных признаках. Причина прежнего провала не установлена.',
        expected='Одна фиксированная добавка к функции обучения; совпадающие данные, признаки, оптимизатор и регуляризация в парном контроле. Сначала точное воспроизведение старой оценки, затем полный train26; CPU до600с. Главная доля отбора1%; дополнительные2%/5% не заменяют её.',
        acceptance=[dict(criterion='Воспроизведены прежние6/14-признаковые оценки всех3 arms до1e-10 и все выбранные ключи; исходники неизменны',met=None),
                    dict(criterion='При1% D_pair_map лучше совпадающего D_point_map и прежнего D_old_map: обе нижние95% повидеоДИ>0',met=None),
                    dict(criterion='При1% D_pair_map лучше D_pair_geometry и R1_pair_map: обе нижние95% повидеоДИ>0',met=None)],
        tags=['Z13','fly-problem-P8'], runs=['runs/z13_pairwise_utility_20261021'], files=['tracking/z13_pairwise_utility.py','scripts/z13_pairwise_utility_20261021.py'],
        evidence=[dict(kind='json',ref='runs/z13_pairwise_utility_20261021/protocol.json',claim='Критерии и источники до расчёта')]))
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('постановка парного обучения оценки', '*Постановка.* '+p['action']+' Принцип парной логистической ошибки — RankNet, раздел3, '+p['literature'][0]['url']+'. Применимость к нашей задаче не установлена; смешанная цель — наша проверка. '+p['primary']+'\n\n*Решение.* Один дешёвый полный train26 по сохранённым данным. '+p['limitations']+' Протокол: runs/z13_pairwise_utility_20261021/protocol.json. Бюджет600с +15с завершения; предел автора '+deadline.isoformat()+'.')


def inputs(p, arm):
    import numpy as np
    from tracking.z13_utility import observable_features
    rows = read(ROOT / f'runs/z13_target_center_20261018/{arm}/center_rows.json')
    crops, offset = {}, 0
    maps = {}
    for video in p['videos']:
        with np.load(ROOT / f'runs/z13_data/{video}.npz') as data:
            for i, crop in enumerate(data['lab_crop']):
                crops[video, str(offset+i)] = crop.copy()
            offset += len(data['lab_kind'])
        for item in read(OLD / arm / (video+'.json'))['features']:
            maps[video, str(item['label'])] = item['values']
    pairs = [(r['video'], str(r['label'])) for r in rows]
    assert len(set(pairs)) == len(rows) and set(pairs) == set(maps)
    geom = np.array([observable_features(r,crops[pair])[0] for r,pair in zip(rows,pairs)])
    full = np.c_[geom, np.array([maps[pair] for pair in pairs])]
    assert full.shape[1] == 14
    assert all('useful' in r for r in rows if r['kind'] >= 0)
    y = np.array([int(bool(r['useful'])) if r['kind'] >= 0 else -1 for r in rows])
    return rows, geom, full, y, [r['video'] for r in rows]


def worker():
    import numpy as np
    from tracking.z13_utility import leave_video_out as previous_fit
    from tracking.z13_pairwise_utility import leave_video_out as new_fit
    from scripts.z13_score_swap import selected_keys, key
    from scripts.z13_continue import paired, per_video_useful, coverage_table
    p = read(RUN/'protocol.json')
    while not (RUN/'controller.json').exists():
        time.sleep(.02)
    control = read(RUN/'controller.json')
    def check():
        if time.time() >= control['soft_timestamp'] or (RUN/'stop.request').exists():
            raise TimeoutError('Authorized budget ended')
    assert not (RUN/'report.json').exists()
    state('Проверка неизменности данных', done=0,total=26)
    verify(p);check()
    cached, reproduction = {}, {}
    # Every old arm/control must reproduce BEFORE any new objective is fitted.
    for arm in p['arms']:
        rows, geom, full, y, videos = inputs(p,arm)
        cached[arm] = (rows,geom,full,y,videos)
        old = {(x['video'],str(x['label'])):x['score'] for x in read(OLD/arm/'predicted_scores.json')}
        old_geom = read(ROOT/f'runs/z13_parallel_next_20261018/utility/{arm}_rows.json')['geometry']
        oldg = {key(r):r['p_present'] for r in old_geom}
        expected = {'map':np.array([old[r['video'],str(r['label'])] for r in rows]),
                    'geometry':np.array([oldg[key(r)] for r in rows])}
        reproduction[arm] = {}
        for name, x in [('map',full),('geometry',geom)]:
            scores, _ = previous_fit(x,y,videos,lambda done,total,video: (check(),state('Воспроизведение прежней оценки',variant=arm+'/'+name,done=done,total=total,video=video)))
            error = float(np.max(np.abs(scores-expected[name])))
            assert error <= 1e-10, (arm,name,error)
            a = [dict(r,p_present=float(s)) for r,s in zip(rows,scores)]
            b = [dict(r,p_present=float(s)) for r,s in zip(rows,expected[name])]
            assert all(selected_keys(a,q) == selected_keys(b,q) for q in (.01,.02,.05))
            reproduction[arm][name] = dict(max_score_difference=error, selection_identical=True,rows=len(rows))
        write(RUN/'reproduction.json',reproduction)
    allpv, arms = {}, {}
    for arm in p['arms']:
        rows, geom, full, y, videos = cached[arm]
        variants = {}
        folder = RUN/arm;folder.mkdir(exist_ok=True)
        old = {(x['video'],str(x['label'])):x['score'] for x in read(OLD/arm/'predicted_scores.json')}
        variants['old_map'] = [dict(r,p_present=float(old[r['video'],str(r['label'])])) for r in rows]
        predictions, folds = {}, {}
        for name, x, strength in [('point_map',full,0.),('pair_map',full,1.),('point_geometry',geom,0.),('pair_geometry',geom,1.)]:
            check()
            scores, fit = new_fit(x,y,videos,strength,
                lambda done,total,video: state('Обучение оценки; ядро и центры неизменны',variant=arm+'/'+name,done=done,total=total,video=video),check)
            predictions[name] = [dict(video=r['video'],label=r['label'],kind=r['kind'],score=float(s)) for r,s in zip(rows,scores)]
            folds[name] = fit
            variants[name] = [dict(r,p_present=float(s)) for r,s in zip(rows,scores)]
            write(folder/'scores.json',predictions);write(folder/'folds.json',folds)
        coverage, levels = {}, {}
        for name, scored in variants.items():
            coverage[name] = coverage_table(scored)
            for q in (.01,.02,.05):
                chosen = selected_keys(scored,q)
                pv = per_video_useful([dict(r,useful=bool(r.get('useful')) and key(r) in chosen) for r in scored if r['kind']==1])
                assert len(pv) == len(p['videos'])
                allpv[arm,name,q] = pv
                levels[f'{name}/{q:.2f}'] = float(np.mean(list(pv.values())))
        comparisons = {f'{q:.2f}':{f'pair_map-{base}':paired(allpv[arm,'pair_map',q],allpv[arm,base,q])
                       for base in ('point_map','old_map','pair_geometry')} for q in (.01,.02,.05)}
        arms[arm] = dict(coverage=coverage,levels=levels,paired=comparisons)
        write(folder/'done.json',arms[arm])
    cross = {f'{q:.2f}':{f'D_pair_map-{arm}_pair_map':paired(allpv['D','pair_map',q],allpv[arm,'pair_map',q]) for arm in ('R1','C')}
             for q in (.01,.02,.05)}
    primary = dict(arms['D']['paired']['0.01'])
    primary['D_pair_map-R1_pair_map'] = cross['0.01']['D_pair_map-R1_pair_map']
    passed = all(x['ci95'][0] > 0 for x in primary.values())
    check();verify(p)
    assert 'torch' not in sys.modules, 'This saved-feature check must not load FlyVis or torch'
    write(RUN/'report.json',dict(arms=arms,cross=cross,primary=primary,primary_pass=passed,
          decision='prepare_separate_causal_policy' if passed else 'fixed_pairwise_loss_gate_not_passed',
          reproduction_pass=True,source_hashes_unchanged=True,torch_loaded=False,gpu=False,core_training=False,
          dev_evaluated=False,hota_measured=False,heldout75=False,limitations=p['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='Полный train26 сохранён',done=26,total=26,updated=datetime.now(MSK).isoformat()))


def controller():
    import psutil
    assert not (RUN/'controller.json').exists(), 'Refuse duplicate controller'
    p = read(RUN/'protocol.json')
    verify(p)
    hard = min(time.time()+p['budget_seconds']+p['shutdown_seconds'],datetime.fromisoformat(p['schedule_hard_stop']).timestamp()-1)
    soft = hard-p['shutdown_seconds']
    assert time.time() < soft
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    with (RUN/'stdout.log').open('w',encoding='utf-8') as out,(RUN/'stderr.log').open('w',encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__).resolve()),'worker'],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born = psutil.Process(child.pid).create_time()
        write(RUN/'controller.json',dict(pid=child.pid,create_time=born,controller_pid=os.getpid(),started_at=datetime.now(MSK).isoformat(),soft_timestamp=soft,
             save_stop_at=datetime.fromtimestamp(soft,MSK).isoformat(),hard_stop_at=datetime.fromtimestamp(hard,MSK).isoformat(),schedule_hard_stop=p['schedule_hard_stop']))
        timeout = False
        while child.poll() is None:
            if time.time() >= soft and not (RUN/'stop.request').exists():
                write(RUN/'stop.request',dict(reason='authorized deadline',at=datetime.now(MSK).isoformat()))
            if time.time() >= hard:
                assert abs(psutil.Process(child.pid).create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=10);timeout=True;break
            time.sleep(.2)
    write(RUN/'controller_result.json',dict(worker_ended=True,exit_code=child.returncode,hard_timeout=timeout,ended_at=datetime.now(MSK).isoformat()))
    if child.returncode:
        write(RUN/'status.json',dict(state='failed',stage='Результат частичный; см. stderr.log',updated=datetime.now(MSK).isoformat()))
    return child.returncode


def finish():
    assert not (RUN/'recorded.json').exists()
    p, report, end = read(RUN/'protocol.json'),read(RUN/'report.json'),read(RUN/'controller_result.json')
    assert end['worker_ended'] and end['exit_code']==0 and report['source_hashes_unchanged']
    assert all((RUN/arm/'done.json').exists() for arm in p['arms'])
    verify(p)
    path = ROOT/'experiments/project_history.json';cat=read(path)
    card = next(x for x in cat['experiments'] if x['id']==IDENT)
    source='runs/z13_pairwise_utility_20261021/report.json'
    def number(v):return f'{v*100:+.4f}'.replace('.',',')
    details='; '.join(f"{name}: {number(v['mean_per_video'])} п.п. [{number(v['ci95'][0])}; {number(v['ci95'][1])}]" for name,v in report['primary'].items())
    counts=report['arms']['D']['coverage']['pair_map']['0.01']
    obtained=source+' (primary): '+details+'. '+source+' (arms.D.coverage.pair_map.0.01): '+json.dumps(counts,ensure_ascii=False)+'. Все26видео train; воспроизведение прежних оценок и выбранных ключей подтверждено в runs/z13_pairwise_utility_20261021/reproduction.json; source_hashes_unchanged=true.'
    passed=report['primary_pass']
    card.update(state='completed_positive' if passed else 'inconclusive',done=datetime.now(MSK).date().isoformat(),obtained=obtained,
        interpretation='Заранее заданный критерий парной цели '+('выполнен; перенос ещё не проверен.' if passed else 'не выполнен. Это итог одной фиксированной реализации, не диагноз представления или отказ от FlyVis.'),
        decision=report['decision'],cannot_claim=p['limitations']+' Дополнительные доли2%/5% и отдельный положительный контраст не заменяют общий критерий. Известный неполезный ответ не тождествен измеренному вреду HOTA.')
    primary=report['primary']
    checks=[True,all(primary[k]['ci95'][0]>0 for k in ('pair_map-point_map','pair_map-old_map')),
            all(primary[k]['ci95'][0]>0 for k in ('pair_map-pair_geometry','D_pair_map-R1_pair_map'))]
    for criterion,met in zip(card['acceptance'],checks):criterion.update(met=met,evidence=source)
    card['evidence'].append(dict(kind='json',ref=source,claim='Полный результат и все основные сравнения'))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    journal('итог парной цели оценки', '*Получено.* '+obtained+'\n\n*Вывод.* '+card['interpretation']+'\n\n*Решение.* '+card['decision']+'. '+card['cannot_claim'])
    write(RUN/'recorded.json',dict(recorded_at=datetime.now(MSK).isoformat(),primary_pass=passed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','controller','worker','finish']);args=parser.parse_args()
    if args.mode=='worker':
        try:worker()
        except BaseException as exc:
            write(RUN/'status.json',dict(state='paused' if isinstance(exc,TimeoutError) else 'failed',stage=str(exc),updated=datetime.now(MSK).isoformat()))
            raise
    elif args.mode=='controller':sys.exit(controller())
    else:{'prepare':prepare,'finish':finish}[args.mode]()
