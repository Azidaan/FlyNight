"""Prospective transfer of the fixed C directional readout; preparation is not a full-run authorization."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha

RUN = ROOT / 'runs/z13_context_transfer_20261020'
SOURCE = ROOT / 'runs/z13_directional_context_20261018'
MEAN = ROOT / 'runs/z13_map_transfer_20261018'
DEV = ROOT / 'runs/z13_target_center_dev_20261018'
TRAIN = ROOT / 'runs/z13_map_utility_20261018'
ARMS = ('C', 'R1')
OWN = ['scripts/z13_context_transfer_20261020.py', 'scripts/z13_context_transfer_progress.ps1']
CONTEXT = {}
DEADLINE = None
START = time.monotonic()


def status(**values):
    if (RUN / 'stop.request').exists() or (DEADLINE and time.time() >= DEADLINE):
        raise TimeoutError('Finite authorized window ended; completed videos retained')
    write(RUN / 'status.json', dict(CONTEXT, state='running', updated=datetime.now(timezone.utc).isoformat(),
          elapsed_seconds=time.monotonic()-START, **values))


def verify(p):
    for path, digest in p['sha256'].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError('Frozen source changed: ' + path)


def prepare():
    if RUN.exists():
        raise RuntimeError('Existing run; do not duplicate')
    original = read(SOURCE / 'protocol.json')
    from scripts.z13_directional_context import verify as verify_original
    verify_original(original)
    ending = read(SOURCE / 'controller_result.json')
    assert ending['worker_ended'] and ending['exit_code'] == 0
    previous = read(MEAN / 'protocol.json')
    paths = set(original['sha256']) | set(previous['sha256']) | set(OWN)
    paths.update(['runs/z13_directional_context_20261018/report.json',
                  'runs/z13_map_transfer_20261018/report.json',
                  'tracking/z13_utility_transfer.py', 'tracking/z13_directional_context.py'])
    for arm in ARMS:
        paths.add(f'runs/z13_directional_context_20261018/{arm}/directional_predictions.json')
        for video in original['videos']:
            paths.add(f'runs/z13_directional_context_20261018/{arm}/videos/{video}.json')
    paths.add('runs/z13_map_transfer_20261018/C/predictions.json')
    now = datetime.now(timezone(timedelta(hours=3)))
    p = dict(id='z13_context_transfer_v1', created_at=now.isoformat(), train_videos=original['videos'],
        videos=previous['videos'], arms=list(ARMS), sha256={x: sha(ROOT/x) for x in sorted(paths)},
        authorization='20.10 author requested continuation: prepare the next existing readout check and a bounded CPU technical probe. Full calculation requires a new finite budget.',
        action='Transfer the unchanged238-channel directional C readout. Same normalized68 inputs plus170 fixed contrasts; original per-arm lambda; no model or feature tuning on dev.',
        selection_reason='C is selected after the auxiliary train26 result:28 useful among48 known selected,36 unknown,84 selected. The original D experiment remains inconclusive and is not rescued.',
        localization='C directional minus C mean: paired video95%CI lower>0 and mean>=0.02; C directional minus R1 directional lower>0.',
        selection='At1% of all rows: C directional map utility minus C mean map utility, C directional geometry, and R1 directional map utility each paired video95%CI lower>0. Both localization and selection required.',
        fitting='Train26 only: StandardScaler plus LogisticRegression C1 lbfgs max_iter500 seed0; geometry6 and map14; exclude unknown labels. All heads saved before dev inference. C mean comparator uses the existing fixed transfer predictions.',
        reproduction='Unchanged mean centers<0.5px, scores<0.001, same labels/kind/useful on every video. Technical C directional center<0.5px, score<0.001, map features<1e-10 against the saved same training video. Normalization exactly equals saved dev normalization.',
        counterargument='Post-hoc C selection and repeatedly used dev35 can yield an unstable apparent improvement. Unknown outcomes are not safe negatives. This check cannot establish blind generalization or an online deployment threshold.',
        decision='If either complete gate fails, preserve this implementation result and do not retune dev or apply it to the tracker. No HOTA or75 heldout in this protocol.',
        technical_budget_seconds=300, full_budget_seconds=None, full_authorization_required=True,
        cpu=True, gpu=False, core_training=False, hota=False, heldout75=False, chat_automation=False,
        literature=['https://arxiv.org/abs/1904.07220'],
        limitations='Exploratory follow-up selected after train26; dev repeatedly used. C is stock core with modernized readout, not proof of D core-training benefit. No new detector or tracker.')
    assert not set(p['train_videos']) & set(p['videos'])
    RUN.mkdir()
    write(RUN / 'protocol.json', p)
    write(RUN / 'status.json', dict(state='ready', stage='prepared; technical probe only', done=0, total=2))
    cat = read(ROOT / 'experiments/project_history.json')
    assert not any(c['id'] == p['id'] for c in cat['experiments'])
    cat['experiments'].append(dict(id=p['id'], title='З13: отдельный перенос пространственного считывания C и контроля R1',
        area='flyvis', state='ready', started=now.date().isoformat(),
        question='Переносится ли вспомогательный результат пространственного считывания C на другие видео?',
        hypothesis='Фиксированное пространственное считывание C может улучшить центры и выбор поправок; перенос не установлен.',
        expected=p['localization']+' '+p['selection'],
        acceptance=[dict(criterion=p[k], met=None) for k in ('reproduction', 'localization', 'selection')],
        tags=['Z13', 'fly-problem-P8'], runs=['runs/z13_context_transfer_20261020'], files=OWN,
        evidence=[dict(kind='json', ref='runs/z13_context_transfer_20261020/protocol.json', claim='Условия до переноса; полная проверка пока не разрешена бюджетом')]))
    (ROOT / 'experiments/project_history.json').write_text(json.dumps(cat, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    from scripts.z13_record_next import journal
    journal('подготовка z13_context_transfer_v1',
        '*Основание.* runs/z13_directional_context_20261018/report.json: основной D-критерий не выполнен; этот итог сохраняется. Вспомогательная C: summary.C_directional.coverage.map_utility.0.01 —28полезных,20известных неполезных,36неизвестных из84выбранных. Это выбор кандидата после просмотра train26. '
        '*Постановка.* Одна прежняя пространственная замена считывания, фиксированные C/R1 и λ, без дообучения ядра. Сравнить на35dev с прежней C и тем же пространственным R1, весь отбор1% и нормировка прежние; оба критерия protocol.json обязательны. Повторный dev не слепой. Сейчас только подготовка и техническая CPU-проба до300с; полный бюджет после фактической цены. HOTA, GPU,75 и замена трекера не запускаются.')


def capture(torch, arm, bb, geom, bank, video, refs, folder, technical=False):
    from tracking import z13_target_center as target
    from tracking.z13_directional_context import directional_neighbors, expand_features
    from tracking.z13_response_utility import response_features
    from tracking.z13_readout import to_frame
    from scripts.z13_target_model import center_and_presence
    from scripts.z13_train import success
    from scripts import z13_map_transfer as service
    choices = read(ROOT / 'runs/z13_target_center_20261018/report.json')['choices']
    with np.load(TRAIN / arm / 'stats.npz') as z:
        stats = z['mu'], z['sd']
    norm = read(DEV / arm / 'normalization.json')
    assert np.array_equal(stats[0], norm['mean']) and np.array_equal(stats[1], norm['sd'])
    slots = directional_neighbors(geom.points, geom.neighbors)
    original = target.ridge_scores
    extra = []
    def tapping(xe, y, xs, lambdas):
        base = original(xe, y, xs, lambdas)
        score = original(expand_features(xe, slots), y, expand_features(xs, slots), lambdas)[0]
        for value in score:
            center, peak = center_and_presence(value, geom)
            extra.append(dict(center_eye=center, peak=peak, map_features=response_features(value, geom).tolist()))
        return base
    old_status = service.status
    target.ridge_scores = tapping
    service.status = status
    data_dir = ROOT / 'runs/z13_data' if technical else DEV / 'data'
    try:
        baseline = service.capture_video(torch, arm, bb, geom, bank.eye_size, video,
            folder / 'mean_capture.json', stats, choices, data_dir, refs, canonical=not technical)
    finally:
        service.status = old_status
        target.ridge_scores = original
    by_key = {str(r['label']): r for r in refs}
    assert len(baseline['features']) == len(extra) == len(refs)
    rows = []
    with np.load(data_dir / (video+'.npz')) as data:
        for item, values in zip(baseline['features'], extra):
            label = str(item['label']); row = dict(by_key[label])
            local = int(label if technical else label.rsplit(':', 1)[1])
            center = to_frame(values['center_eye'], data['lab_crop'][local], bank.eye_size)
            row.update(center=list(center), p_present=values['peak'], map_features=values['map_features'], useful=False)
            if row['kind'] == 1:
                row['err'] = float(np.linalg.norm(np.asarray(center)-data['lab_center'][local]))
                row['useful'] = bool(success(row))
            rows.append(row)
    result = dict(rows=rows, timing=baseline['timing'], reproduction={k:baseline[k] for k in
        ('max_center_difference', 'max_peak_difference', 'known_outcomes_match')})
    write(folder / 'directional_rows.json', result)
    return result


def technical(p, torch):
    from scripts import z13_target_model as old
    status(stage='build unchanged C core', variant='C', done=0, total=0)
    ck = torch.load(ROOT / 'runs/z13_spatial_v2_20261013/D/full/model.pt', map_location='cpu', weights_only=False)
    bank, geom, bb = old.build_arm('C', torch, ck)
    assert str(bb.dev) == 'cpu'
    video = p['train_videos'][0]
    refs = [r for r in read(ROOT / 'runs/z13_target_center_20261018/C/center_rows.json') if r['video'] == video]
    result = capture(torch, 'C', bb, geom, bank, video, refs, RUN / 'technical', True)
    saved = read(SOURCE / 'C' / 'videos' / (video+'.json'))
    before = {str(r['label']):r for r in saved['variants']['directional']}
    dc = dp = df = 0.
    for r in result['rows']:
        ref = before[str(r['label'])]
        assert r['kind'] == ref['kind'] and bool(r['useful']) == bool(ref['useful'])
        dc = max(dc, float(np.max(np.abs(np.asarray(r['center'])-ref['center']))))
        dp = max(dp, abs(r['p_present']-ref['p_present']))
        df = max(df, float(np.max(np.abs(np.asarray(r['map_features'])-ref['map_features']))))
    assert dc < .5 and dp < .001 and df < 1e-10
    assert not torch.cuda.is_initialized()
    verify(p)
    write(RUN / 'technical.json', dict(passed=True, video=video, timing=result['timing'],
        directional_center_max_error=dc, directional_peak_max_error=dp, directional_features_max_error=df,
        mean_reproduction=result['reproduction'], cuda_initialized=False, scientific_efficacy_measured=False))
    write(RUN / 'status.json', dict(state='ready', stage='technical parity passed; full budget pending', done=0, total=2,
        updated=datetime.now(timezone.utc).isoformat()))


def fit_all(p):
    import joblib
    from scripts.z13_map_transfer import metadata
    from tracking.z13_utility import observable_features
    from tracking.z13_utility_transfer import fit_training_only
    crops = metadata(p['train_videos'], ROOT / 'runs/z13_data')
    for arm in ARMS:
        folder = RUN / arm
        if (folder / 'fit.json').exists():
            for name, digest in read(folder / 'fit.json')['models'].items():
                assert sha(folder / (name+'.joblib')) == digest
            continue
        rows = read(SOURCE / arm / 'directional_predictions.json')['original']
        x = {'geometry':[], 'map_utility':[]}
        for r in rows:
            geometry, _ = observable_features(r, crops[r['video'],str(r['label'])])
            x['geometry'].append(geometry); x['map_utility'].append(np.r_[geometry,r['map_features']])
        y = [int(bool(r['useful'])) if r['kind'] >= 0 else -1 for r in rows]
        models = {}
        for name in x:
            model, info = fit_training_only(x[name], y, [r['video'] for r in rows], p['videos'])
            folder.mkdir(exist_ok=True)
            joblib.dump(model, folder / (name+'.joblib'))
            models[name] = sha(folder / (name+'.joblib'))
        write(folder / 'fit.json', dict(models=models, info=info, evaluation_outcomes_used=False))
    write(RUN / 'all_heads_fitted.json', dict(fitted_before_dev_inference=True,
        models={a:read(RUN / a / 'fit.json')['models'] for a in ARMS}))


def full(p, torch):
    import joblib
    from scripts import z13_target_model as old
    from scripts.z13_map_transfer import metadata
    from tracking.z13_utility import observable_features
    fit_all(p)
    crops = metadata(p['videos'], DEV / 'data', canonical=True)
    ck = torch.load(ROOT / 'runs/z13_spatial_v2_20261013/D/full/model.pt', map_location='cpu', weights_only=False)
    for ai, arm in enumerate(ARMS):
        CONTEXT.update(variant=arm, arms_done=ai, arms_total=2, videos_total=len(p['videos']))
        if (RUN / arm / 'done.json').exists():
            continue
        status(stage='build unchanged CPU arm', done=0, total=0)
        bank, geom, bb = old.build_arm(arm, torch, ck)
        assert bb is None or str(bb.dev) == 'cpu'
        rows = []
        for vi, video in enumerate(p['videos']):
            CONTEXT.update(videos_done=vi, video=video)
            status(stage='fixed directional inference', done=0, total=0)
            folder = RUN / arm / 'videos' / video
            path = folder / 'directional_rows.json'
            result = read(path) if path.exists() else capture(torch, arm, bb, geom, bank, video,
                read(DEV / arm / video / 'center_rows.json'), folder)
            rows.extend(result['rows'])
        del bb, bank
        x = {'geometry':[], 'map_utility':[]}
        for r in rows:
            geometry, _ = observable_features(r, crops[r['video'],str(r['label'])])
            x['geometry'].append(geometry); x['map_utility'].append(np.r_[geometry,r['map_features']])
        variants = {'original':rows}
        for name in x:
            assert sha(RUN / arm / (name+'.joblib')) == read(RUN / arm / 'fit.json')['models'][name]
            model = joblib.load(RUN / arm / (name+'.joblib'))
            variants[name] = [dict(r,p_present=float(s)) for r,s in zip(rows,model.predict_proba(x[name])[:,1])]
        write(RUN / arm / 'predictions.json', variants)
        write(RUN / arm / 'done.json', dict(videos=len(p['videos']), rows=len(rows)))
    summarize(p)
    verify(p)
    assert not torch.cuda.is_initialized()
    write(RUN / 'status.json', dict(state='completed', stage='full fixed transfer saved', done=2, total=2,
        updated=datetime.now(timezone.utc).isoformat()))


def summarize(p):
    from scripts.z13_continue import per_video_useful, paired, coverage_table
    from scripts.z13_score_swap import selected_keys, key
    c = read(RUN / 'C/predictions.json'); r = read(RUN / 'R1/predictions.json')
    mean_map = read(MEAN / 'C/predictions.json')['map_utility']
    mean_rows = [x for v in p['videos'] for x in read(DEV / 'C' / v / 'center_rows.json')]
    loc = {'C_directional-C_mean':paired(per_video_useful(c['original']),per_video_useful(mean_rows)),
           'C_directional-R1_directional':paired(per_video_useful(c['original']),per_video_useful(r['original']))}
    def selected(rows):
        ids = selected_keys(rows,.01)
        return per_video_useful([dict(x,useful=bool(x['useful']) and key(x) in ids) for x in rows])
    selection = {name:paired(selected(c['map_utility']),selected(rows)) for name,rows in
        [('C_context_map-C_mean_map',mean_map),('C_context_map-C_context_geometry',c['geometry']),
         ('C_context_map-R1_context_map',r['map_utility'])]}
    lp = all(v['ci95'][0] > 0 for v in loc.values()) and loc['C_directional-C_mean']['mean_per_video'] >= .02
    sp = all(v['ci95'][0] > 0 for v in selection.values())
    write(RUN / 'report.json', dict(localization=loc,selection_at1percent=selection,
        coverage={a:{h:coverage_table(v) for h,v in variants.items()} for a,variants in [('C',c),('R1',r)]},
        localization_pass=lp, selection_pass=sp, proceed=lp and sp, hota_measured=False,
        decision='prepare_application_only_after_separate_policy' if lp and sp else 'context_transfer_not_confirmed; no_dev_retuning_or_tracker_application',
        limitations=p['limitations'], unknowns_not_safety_guarantee=True))


def worker(technical_only):
    global DEADLINE
    p = read(RUN / 'protocol.json'); verify(p)
    DEADLINE = read(RUN / ('technical_controller.json' if technical_only else 'controller.json'))['soft_timestamp']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
    import torch
    torch.cuda.is_available = lambda:False
    torch.set_default_device('cpu'); torch.set_num_threads(2)
    if technical_only:
        technical(p,torch)
    else:
        if not (RUN / 'full_authorization.json').exists():
            raise RuntimeError('Full budget authorization missing')
        full(p,torch)


def controller(technical_only):
    import psutil
    p = read(RUN / 'protocol.json'); verify(p)
    prefix = 'technical_' if technical_only else ''
    path = RUN / (prefix+'controller.json')
    if path.exists():
        raise RuntimeError('Existing phase controller; do not duplicate')
    budget = p['technical_budget_seconds'] if technical_only else read(RUN / 'full_authorization.json')['budget_seconds']
    assert 0 < budget <= 10800
    now = datetime.now(timezone(timedelta(hours=3)))
    soft = time.time()+budget; hard = soft+30
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'technical_worker' if technical_only else 'worker']
    with (RUN / (prefix+'stdout.log')).open('w',encoding='utf-8') as out, (RUN / (prefix+'stderr.log')).open('w',encoding='utf-8') as err:
        child = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born = psutil.Process(child.pid).create_time()
        write(path,dict(pid=child.pid,create_time=born,controller_pid=os.getpid(),started_at=now.isoformat(),
            soft_timestamp=soft,save_stop_at=(now+timedelta(seconds=budget)).isoformat(),hard_stop_at=(now+timedelta(seconds=budget+30)).isoformat(),command=cmd))
        timed_out = False
        while child.poll() is None:
            if time.time() >= soft and not (RUN / 'stop.request').exists():
                (RUN / 'stop.request').write_text('Save completed videos; budget ended\n',encoding='utf-8')
            if time.time() >= hard:
                proc = psutil.Process(child.pid); assert abs(proc.create_time()-born) < .01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=20); timed_out=True; break
            time.sleep(1)
    write(RUN / (prefix+'controller_result.json'),dict(exit_code=child.returncode,worker_ended=True,hard_timeout=timed_out,
        ended=datetime.now(timezone.utc).isoformat(),chat_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','technical','technical_worker','controller','worker'])
    a = parser.parse_args()
    try:
        if a.mode == 'prepare': prepare()
        elif a.mode in ('technical','controller'): controller(a.mode == 'technical')
        else: worker(a.mode == 'technical_worker')
    except BaseException as exc:
        if a.mode in ('technical_worker','worker') and RUN.exists():
            write(RUN / 'status.json',dict(state='failed',stage=str(exc),updated=datetime.now(timezone.utc).isoformat()))
        raise
