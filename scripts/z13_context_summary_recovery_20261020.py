"""Recover only the saved-row summary; original sources, predictions and failure remain intact."""
import argparse
from datetime import datetime, timedelta, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_parallel_next import read, write, sha

RUN = ROOT / 'runs/z13_context_transfer_20261020'
PREFIX = 'recovery_summary_'


def selected_video_useful(rows, chosen):
    from scripts.z13_continue import per_video_useful
    from scripts.z13_score_swap import key
    # Ranking was performed over ALL rows. This metric has always counted only kind == 1.
    return per_video_useful([dict(r, useful=bool(r['useful']) and key(r) in chosen)
                             for r in rows if r['kind'] == 1])


def corrected_summary():
    from scripts import z13_context_transfer_20261020 as frozen
    source = inspect.getsource(frozen.summarize)
    old = "return per_video_useful([dict(x,useful=bool(x['useful']) and key(x) in ids) for x in rows])"
    assert source.count(old) == 1, 'Frozen aggregation does not match the diagnosed failure'
    fixed = source.replace(old, 'return selected_video_useful(rows, ids)', 1)
    namespace = dict(vars(frozen), selected_video_useful=selected_video_useful)
    exec(compile(fixed, str(Path(__file__).resolve()) + ':fixed_summary', 'exec'), namespace)
    return namespace['summarize'], fixed


def state(stage, done, final=False):
    write(RUN / (PREFIX+'status.json'), dict(state='completed' if final else 'running',
          stage=stage, done=done, total=4, updated=datetime.now(timezone.utc).isoformat()))


def prepare():
    path = RUN / (PREFIX+'protocol.json')
    assert not path.exists() and not (RUN / 'report.json').exists(), 'Refuse duplicate recovery'
    end = read(RUN / 'controller_result.json')
    assert end['worker_ended'] and end['exit_code'] == 1
    p = read(RUN / 'protocol.json')
    from scripts import z13_context_transfer_20261020 as frozen
    frozen.verify(p)
    names = ['scripts/z13_context_summary_recovery_20261020.py',
             'scripts/z13_context_summary_recovery_progress.ps1', 'tests/test_z13_context_summary_recovery.py',
             'runs/z13_map_transfer_20261018/C/predictions.json']
    for arm in p['arms']:
        names += [f'runs/z13_context_transfer_20261020/{arm}/{name}' for name in ('predictions.json', 'done.json')]
        names += [f'runs/z13_context_transfer_20261020/{arm}/videos/{v}/directional_rows.json' for v in p['videos']]
    for name in ('status.json', 'controller_result.json'):
        write(RUN / (PREFIX+'previous_'+name), read(RUN / name))
    summary, fixed = corrected_summary()
    (RUN / (PREFIX+'effective_source.py')).write_text(fixed, encoding='utf-8')
    write(path, dict(authorization='User explicitly requested continuation after the saved-row recovery proposal',
          budget_seconds=180, shutdown_seconds=15, original_protocol='protocol.json',
          original_failure='controller_result.json', original_failure_preserved=True,
          correction='Only positive-kind rows are passed to the existing positive-case metric, after unchanged ranking over all rows. Unknown outcomes stay unknown.',
          predictions_recomputed=False, core_training=False, gpu=False, new_data=False,
          thresholds_changed=False, bootstrap_changed=False, sha256={x:sha(ROOT/x) for x in names}))
    state('Подготовлено исправление сборки сохранённых строк', 0)


def worker():
    p = read(RUN / 'protocol.json')
    recovery = read(RUN / (PREFIX+'protocol.json'))
    launch_path = RUN / (PREFIX+'controller.json')
    for _ in range(100):
        if launch_path.exists():
            break
        time.sleep(.02)
    launch = read(launch_path)
    assert time.time() < launch['soft_timestamp']
    from scripts import z13_context_transfer_20261020 as frozen
    state('Проверка неизменности исходников и готовых предсказаний', 0)
    frozen.verify(p)
    for name, digest in recovery['sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert not (RUN / 'report.json').exists()
    state('Сборка прежних сравнений с исправленным порядком обработки', 1)
    summarize, _ = corrected_summary()
    summarize(p)
    state('Проверка полноты результата и сохранности исходных файлов', 2)
    from scripts.z13_continue import per_video_useful
    from scripts.z13_score_swap import selected_keys
    import numpy as np
    c = read(RUN / 'C/predictions.json')
    r = read(RUN / 'R1/predictions.json')
    mean_map = read(ROOT / 'runs/z13_map_transfer_20261018/C/predictions.json')['map_utility']
    mean_rows = [x for v in p['videos'] for x in read(frozen.DEV / 'C' / v / 'center_rows.json')]
    variants = {'C_directional': c['original'], 'C_mean': mean_rows, 'R1_directional':r['original'],
                'C_context_map':c['map_utility'], 'C_mean_map':mean_map,
                'C_context_geometry':c['geometry'], 'R1_context_map':r['map_utility']}
    levels = {}
    for name, rows in variants.items():
        assert len(rows) == 9314 and set(x['video'] for x in rows) == set(p['videos'])
        for row in rows:
            assert 'useful' in row or row['kind'] == -1, 'Missing outcome in an evaluated case'
        pv = per_video_useful(rows) if name in ('C_directional','C_mean','R1_directional') else selected_video_useful(rows,selected_keys(rows,.01))
        assert len(pv) == 35
        levels[name] = float(np.mean(list(pv.values())))
    report = read(RUN / 'report.json')
    assert len(report['localization']) == 2 and len(report['selection_at1percent']) == 3
    for group in ('localization', 'selection_at1percent'):
        assert all(x['n'] == 35 for x in report[group].values())
    frozen.verify(p)
    for name, digest in recovery['sha256'].items():
        assert sha(ROOT / name) == digest, name
    assert time.time() < launch['soft_timestamp']
    write(RUN / (PREFIX+'audit.json'), dict(passed=True, original_sources_unchanged=True,
          predictions_unchanged=True, original_failure_preserved=True,
          original_worker_exit_code=read(RUN / 'controller_result.json')['exit_code'],
          missing_outcomes=sum('useful' not in x for x in mean_map), missing_outcomes_kind=-1, unknowns_imputed=False,
          levels_mean_per_video=levels, comparisons_complete=True, videos=35,
          hypotheses_or_thresholds_changed=False, predictions_recomputed=False))
    state('Сводка сохранена; исходные предсказания и ошибка сохранены', 4, True)


def controller():
    import psutil
    path = RUN / (PREFIX+'controller.json')
    assert not path.exists(), 'Refuse duplicate summary controller'
    recovery = read(RUN / (PREFIX+'protocol.json'))
    now = datetime.now(timezone(timedelta(hours=3)))
    soft = time.time()+recovery['budget_seconds']
    hard = soft+recovery['shutdown_seconds']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    with (RUN / (PREFIX+'stdout.log')).open('w',encoding='utf-8') as out, (RUN / (PREFIX+'stderr.log')).open('w',encoding='utf-8') as err:
        child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],cwd=ROOT,
                env=env,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS)
        born = psutil.Process(child.pid).create_time()
        write(path,dict(pid=child.pid,create_time=born,controller_pid=os.getpid(),started_at=now.isoformat(),
              soft_timestamp=soft, save_stop_at=(now+timedelta(seconds=recovery['budget_seconds'])).isoformat(),
              hard_stop_at=(now+timedelta(seconds=recovery['budget_seconds']+recovery['shutdown_seconds'])).isoformat()))
        timed_out=False
        while child.poll() is None:
            if time.time() >= hard:
                assert abs(psutil.Process(child.pid).create_time()-born)<.01
                subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],check=True,capture_output=True)
                child.wait(timeout=15); timed_out=True; break
            time.sleep(.25)
    write(RUN / (PREFIX+'controller_result.json'),dict(exit_code=child.returncode,worker_ended=True,
          hard_timeout=timed_out,ended=datetime.now(timezone.utc).isoformat()))
    if child.returncode:
        write(RUN / (PREFIX+'status.json'),dict(state='failed',stage='См. отдельный журнал восстановления',
              updated=datetime.now(timezone.utc).isoformat()))
    return child.returncode


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','controller','worker'])
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='controller': sys.exit(controller())
    else: worker()
