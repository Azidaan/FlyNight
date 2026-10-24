"""Resume fixed ten-system evaluation under the author's coordinate-only guard."""
from pathlib import Path
from datetime import datetime
import inspect
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import z13_val20_common_20261024 as S
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_coordinate_replay_guard import compare_outputs
from scripts.z13_hota_deadline_guard import WindowsProcesses

PREVIOUS = S.STAGE
REC = S.RUN / 'common_authorized'
NATIVE = S.RUN / 'native_masa_recovery2'
W = ROOT / 'runs/z13_weekend_20261024'
FLY = tuple(a for a in S.SYSTEMS if a not in S.ARMS)
ORIGINAL_SIGNATURE = S.build_signature
ORIGINAL_EVALUATE = S.evaluate


def prepare():
    import numpy as np
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists()
    S.ended(NATIVE)
    previous = read(PREVIOUS / 'protocol.json')
    verify_sources(previous)
    controller = read(PREVIOUS / 'controller.json')
    failed = read(PREVIOUS / 'controller_result.json')
    ps = WindowsProcesses()
    assert failed['exit_code'] == 1 and failed['all_owned_ended'] and not failed['remaining']
    assert ps.creation(controller['pid']) != controller['birth']
    assert all(ps.creation(int(pid)) != birth for pid, birth in failed['owned'].items())
    seconds = int((datetime.fromisoformat(controller['save_at']) - now()).total_seconds())
    assert seconds > 3600
    approval = dict(at=now().isoformat(), status='approved',
        author_answer='Разрешаю этот служебный допуск',
        question_item_id='["request_user_input_async","call_788f2132fa6b44c59facbbc8ca50eb54",0]',
        coordinate_absolute_tolerance=0.0001,
        exact_fields=['row_count', 'frame', 'track_id', 'class', 'confidence'],
        no_relative_tolerance=True, scientific_settings_changed=False,
        original_failure_retained=(PREVIOUS / 'controller_result.json').relative_to(ROOT).as_posix())
    assert not (W / 'val20_replay_authorized.json').exists()
    write(W / 'val20_replay_authorized.json', approval)
    pending = W / 'val20_replay_pending_author_decision.json'
    backup = W / 'val20_replay_pending_author_decision.before_approval.json'
    assert not backup.exists()
    shutil.copy2(pending, backup)
    write(pending, {**read(pending), 'status': 'approved', 'resolved_at': approval['at'],
                    'answer': approval['author_answer'], 'authorization': 'val20_replay_authorized.json'})
    native = read(NATIVE / 'report.json')
    assert native['videos'] == previous['videos'] and len(native['pooled']) == 8
    reuse = {}
    markers = list((NATIVE / 'jobs').glob('*.json'))
    assert len(markers) == 105
    for marker in markers:
        row = read(marker)
        for name, digest in row['outputs'].items():
            assert sha(ROOT / name) == digest
            reuse[name] = digest
        # Each new-video MASA marker will combine retained4 and missing6 arms.
        if not marker.name.startswith('masa_all_'):
            write(REC / 'jobs' / marker.name, {**row,
                'reused_from': marker.relative_to(ROOT).as_posix()})
    video = previous['probe_video']
    failed_output = PREVIOUS / 'probe/legacyD/signatures' / video / 'signatures.npz'
    reference = S.OLD / 'legacyD/signatures' / video / 'signatures.npz'
    with np.load(failed_output) as a, np.load(reference) as b:
        replay = compare_outputs(a['outputs'], b['outputs'])
    write(REC / 'jobs' / ('probe_signature_legacyD_' + video + '.json'),
        dict(at=now().isoformat(), outputs={failed_output.relative_to(ROOT).as_posix(): sha(failed_output)},
             result=dict(probe=True, exact_replay=replay['exact'], replay_guard=replay,
                         signature_sha256=sha(failed_output)),
             reused_from_failed_exact_probe=True,
             explanation='Existing unchanged output passes newly authorized coordinate guard; original failed exact result retained. No GPU replay repeated.'))
    extra = ['scripts/z13_val20_common_authorized_20261024.py',
             'scripts/z13_coordinate_replay_guard.py',
             'tests/test_z13_coordinate_replay_guard.py',
             'scripts/z13_val20_common_finish_20261024.py',
             'scripts/z13_val20_native_masa_recovery2_20261024.py',
             'scripts/z13_val20_common_console.ps1',
             'runs/z13_weekend_20261024/val20_replay_authorized.json',
             (NATIVE / 'report.json').relative_to(ROOT).as_posix(),
             failed_output.relative_to(ROOT).as_posix()]
    protocol = {**previous, 'created': now().isoformat(), 'budget_seconds': seconds,
        'save_at': controller['save_at'], 'hard_stop': controller['hard_stop'],
        'stop_request_path': (REC / 'stop.request').relative_to(ROOT).as_posix(),
        'sha256': {**previous['sha256'], **{n: sha(ROOT / n) for n in extra}},
        'authorized_service_recovery': dict(authorization=approval,
            original_protocol_sha256=sha(PREVIOUS / 'protocol.json'),
            original_result_sha256=sha(PREVIOUS / 'controller_result.json'),
            original_86400second_absolute_end_unchanged=True,
            reused_native_report_sha256=sha(NATIVE / 'report.json'),
            native_completed_outputs=reuse,
            gates='Only FlyVis/control-model old outputs: absolute coordinate error<=0.0001px;count/frame/ID/class/confidence exact. Native raw and previous baseline MASA remain exact. Existing86verified markers retained. No output rounded,altered or substituted. Old failed bit-exact criterion retained. Missing6arms only;native4raw/MASA/evaluations reused unchanged. Missing6 MASA uses identical per-frame backbone and per-arm ROI operations. Same-frame fp16 guard remains exact.',
            price='After first fixed new video, measure only missing6signatures+theirMASA. Original2xwalltime*remaining19+900 reserve must fit original25Oct06:00:07soft deadline. Control time not reset;stage budget never extended.')}
    assert len(list((REC / 'jobs').glob('*.json'))) == 86
    write(REC / 'protocol.json', protocol)
    write(REC / 'status.json', dict(state='ready', stage='86готовых результатов проверены;разрешён координатный контроль',
        jobs_done=86, jobs_total=231, updated=now().isoformat()))
    catalog_path = ROOT / 'experiments/project_history.json'
    catalog = read(catalog_path)
    card = next(c for c in catalog['experiments'] if c['id'] == S.IDENT)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    fields = dict(summary='Автор разрешил координатный служебный допуск;общее сравнение продолжается с86проверенных заданий.',
        decision='Продолжить только недостающие6вариантов на техже20val;готовые4трекера/MASA не повторять. Исходный конец25.10 06:00:27МСК не продлён. Первый новый полный ролик проверяет цену остатка. Старый точный контроль остаётся непройденным и сохранённым.')
    for key, value in fields.items():
        if card.get(key) and card[key] != value:
            card.setdefault('revisions', []).append(dict(date=now().strftime('%Y-%m-%d'),
                commit=head, field=key, old=card[key], reason='Явное разрешение автора на отдельный координатный служебный допуск.'))
        card[key] = value
    card['state'] = 'running'
    card['acceptance'][0]['historical'] = True
    card['acceptance'].append(dict(criterion='Дописан после разрешения автора,до основного20val-прогона: '+protocol['authorized_service_recovery']['gates'], met=None))
    card.setdefault('evidence', []).append(dict(kind='file',
        ref=(REC / 'protocol.json').relative_to(ROOT).as_posix(),
        claim='Явное разрешение,новый служебный контроль,86готовых результатов и исходный абсолютный срок.'))
    write(catalog_path, catalog)
    S.journal('автор разрешил координатный контроль;продолжается общая20val-проверка',
        '*Решение автора.* «Разрешаю этот служебный допуск». '
        'runs/z13_weekend_20261024/val20_replay_authorized.json:только координаты<=0,0001px,'
        'число строк/кадры/ID/классы/уверенности точно. Исходная ошибка сохранена,'
        'непройденный критерий не удалён.\n\n*Постановка продолжения.* '
        'runs/z13_val20_matched_20261024/common_authorized/protocol.json. '
        'Готовые86заданий проверены,включая прежний legacyD выход без повторного GPU. '
        'Новые вычисления:5недостающих контролей,6вариантов×20видео и их общая MASA. '
        'Готовые4native/MASA и их оценки не перезаписываются. '
        'Веса/данные/пороги/критерии научного результата прежние. '
        'Цена первого нового ролика:2×время недостающих6вариантов иMASA×19+900с '
        'должна помещаться в остаток первоначального окна до25.10 06:00:07МСК,'
        'жёсткий конец06:00:27. Никакого продления86400с.')
    print('Prepared:86verified jobs retained;only approved coordinate guard changed.', flush=True)


def status_read(path):
    for attempt in range(40):
        try:
            return read(path)
        except (PermissionError, FileNotFoundError):
            if attempt == 39:
                raise
            time.sleep(.05)


def bind():
    S.STAGE = REC
    S.T0 = time.monotonic()
    S.__file__ = str(Path(__file__))
    # Keep the scientific builder unchanged except its probe assertion/receipt.
    source = inspect.getsource(ORIGINAL_SIGNATURE)
    old = "for key in keys:assert np.array_equal(actual[key],old[key]),('Old control replay',arm,key)"
    replacement = "for key in keys:\n                    if arm in FLY and key=='outputs':\n                        replay=compare_outputs(actual[key],old[key])\n                    else:\n                        assert np.array_equal(actual[key],old[key]),('Old control replay',arm,key)"
    assert source.count(old) == 1
    source = source.replace(old, replacement)
    old_return = 'return dict(signature_sha256=sha(path),probe=probe,exact_replay=probe or arm in ARMS)'
    assert source.count(old_return) == 1
    source = source.replace(old_return,
        "return dict(signature_sha256=sha(path),probe=probe,exact_replay=replay['exact'] if probe and arm in FLY else probe or arm in ARMS,replay_guard=replay if probe and arm in FLY else None)")
    namespace = {**vars(S), 'FLY': FLY, 'compare_outputs': compare_outputs}
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    S.build_signature = namespace['build_signature']
    S.child = child


def child(mode, arm, video, probe=False):
    marker = REC / 'jobs' / f'{"probe_" if probe else ""}{mode}_{arm}_{video}.json'
    if marker.exists():
        item = read(marker)
        for name, digest in item['outputs'].items():
            assert sha(ROOT / name) == digest
        return item
    ps = WindowsProcesses()
    log = REC / 'logs' / marker.stem
    log.parent.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    with log.with_suffix('.stdout.log').open('w', encoding='utf-8') as out, log.with_suffix('.stderr.log').open('w', encoding='utf-8') as err:
        proc = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-u', str(Path(__file__)),
            '--job', mode, arm, video, *(['--probe'] if probe else [])], cwd=ROOT,
            stdout=out, stderr=err, creationflags=subprocess.IDLE_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW)
        birth = ps.creation(proc.pid)
        assert birth is not None
        write(REC / 'live_child.json', dict(pid=proc.pid, birth=birth, mode=mode, arm=arm, video=video))
        try:
            while proc.poll() is None:
                S.check()
                stamp = status_read(REC / 'status.json')['updated']
                assert (now()-datetime.fromisoformat(stamp)).total_seconds() <= 600, '600s without original frame/stage progress'
                time.sleep(.5)
        except BaseException:
            if proc.poll() is None and ps.creation(proc.pid) == birth:
                result = subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                write(REC / 'child_cleanup.json', dict(at=now().isoformat(), pid=proc.pid,
                    birth=birth, exit_code=result.returncode))
                proc.wait(timeout=20)
            raise
        assert proc.returncode == 0, (mode, arm, video, proc.returncode)
    item = read(marker)
    item['wall_seconds_with_imports'] = time.monotonic()-began
    write(marker, item)
    return item


def build_missing_masa(video):
    import torch
    torch.set_num_threads(2)
    torch.set_default_device('cpu')
    from tracking.masa_r50 import build
    from tracking.gpu_lock import GpuLock
    from scripts.z13_shared_masa import extract_video
    protocol = read(REC / 'protocol.json')
    for arm in S.ARMS:
        path = S.RUN / arm / 'masa' / (video+'.npz')
        name = path.relative_to(ROOT).as_posix()
        assert sha(path) == protocol['authorized_service_recovery']['native_completed_outputs'][name]
    paths = {a: S.RUN / a / 'signatures' / video / 'signatures.npz' for a in FLY}
    outputs = {a: S.RUN / a / 'masa' / (video+'.npz') for a in FLY}
    assert all(not p.exists() for p in outputs.values())
    with GpuLock('flyvis_fovea_gpu'):
        return extract_video(build(), ROOT / f'data/bdd100k/videos/val/{video}.mov',
            paths, outputs, progress=lambda i, n: S.status('Общая MASA:только6недостающих вариантов', variant=video, done=i, total=n))


def evaluate():
    source = inspect.getsource(ORIGINAL_EVALUATE)
    init = 'per={};pooled={};hashes={};links={}'
    assert source.count(init) == 1
    source = source.replace(init, "per=dict(_native['per_video']);pooled=dict(_native['pooled']);hashes=dict(_native['output_sha256']);links=dict(_native['masa_links'])")
    assert source.count('for arm in SYSTEMS:') == 1
    source = source.replace('for arm in SYSTEMS:', 'for arm in FLY:')
    namespace = {**vars(S), 'FLY': FLY, '_native': read(NATIVE / 'report.json')}
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    namespace['evaluate']()


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    protocol = read(REC / 'protocol.json')
    verify_sources(protocol)
    for arm in FLY:
        S.status('Разрешённый контроль прежнего видео', variant=arm+' '+protocol['probe_video'])
        child('signature', arm, protocol['probe_video'], True)
    for index, video in enumerate(protocol['videos']):
        began = time.monotonic()
        for arm in FLY:
            S.status('Только недостающие6вариантов', variant=arm+' '+video)
            child('signature', arm, video)
        child('masa', 'all', video)
        if index == 0:
            measured = time.monotonic()-began
            estimate = 2*measured*19+900
            available = min(protocol['budget_seconds']-(time.monotonic()-S.T0),
                (datetime.fromisoformat(protocol['save_at'])-now()).total_seconds())
            write(REC / 'price.json', dict(first_missing_six_plus_masa_seconds=measured,
                conservative_remaining_seconds=estimate, available_seconds=available,
                already_completed_native_work_excluded=True, proceed=estimate < available))
            assert estimate < available, 'Remaining full20 does not fit original absolute deadline'
    evaluate()
    write(REC / 'status.json', dict(state='completed', stage='10систем×20val,raw/MASAготовы',
        jobs_done=231, jobs_total=231, updated=now().isoformat()))


def record():
    from scripts import z13_val20_common_finish_20261024 as F
    source = inspect.getsource(F.record)
    old = "for item in c['acceptance']:item.update(met=True,evidence=source+';verified_summary.json;controller_result.json;231jobs')"
    assert source.count(old) == 1
    source = source.replace(old,
        "for item in c['acceptance']:\n        if not item.get('historical'):\n            item.update(met=True,evidence=source+';verified_summary.json;controller_result.json;231jobs')")
    namespace = dict(vars(F))
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    namespace['record']()


def monitor():
    import msvcrt
    from scripts.z13_weekend_driver_20261024 import execute, verify_and_graph
    ps = WindowsProcesses()
    lock = (REC / 'completion.lock').open('a+b')
    lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    write(REC / 'completion.json', dict(pid=os.getpid(), birth=ps.creation(os.getpid()), at=now().isoformat(), sha256=sha(Path(__file__))))
    while not (REC / 'controller_result.json').exists():
        time.sleep(5)
    controller = read(REC / 'controller.json')
    while ps.creation(controller['pid']) == controller['birth']:
        time.sleep(.5)
    result = read(REC / 'controller_result.json')
    if result['exit_code'] != 0 or not result['all_owned_ended'] or result['remaining']:
        write(W / 'val20_common_authorized_failure.json', dict(at=now().isoformat(), result=result,
            status=status_read(REC / 'status.json')))
        return
    execute('record_val20_common_authorized', [str(Path(__file__)), '--record'])
    verify_and_graph('val20_common_authorized')
    write(W / 'val20_common_authorized_complete.json', dict(at=now().isoformat(), publication_pending=True))


if __name__ == '__main__':
    if '--prepare' in sys.argv:
        prepare()
    else:
        bind()
        if '--worker' in sys.argv:
            try:
                worker()
            except Exception as error:
                write(REC / 'status.json', dict(state='failed', stage=repr(error), updated=now().isoformat()))
                raise
        elif '--job' in sys.argv:
            mode, arm, video = sys.argv[2:5]
            probe = '--probe' in sys.argv
            protocol = read(REC / 'protocol.json')
            assert video in ([protocol['probe_video']] if probe else protocol['videos'])
            assert (mode == 'signature' and arm in FLY) or (mode == 'masa' and arm == 'all' and not probe)
            result = S.build_signature(arm, video, probe) if mode == 'signature' else build_missing_masa(video)
            base = REC / 'probe' if probe else S.RUN
            paths = [base / arm / 'signatures' / video / 'signatures.npz'] if mode == 'signature' else [S.RUN / a / 'masa' / (video+'.npz') for a in S.SYSTEMS]
            write(REC / 'jobs' / f'{"probe_" if probe else ""}{mode}_{arm}_{video}.json',
                dict(at=now().isoformat(), result=result, outputs={p.relative_to(ROOT).as_posix(): sha(p) for p in paths}))
        elif '--record' in sys.argv:
            record()
        elif '--monitor' in sys.argv:
            try:
                monitor()
            except Exception as error:
                write(W / 'val20_common_authorized_monitor_failure.json', dict(at=now().isoformat(), error=repr(error)))
                raise
        else:
            raise RuntimeError('Explicit authorized recovery mode required')
