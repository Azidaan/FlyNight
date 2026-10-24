"""One fresh-process recovery after MASA stopped reporting progress; no science changes."""
from pathlib import Path
from datetime import datetime
import ctypes
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read, write, sha, now
from scripts.z13_hota_deadline_guard import WindowsProcesses, ProcessEntry
W = ROOT / 'runs/z13_weekend_20261024'
OLD = ROOT / 'runs/z13_val20_matched_20261024/common_authorized'
REC = OLD.parent / 'common_authorized_recovery1'


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts import z13_val20_common_20261024 as S
    assert not REC.exists()
    p = read(OLD / 'protocol.json'); verify_sources(p)
    result = read(OLD / 'controller_result.json')
    assert result['exit_code'] == 1 and result['all_owned_ended'] and not result['remaining']
    assert '600s without original frame/stage progress' in read(OLD / 'status.json')['stage']
    c = read(OLD / 'controller.json'); m = read(OLD / 'completion.json'); ps = WindowsProcesses()
    identities = {**result['owned'], str(c['pid']):c['birth'], str(m['pid']):m['birth']}
    assert all(ps.creation(int(pid)) != birth for pid,birth in identities.items())
    jobs = list((OLD / 'jobs').glob('*.json')); assert len(jobs) == 118
    for marker in jobs:
        for name,digest in read(marker)['outputs'].items():
            assert sha(ROOT / name) == digest, name
    original_price = read(OLD / 'price.json'); assert original_price['proceed']
    remaining_videos = sum(not (OLD / 'jobs' / ('masa_all_'+v+'.json')).exists() for v in p['videos'])
    estimate = 2*original_price['first_missing_six_plus_masa_seconds']*remaining_videos+900
    available = int((datetime.fromisoformat(c['save_at'])-now()).total_seconds())
    assert estimate < available
    names = ['scripts/z13_common_authorized_recovery1_20261024.py',
             'scripts/z13_common_recovery1_console_20261024.ps1',
             'scripts/z13_launch_common_recovery1_20261024.ps1']
    proof = dict(at=now().isoformat(), prior_result_sha256=sha(OLD / 'controller_result.json'),
        all_prior_identities_ended=True, checked_identities=len(identities),
        verified_jobs=len(jobs), failed_job=read(OLD / 'live_child.json'),
        no_completed_or_partial_fly_masa_output_for_failed_video=True,
        observed='MASA fourth video ceased frame updates after500frames;600s watchdog ended its exact own tree. No data/guard traceback before termination. Root cause not established. GPU currently heavily used and WarThunder present;not proof of causality. One unchanged fresh-process retry,only after resources free;no automatic repeated failed retry.')
    stalled = proof['failed_job']['video']
    assert proof['failed_job']['mode'] == 'masa'
    fly = [a for a in p['systems'] if a not in S.ARMS]
    assert all(not (S.RUN/a/'masa'/(stalled+'.npz')).exists() for a in fly)
    assert all(not (S.RUN/a/'masa'/(stalled+'.npz.next.npz')).exists() for a in fly)
    write(REC / 'previous_end_audit.json', proof)
    p = {**p, 'created':now().isoformat(), 'budget_seconds':available,
        'save_at':c['save_at'], 'hard_stop':c['hard_stop'],
        'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),
        'sha256':{**p['sha256'], **{n:sha(ROOT/n) for n in names}},
        'service_recovery_after_masa_stall':dict(previous_protocol_sha256=sha(OLD/'protocol.json'),
            previous_end_audit_sha256=sha(REC/'previous_end_audit.json'),
            completed_jobs_retained=118, original_first_video_price=original_price,
            remaining_videos_including_partial=remaining_videos,
            conservative_remaining_seconds=estimate, available_seconds_at_prepare=available,
            original_absolute_deadline_unchanged=True,
            change='Resume only unfinished jobs in a fresh interpreter after resource availability. Same scientific builders,math,models,data,approvedcoordinateguard,600swatchdog and tolerances. Do not use skipped-job times as fresh estimate. No driver reset or foreign-process termination.',
            resource_gate=dict(no_aces_process=True,minimum_free_gpu_mib=8192,
                               maximum_gpu_utilization_percent=30,stable_seconds=30,poll_seconds=15))}
    write(REC/'protocol.json',p)
    write(REC/'price.json',{**original_price,'recovery_price':p['service_recovery_after_masa_stall']})
    for marker in jobs:write(REC/'jobs'/marker.name,read(marker))
    write(REC/'status.json',dict(state='queued',stage='118готовых заданий сохранены;ожидание свободной GPU',
        updated=now().isoformat(),jobs_done=118,jobs_total=231,
        save_at=p['save_at'],hard_stop=p['hard_stop']))
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath)
    card=next(x for x in cat['experiments'] if x['id']==S.IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='118заданий сохранены;после остановки MASA ожидается свободная GPU для единственного служебного возобновления.',
        decision='Продолжить только незавершённые задания в прежнем окне до25.10 06:00:27МСК после освобождения ресурсов. Не останавливать чужую игру. Повтор того же зависания заново диагностировать,не перезапускать бесконечно. До полного20valитога вывода об эффективности нет.')
    for k,v in fields.items():
        if card.get(k) and card[k]!=v:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=k,old=card[k],reason='Сохранены результаты после600с без прогресса MASA;ресурсная очередь без изменения научных настроек.'))
        card[k]=v
    card['evidence'].append(dict(kind='file',ref=(REC/'protocol.json').relative_to(ROOT).as_posix(),claim='118готовых заданий,прежняя цена/срок и ожидание ресурсов;старые логи/ошибка сохранены.'))
    write(catpath,cat)
    S.journal('MASA остановлена без прогресса;118результатов сохранены',
        '*Получено.* runs/z13_val20_matched_20261024/common_authorized/controller_result.json:exit1,'
        'своё дерево окончено15:28:01МСК. На четвёртом видео b230132b-ff8f2719 MASA '
        'перестала сообщать прогресс после500кадров;через600с прежний сторож завершил только '
        'свой процесс. До этого нет ошибки научных данных/допуска. Проверены118маркеров '
        'и хеши;все прежниеPID/birth окончены. Причина зависания пока не установлена.\n\n'
        '*Решение.* Новый служебный common_authorized_recovery1/protocol.json:один повтор '
        'незавершённой MASA в свежем процессе,118готовых заданий не повторяются. '
        'Сейчас GPU94% и запущена WarThunder;это наблюдение,не доказательство причины. '
        'Чужую игру не останавливаем. Ресурсная очередь ждёт отсутствияaces,не менее8192МиБ '
        'свободной памяти и загрузки<=30% в течение30с. Это только условие запуска,'
        'не научный порог. Цена по первоначальным910,516с/полный ролик:2×17×910,516+900='
        +str(estimate)+'с;перед запуском повторно сравнить с остатком старого окна. '
        'Сохранение25.10 06:00:07МСК,жёсткий конец06:00:27 неизменны. '
        'Повторный такой же сбой не перезапускается автоматически. '
        'Научные источники/веса/пороги и разрешённый допуск0,0001px неизменны. '
        'Неполный результат не является вердиктом об эффективности.')
    print('Prepared recovery:118verified jobs retained;original budget and price retained.',flush=True)


def bind():
    from scripts import z13_val20_common_authorized_20261024 as R
    R.REC=REC;R.__file__=str(Path(__file__));R.bind()
    return R


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    R=bind();p=read(REC/'protocol.json');verify_sources(p)
    for video in p['videos']:
        for arm in R.FLY:
            R.S.status('Только оставшиеся варианты;118готовых сохранены',variant=arm+' '+video)
            R.child('signature',arm,video)
        R.S.status('Общая MASA:только недостающий результат',variant=video)
        R.child('masa','all',video)
    R.evaluate()
    write(REC/'status.json',dict(state='completed',stage='10систем×20val,raw/MASAготовы',
        jobs_done=231,jobs_total=231,updated=now().isoformat()))


def monitor():
    import msvcrt
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    ps=WindowsProcesses();lock=(REC/'completion.lock').open('a+b')
    lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    e=read(REC/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:
        write(W/'val20_common_authorized_recovery1_failure.json',dict(at=now().isoformat(),result=e,status=read(REC/'status.json')));return
    execute('record_val20_common_authorized_recovery1',[str(Path(__file__)),'--record'])
    verify_and_graph('val20_common_authorized_recovery1')
    write(W/'val20_common_authorized_recovery1_complete.json',dict(at=now().isoformat(),publication_pending=True))


def gpu_state():
    ps=WindowsProcesses();k=ps.k;snap=k.CreateToolhelp32Snapshot(2,0)
    assert snap not in (None,ctypes.c_void_p(-1).value)
    games=[]
    try:
        entry=ProcessEntry();entry.size=ctypes.sizeof(entry)
        ok=k.Process32FirstW(snap,ctypes.byref(entry))
        while ok:
            if entry.exe.lower()=='aces.exe':games.append(dict(pid=int(entry.pid),birth=ps.creation(int(entry.pid))))
            ok=k.Process32NextW(snap,ctypes.byref(entry))
    finally:k.CloseHandle(snap)
    query=subprocess.run(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=15)
    assert query.returncode==0
    free,util=[int(x.strip()) for x in query.stdout.strip().split(',')]
    return dict(free_gpu_mib=free,gpu_utilization_percent=util,active_game_identities=games)


def queue():
    import msvcrt
    p=read(REC/'protocol.json');ps=WindowsProcesses()
    lock=(REC/'queue.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    write(REC/'queue.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),script_sha256=sha(Path(__file__))))
    gate=p['service_recovery_after_masa_stall']['resource_gate'];stable=None
    estimate=p['service_recovery_after_masa_stall']['conservative_remaining_seconds']
    while True:
        available=(datetime.fromisoformat(p['save_at'])-now()).total_seconds()
        if (REC/'stop.request').exists() or available<=estimate:
            write(REC/'queue_result.json',dict(at=now().isoformat(),state='not_started',reason='Remaining original window no longer fits conservative workload,or own stop requested.',available_seconds=available,required_seconds=estimate,completed_jobs_retained=118))
            write(REC/'status.json',dict(state='paused',stage='Остаток сохранён;прежнего окна уже недостаточно для полного20val',updated=now().isoformat(),jobs_done=118,jobs_total=231))
            return
        try:
            resources=gpu_state()
            ready=not resources['active_game_identities'] and resources['free_gpu_mib']>=gate['minimum_free_gpu_mib'] and resources['gpu_utilization_percent']<=gate['maximum_gpu_utilization_percent']
        except Exception as error:
            resources=dict(resource_check_error=repr(error));ready=False
        if ready:
            if stable is None:stable=time.monotonic()
        else:stable=None
        write(REC/'status.json',dict(state='queued',stage='Ожидание свободной GPU;игра и чужие процессы не затрагиваются',updated=now().isoformat(),jobs_done=118,jobs_total=231,resources=resources,stable_seconds=0 if stable is None else round(time.monotonic()-stable,1),save_at=p['save_at'],hard_stop=p['hard_stop']))
        if stable is not None and time.monotonic()-stable>=gate['stable_seconds']:
            from scripts.z13_localization_labels_20261023 import verify_sources
            verify_sources(p);assert not (REC/'controller.json').exists()
            for label,args in (
                ('controller',['scripts/z13_bounded_recovery_controller_20261022.py','--run',str(REC),'--script',str(Path(__file__))]),
                ('monitor',[str(Path(__file__)),'--monitor'])):
                with (REC/(label+'.stdout.log')).open('w',encoding='utf-8') as out,(REC/(label+'.stderr.log')).open('w',encoding='utf-8') as err:
                    child=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',*args],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
                    write(REC/(label+'_launch.json'),dict(pid=child.pid,birth=ps.creation(child.pid),at=now().isoformat()))
            write(REC/'queue_result.json',dict(at=now().isoformat(),state='launched',remaining_original_seconds=available,resources=resources))
            return
        time.sleep(gate['poll_seconds'])


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--queue' in sys.argv:
        try:queue()
        except Exception as error:write(REC/'queue_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as error:write(W/'val20_common_authorized_recovery1_monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(REC/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif '--record' in sys.argv:bind().record()
    elif '--job' in sys.argv:
        R=bind();mode,arm,video=sys.argv[2:5];p=read(REC/'protocol.json')
        assert video in p['videos'] and ((mode=='signature' and arm in R.FLY) or (mode=='masa' and arm=='all'))
        result=R.S.build_signature(arm,video,False) if mode=='signature' else R.build_missing_masa(video)
        paths=[R.S.RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [R.S.RUN/a/'masa'/(video+'.npz') for a in R.S.SYSTEMS]
        write(REC/'jobs'/f'{mode}_{arm}_{video}.json',dict(at=now().isoformat(),result=result,outputs={f.relative_to(ROOT).as_posix():sha(f) for f in paths}))
    else:raise RuntimeError('Explicit recovery mode required')
