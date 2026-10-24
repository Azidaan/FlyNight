"""Retain verified diagnostic output and capture own stacks before any new timeout."""
from pathlib import Path
from datetime import datetime
import os
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
OLD=ROOT/'runs/z13_val20_matched_20261024/common_authorized_recovery1'
DIAG=OLD.parent/'signature_stall_diagnostic'
REC=OLD.parent/'common_authorized_recovery2'
W=ROOT/'runs/z13_weekend_20261024'


def prepare():
    import numpy as np
    from scripts import z13_val20_common_20261024 as S
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();S.ended(DIAG)
    p=read(OLD/'protocol.json');verify_sources(p);verify_sources(read(DIAG/'protocol.json'))
    e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');m=read(OLD/'completion.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and e['all_owned_ended'] and not e['remaining']
    ids={**e['owned'],str(c['pid']):c['birth'],str(m['pid']):m['birth']}
    assert all(ps.creation(int(pid))!=birth for pid,birth in ids.items())
    markers=list((OLD/'jobs').glob('*.json'));assert len(markers)==180
    for marker in markers:
        for name,digest in read(marker)['outputs'].items():assert sha(ROOT/name)==digest
    d=read(DIAG/'report.json');assert d['passed_previous_frame950'] and d['output_directory_only_changed']
    assert d['complete_video_frames']==1204 and not d['root_cause_established']
    assert d['stack_samples'] and all(x['exit_code']==0 for x in d['stack_samples'])
    for sample in d['stack_samples']:assert sha(ROOT/sample['path'])==sample['sha256']
    arm='centerR1';video='b23adb0d-72704b27'
    source=DIAG/arm/'signatures'/video/'signatures.npz'
    assert sha(source)==d['outputs'][source.relative_to(ROOT).as_posix()]
    with np.load(source) as z:assert int(z['frame_count'])==1204 and np.isfinite(z['outputs']).all()
    dest=S.RUN/arm/'signatures'/video/'signatures.npz'
    assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(dest)==sha(source)
    for marker in markers:write(REC/'jobs'/marker.name,read(marker))
    write(REC/'jobs'/f'signature_{arm}_{video}.json',dict(at=now().isoformat(),
        result=d['result'],outputs={dest.relative_to(ROOT).as_posix():sha(dest)},
        reused_instrumented_diagnostic=DIAG.relative_to(ROOT).as_posix(),
        original_scientific_builder_output_retained=True))
    original_price=read(OLD/'price.json')['recovery_price']['original_first_video_price']
    remaining_videos=sum(not (REC/'jobs'/('masa_all_'+v+'.json')).exists() for v in p['videos'])
    estimate=2*original_price['first_missing_six_plus_masa_seconds']*remaining_videos+900
    available=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert estimate<available
    p={**p,'created':now().isoformat(),'budget_seconds':available,'save_at':c['save_at'],
        'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),
        'sha256':{**p['sha256'],'scripts/z13_common_authorized_recovery2_20261024.py':sha(Path(__file__)),
            'scripts/z13_signature_stall_diagnostic_20261024.py':sha(ROOT/'scripts/z13_signature_stall_diagnostic_20261024.py'),
            '.venv/Scripts/py-spy.exe':sha(ROOT/'.venv/Scripts/py-spy.exe'),
            (DIAG/'report.json').relative_to(ROOT).as_posix():sha(DIAG/'report.json'),
            dest.relative_to(ROOT).as_posix():sha(dest)},
        'service_recovery2':dict(previous_result_sha256=sha(OLD/'controller_result.json'),
            verified_jobs_retained=181,diagnostic_sha256=sha(DIAG/'report.json'),
            diagnostic_result='Same unfinished R1 video completed all1204frames with unchanged arithmetic;prior stall not reproduced. Native/Python samples retained;they do not establish original cause.',
            change='Reuse validated full diagnostic output;do not compute it again. Before any future600s timeout,capture exact own Python/native stacks at120/240/360/480s without progress. Observation only,no locals. No weakened watchdog or coordinate guard. No automatic failed retry.',
            original_first_video_price=original_price,remaining_videos_including_partial=remaining_videos,
            conservative_remaining_seconds=estimate,available_seconds=available,
            original_absolute_end_unchanged=True)}
    write(REC/'protocol.json',p);write(REC/'price.json',p['service_recovery2'])
    write(REC/'status.json',dict(state='ready',stage='181результат проверен;остаток после диагностики',updated=now().isoformat(),jobs_done=181,jobs_total=231))
    catpath=ROOT/'experiments/project_history.json';cat=read(catpath);card=next(x for x in cat['experiments'] if x['id']==S.IDENT)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
    fields=dict(summary='181задание проверено;незавершённый R1 прошёл инструментальный повтор,продолжается остаток в исходном окне.',
        decision='Использовать проверенный полный R1выход диагностического повтора;не вычислять готовое снова. Продолжить только остаток,с предварительными снимками собственного стека при новом простое. Причина прежних зависаний не установлена. Исходный конец25.10 06:00:27МСК не продлевать;до полного20valитога эффективности не оценивать.')
    for key,value in fields.items():
        if card.get(key) and card[key]!=value:card.setdefault('revisions',[]).append(dict(date=now().strftime('%Y-%m-%d'),commit=head,field=key,old=card[key],reason='Полный инструментальный повтор незавершённого R1 и проверка181результата;старые сбои сохранены.'))
        card[key]=value
    card['evidence'].append(dict(kind='file',ref=(REC/'protocol.json').relative_to(ROOT).as_posix(),claim='Диагностический повтор,181готовый результат и наблюдение стека без изменения научных настроек.'))
    write(catpath,cat)
    S.journal('инструментальный повтор R1 завершён;сохранён181результат',
        '*Получено.* signature_stall_diagnostic/report.json:неизменный R1выход получен '
        'для всех1204кадров,пройдено прежнее место950. Снимки Python/нативного стека '
        'сохранены с хешами. Прежнее зависание не воспроизвелось,его причина не '
        'установлена;нормальные снимки не доказывают прежнюю точку ожидания. '
        'Валидный NPZсохранён как результат незавершённого задания и не повторяется.\n\n'
        '*Решение.* common_authorized_recovery2/protocol.json:181маркер/хеш проверен. '
        'При новом отсутствии прогресса снимаются собственные стеки на120/240/360/480с '
        'до прежнего600с завершения. Это служебное наблюдение,не изменение вычислений. '
        'Исходный мягкий конец25.10 06:00:07МСК/жёсткий06:00:27 прежние. '
        'Цена по первоначальным910,516с×'+str(remaining_videos)+'×2+900='+str(estimate)+
        'с против'+str(available)+'с остатка. Оба прежних кода ошибок сохранены, '
        'научные параметры/данные/разрешённый допуск не изменены. '
        'Вывода по неполной эффективности нет.')
    print('Prepared recovery2:181verified jobs retained;diagnostic output reused;original deadline unchanged.',flush=True)


def bind():
    from scripts import z13_common_authorized_recovery1_20261024 as Previous
    Previous.REC=REC;Previous.__file__=str(Path(__file__))
    R=Previous.bind();R.child=child;R.S.child=child
    return R


def child(mode,arm,video,probe=False):
    assert not probe,'All control jobs already retained'
    from scripts import z13_val20_common_authorized_20261024 as R
    marker=REC/'jobs'/f'{mode}_{arm}_{video}.json'
    if marker.exists():
        item=read(marker)
        for name,digest in item['outputs'].items():assert sha(ROOT/name)==digest
        return item
    ps=WindowsProcesses();log=REC/'logs'/marker.stem;log.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic();captured=set()
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',mode,arm,video],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW)
        birth=ps.creation(proc.pid);assert birth is not None
        write(REC/'live_child.json',dict(pid=proc.pid,birth=birth,mode=mode,arm=arm,video=video))
        try:
            while proc.poll() is None:
                R.S.check();stamp=R.status_read(REC/'status.json')['updated'];age=(now()-datetime.fromisoformat(stamp)).total_seconds()
                assert age<=600,'600s without original frame/stage progress'
                bucket=int(age//120)
                if 1<=bucket<=4 and bucket not in captured and (REC/'actual_child.json').exists():
                    target=read(REC/'actual_child.json')
                    if (target['mode'],target['arm'],target['video'])==(mode,arm,video) and ps.creation(target['pid'])==target['birth']:
                        path=REC/'stacks'/f'{marker.stem}_{bucket}.txt';path.parent.mkdir(parents=True,exist_ok=True)
                        try:
                            dump=subprocess.run([str(ROOT/'.venv/Scripts/py-spy.exe'),'dump','--native','--full-filenames','--pid',str(target['pid'])],capture_output=True,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
                            path.write_bytes(dump.stdout+dump.stderr);write(path.with_suffix('.json'),dict(at=now().isoformat(),target=target,stale_seconds=age,exit_code=dump.returncode,sha256=sha(path)))
                        except subprocess.TimeoutExpired:write(path.with_suffix('.json'),dict(at=now().isoformat(),target=target,stale_seconds=age,stack_capture_timeout=True))
                        captured.add(bucket)
                time.sleep(.5)
        except BaseException:
            if proc.poll() is None and ps.creation(proc.pid)==birth:
                result=subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                write(REC/'child_cleanup.json',dict(at=now().isoformat(),pid=proc.pid,birth=birth,exit_code=result.returncode));proc.wait(timeout=20)
            raise
        assert proc.returncode==0,(mode,arm,video,proc.returncode)
    item=read(marker);item['wall_seconds_with_imports']=time.monotonic()-began;write(marker,item);return item


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    from scripts.z13_common_authorized_recovery1_20261024 import gpu_state
    R=bind();p=read(REC/'protocol.json');verify_sources(p);stable=None
    estimate=p['service_recovery2']['conservative_remaining_seconds']
    while True:
        R.S.check();assert (datetime.fromisoformat(p['save_at'])-now()).total_seconds()>estimate,'Original remaining window no longer fits conservative workload'
        resources=gpu_state();ready=not resources['active_game_identities'] and resources['free_gpu_mib']>=8192 and resources['gpu_utilization_percent']<=30
        if ready:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=30:break
        else:stable=None
        R.S.status('Проверка свободной GPU перед остатком',resources=resources);time.sleep(15)
    for video in p['videos']:
        for arm in R.FLY:R.S.status('Только оставшиеся варианты;181результат сохранён',variant=arm+' '+video);child('signature',arm,video)
        R.S.status('Общая MASA:оставшиеся варианты',variant=video);child('masa','all',video)
    R.evaluate();write(REC/'status.json',dict(state='completed',stage='10систем×20val,raw/MASAготовы',jobs_done=231,jobs_total=231,updated=now().isoformat()))


def monitor():
    import msvcrt
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    ps=WindowsProcesses();lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    e=read(REC/'controller_result.json')
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:
        write(W/'val20_common_authorized_recovery2_failure.json',dict(at=now().isoformat(),result=e,status=read(REC/'status.json')));return
    execute('record_val20_common_authorized_recovery2',[str(Path(__file__)),'--record']);verify_and_graph('val20_common_authorized_recovery2')
    write(W/'val20_common_authorized_recovery2_complete.json',dict(at=now().isoformat(),publication_pending=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as error:write(REC/'status.json',dict(state='failed',stage=repr(error),updated=now().isoformat()));raise
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as error:write(W/'val20_common_authorized_recovery2_monitor_failure.json',dict(at=now().isoformat(),error=repr(error)));raise
    elif '--record' in sys.argv:bind().record()
    elif '--job' in sys.argv:
        R=bind();mode,arm,video=sys.argv[2:5];p=read(REC/'protocol.json');ps=WindowsProcesses()
        assert video in p['videos'] and ((mode=='signature' and arm in R.FLY) or (mode=='masa' and arm=='all'))
        write(REC/'actual_child.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),mode=mode,arm=arm,video=video))
        result=R.S.build_signature(arm,video,False) if mode=='signature' else R.build_missing_masa(video)
        paths=[R.S.RUN/arm/'signatures'/video/'signatures.npz'] if mode=='signature' else [R.S.RUN/a/'masa'/(video+'.npz') for a in R.S.SYSTEMS]
        write(REC/'jobs'/f'{mode}_{arm}_{video}.json',dict(at=now().isoformat(),result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths}))
    else:raise RuntimeError('Explicit recovery2 mode required')
