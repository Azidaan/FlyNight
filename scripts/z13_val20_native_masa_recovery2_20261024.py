"""Bounded status-read retry and explicit child cleanup; science unchanged."""
from pathlib import Path
from datetime import datetime
import sys,time,subprocess,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_native_masa_20261024 as N
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
from scripts.z13_hota_deadline_guard import WindowsProcesses
OLD=N.RUN/'native_masa_recovery1';REC=N.RUN/'native_masa_recovery2'


def status_read(path):
    for attempt in range(40):
        try:return read(path)
        except (PermissionError,FileNotFoundError):
            if attempt==39:raise
            time.sleep(.05)


def prepare():
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();p=read(OLD/'protocol.json');verify_sources(p);e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');ps=WindowsProcesses()
    assert e['exit_code']==1 and ps.creation(c['pid'])!=c['birth'];assert all(ps.creation(int(pid))!=b for pid,b in e['owned'].items()),'Old own child remains; no new GPU'
    jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==59
    for f in jobs:
        for n,h in read(f)['outputs'].items():assert sha(ROOT/n)==h
    price=read(OLD/'price.json');assert price['proceed'];remaining=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert remaining>600
    videos_remaining=sum(any(not (OLD/'jobs'/f'{mode}_{a}_{v}.json').exists() for mode,a in [('signature',x) for x in N.S.ARMS]+[('masa','all')]) for v in p['videos']);estimate=2*price['first_new_video_seconds']*videos_remaining+900;assert estimate<remaining
    proof=dict(at=now().isoformat(),original_result_all_owned_ended=e['all_owned_ended'],late_owned=e['remaining'],all_original_known_births_now_ended=True,verified_jobs=59,reason='Transient PermissionError while watchdog reads status. Last TrackTrack child continued and wrote verified marker;all PID+creation identities now ended. No scientific guard failed.')
    write(REC/'previous_end_audit.json',proof);p={**p,'created':now().isoformat(),'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],'scripts/z13_val20_native_masa_recovery2_20261024.py':sha(Path(__file__))},'service_recovery2':dict(previous_result_sha256=sha(OLD/'controller_result.json'),previous_protocol_sha256=sha(OLD/'protocol.json'),previous_end_audit_sha256=sha(REC/'previous_end_audit.json'),verified_jobs=59,change='Only transient PermissionError/FileNotFound status reads retry up to40x0.05s. Original updated timestamps/600s watchdog unchanged. Persist child PID+birth;on parent check failure terminate only that verified own child tree and wait. Original frames/models/math/thresholds unchanged. Completed jobs retained;original10hour window not extended. Price uses original measured530.25s per full video and only remaining10videos,including partialvideo conservatively.'),'recovery_price':dict(original_measured_first_video=price['first_new_video_seconds'],remaining_videos_including_partial=videos_remaining,conservative_remaining_seconds=estimate,available_seconds=remaining,proceed=True)}
    write(REC/'protocol.json',p)
    for f in jobs:write(REC/'jobs'/f.name,read(f))
    write(REC/'status.json',dict(state='ready',stage='59готовых заданий проверены;восстановлено чтение прогресса',updated=now().isoformat(),jobs_done=59,jobs_total=105))
    N.journal('восстановлено чтение прогресса общего базиса20val','PermissionErrorпри чтении native_masa_recovery1/status.json остановилродителя09:51:58. ПоследнийTrackTrackchildзавершилсяпозжеиоставилправильный59-ймаркер;все прежниеPID/birthпроверены иокончены. Вrecovery2 повторяетсятолькослужебноечтениеприPermissionError/FileNotFoundдо40разпо0,05с;прежняядатаupdatedи600ссторожнеизменны. ПриошибкепроверкиродителяегоявнозарегистрированныйдочернийпроцессзавершаетсяпосверкеPID/birth,ожидаетсяконецдерева. Научныефайлы/математика/данные/допускинеизменны. 59заданийпроверены,неповторяются. Ценаоставшихся'+str(videos_remaining)+'видеовключаячастичный='+str(estimate)+'с,остаток'+str(remaining)+'с;первыйфактический530,25ссохранён,нулевоевремяпропускаготовогонеподменяетцену. Исходныйконец18:05:13МСКнепродлён. FlyVisожидаетпрежнегорешенияавтора,допускнеменялся.')


def bind():
    N.STAGE=REC;N.__file__=str(Path(__file__));N.bind();N.S.child=child


def child(mode,arm,video,probe=False):
    S=N.S;marker=REC/'jobs'/f'{"probe_" if probe else ""}{mode}_{arm}_{video}.json'
    if marker.exists():
        m=read(marker)
        for n,h in m['outputs'].items():assert sha(ROOT/n)==h
        return m
    ps=WindowsProcesses();log=REC/'logs'/marker.stem;log.parent.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with log.with_suffix('.stdout.log').open('w',encoding='utf-8') as out,log.with_suffix('.stderr.log').open('w',encoding='utf-8') as err:
        proc=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(Path(__file__)),'--job',mode,arm,video,*(['--probe'] if probe else [])],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.IDLE_PRIORITY_CLASS|subprocess.CREATE_NO_WINDOW);birth=ps.creation(proc.pid);write(REC/'live_child.json',dict(pid=proc.pid,birth=birth,mode=mode,arm=arm,video=video))
        try:
            while proc.poll() is None:
                S.check();stamp=status_read(REC/'status.json')['updated']
                if (now()-datetime.fromisoformat(stamp)).total_seconds()>600:raise TimeoutError('600s without original timestamp progress')
                time.sleep(.5)
        except BaseException:
            if proc.poll() is None and ps.creation(proc.pid)==birth:
                result=subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW);write(REC/'child_cleanup.json',dict(at=now().isoformat(),pid=proc.pid,birth=birth,exit_code=result.returncode));proc.wait(timeout=20)
            raise
        assert proc.returncode==0,('Native GPU child failed',mode,arm,video,proc.returncode)
    m=read(marker);m['wall_seconds_with_imports']=time.monotonic()-began;write(marker,m);return m


def worker():
    from scripts.z13_localization_labels_20261023 import verify_sources
    p=read(REC/'protocol.json');verify_sources(p)
    for v in p['videos']:
        for a in N.S.ARMS:N.S.status('Только оставшиеся общие сигнатуры',variant=a+' '+v);child('signature',a,v)
        N.S.status('Только оставшаяся общая MASA',variant=v);child('masa','all',v)
    N.evaluate();write(REC/'status.json',dict(state='completed',stage='4трекера×20val,raw/MASAготовы;FlyVisнеиспользуется',updated=now().isoformat(),jobs_done=105,jobs_total=105))


def monitor():
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    import msvcrt
    ps=WindowsProcesses();lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    e=read(REC/'controller_result.json');W=ROOT/'runs/z13_weekend_20261024'
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:write(W/'val20_native_masa_recovery2_failure.json',dict(at=now().isoformat(),result=e,status=status_read(REC/'status.json')));return
    execute('record_val20_native_masa_recovery2',[str(Path(__file__)),'--record']);verify_and_graph('val20_native_masa_recovery2');write(W/'val20_native_masa_recovery2_complete.json',dict(at=now().isoformat(),publication_pending=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        bind();N.S.status('59готовых заданий сохранены;продолжение оставшегося')
        try:worker()
        except Exception as e:write(REC/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    elif '--job' in sys.argv:
        bind();mode,a,v=sys.argv[2:5];probe='--probe' in sys.argv;p=read(REC/'protocol.json');assert v in ([p['probe_video']] if probe else p['videos']) and a in (*N.S.ARMS,'all');result=N.S.build_signature(a,v,probe) if mode=='signature' else N.S.build_masa(v,probe);base=REC/'probe' if probe else N.RUN;paths=[base/a/'signatures'/v/'signatures.npz'] if mode=='signature' else [base/x/'masa'/(v+'.npz') for x in N.S.ARMS];write(REC/'jobs'/f'{"probe_" if probe else ""}{mode}_{a}_{v}.json',dict(at=now().isoformat(),result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths}))
    elif '--record' in sys.argv:bind();N.record()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:write(ROOT/'runs/z13_weekend_20261024/val20_native_masa_recovery2_monitor_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit recovery2 mode required')
