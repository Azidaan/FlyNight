"""Initialize progress before the first child; retain late completed MASA probe."""
from pathlib import Path
from datetime import datetime
import sys,time,inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_val20_native_masa_20261024 as N
from scripts.z13_candidate_reserve_20261021 import read,write,sha,now
OLD=N.STAGE;REC=N.RUN/'native_masa_recovery1'


def prepare():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_localization_labels_20261023 import verify_sources
    assert not REC.exists();p=read(OLD/'protocol.json');verify_sources(p);e=read(OLD/'controller_result.json');c=read(OLD/'controller.json');ps=WindowsProcesses();assert e['exit_code']==1 and ps.creation(c['pid'])!=c['birth']
    # Parent ended before generic controller discovered its just-spawned child.
    late={22492:134372918982859480,20492:134372918982949350,26836:134372918983115582}
    assert all(ps.creation(pid)!=b for pid,b in late.items());jobs=list((OLD/'jobs').glob('*.json'));assert len(jobs)==5
    for f in jobs:
        for n,h in read(f)['outputs'].items():assert sha(ROOT/n)==h
    remaining=int((datetime.fromisoformat(c['save_at'])-now()).total_seconds());assert remaining>600
    proof=dict(at=now().isoformat(),late_owned_processes={str(k):v for k,v in late.items()},all_late_owned_ended=True,probe_marker_sha256=sha(OLD/'jobs/probe_masa_all_b1c66a42-6f7d68ca.json'),explanation='Controller recorded only parent family before first child was discovered;reported all_owned_ended at08:04:58 was incomplete. Explicit process-parent/creation audit found the surviving MASA family;its valid result completed08:06:42,all listed identities now ended. Old result retained unchanged.')
    write(REC/'late_child_completion_audit.json',proof);p={**p,'created':now().isoformat(),'budget_seconds':remaining,'save_at':c['save_at'],'hard_stop':c['hard_stop'],'stop_request_path':(REC/'stop.request').relative_to(ROOT).as_posix(),'sha256':{**p['sha256'],'scripts/z13_val20_native_masa_recovery1_20261024.py':sha(Path(__file__))},'service_recovery':dict(previous_result_sha256=sha(OLD/'controller_result.json'),previous_protocol_sha256=sha(OLD/'protocol.json'),late_child_audit_sha256=sha(REC/'late_child_completion_audit.json'),verified_jobs=5,change='Write initial status before starting first child,prevent missing-file error in600s watchdog;persist explicit child PID+birth. No changes to signatures/MASA/scorer,data,precision,thresholds or probes. Retain valid late completed control;original36000s absolute deadline unchanged.')};write(REC/'protocol.json',p)
    for f in jobs:write(REC/'jobs'/f.name,read(f))
    write(REC/'status.json',dict(state='ready',stage='5проверенных контролей сохранены;восстановлен начальный status',updated=now().isoformat(),jobs_done=5,jobs_total=105))
    N.journal('исправлена инициализация наблюдения базиса MASA','До первого MASAchildнебылсозданstatus.json;600сслужебныйконтрольродителяпрочиталнесуществующийфайл иостановился08:04:58. Это моя ошибкаобёртки. ДочернийMASAпроцессуспелстартовать,номалыйинтервалнепопалвобходдереваконтроллера;поэтомустарыйall_owned_endedбылнеполон. Дополнительнопроверенасемья22492/20492/26836повремени создания;правильныйконтрольMASAсамзавершился08:06:42,теперьвсясемьязакончена. Результатиегохешипроверены,готовые5контролейнеповторяются. Отдельнаяrecovery1сразуинициализируетstatus,фиксируеткаждыйchildPID+birth,исходный36000сбюджет/абсолютныйконец18:05:13непродлены. Источники/веса/пороги/данные прежние;ошибки/логи/коды сохранены.')


def bind():
    N.STAGE=REC;N.__file__=str(Path(__file__));N.bind();S=N.S
    source=inspect.getsource(S.child);old='birth=ps.creation(proc.pid)';assert source.count(old)==1
    source=source.replace(old,old+";write(STAGE/'live_child.json',dict(pid=proc.pid,birth=birth,mode=mode,arm=arm,video=video))")
    ns=vars(S).copy();exec(compile(source,str(Path(__file__)),'exec'),ns);S.child=ns['child']


def monitor():
    from scripts.z13_hota_deadline_guard import WindowsProcesses
    from scripts.z13_weekend_driver_20261024 import execute,verify_and_graph
    import os,msvcrt
    ps=WindowsProcesses();lock=(REC/'completion.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);write(REC/'completion.json',dict(pid=os.getpid(),birth=ps.creation(os.getpid()),at=now().isoformat(),sha256=sha(Path(__file__))))
    while not (REC/'controller_result.json').exists():time.sleep(5)
    c=read(REC/'controller.json')
    while ps.creation(c['pid'])==c['birth']:time.sleep(.5)
    e=read(REC/'controller_result.json');W=ROOT/'runs/z13_weekend_20261024'
    if e['exit_code']!=0 or not e['all_owned_ended'] or e['remaining']:write(W/'val20_native_masa_recovery1_failure.json',dict(at=now().isoformat(),result=e,status=read(REC/'status.json')));return
    execute('record_val20_native_masa_recovery1',[str(Path(__file__)),'--record']);verify_and_graph('val20_native_masa_recovery1');write(W/'val20_native_masa_recovery1_complete.json',dict(at=now().isoformat(),publication_pending=True))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        bind();N.S.status('Возобновление:контроли готовы,только оставшиеся20видео')
        try:N.worker()
        except Exception as e:write(REC/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    elif '--job' in sys.argv:
        bind();mode,a,v=sys.argv[2:5];probe='--probe' in sys.argv;p=read(REC/'protocol.json');assert v in ([p['probe_video']] if probe else p['videos']) and a in (*N.S.ARMS,'all');result=N.S.build_signature(a,v,probe) if mode=='signature' else N.S.build_masa(v,probe);base=REC/'probe' if probe else N.RUN;paths=[base/a/'signatures'/v/'signatures.npz'] if mode=='signature' else [base/x/'masa'/(v+'.npz') for x in N.S.ARMS];write(REC/'jobs'/f'{"probe_" if probe else ""}{mode}_{a}_{v}.json',dict(at=now().isoformat(),result=result,outputs={q.relative_to(ROOT).as_posix():sha(q) for q in paths}))
    elif '--record' in sys.argv:bind();N.record()
    elif '--monitor' in sys.argv:
        try:monitor()
        except Exception as e:write(ROOT/'runs/z13_weekend_20261024/val20_native_masa_recovery1_monitor_failure.json',dict(at=now().isoformat(),error=repr(e)));raise
    else:raise RuntimeError('Explicit recovery mode required')
