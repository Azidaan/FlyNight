"""Preserve the failure, test actual Windows sharing, register recovery before work."""
import sys,ast,time,subprocess,ctypes,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_full_recovery1_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import bounded_write


def verify_status_retry():
    from ctypes import wintypes
    path=RECOVERY/'status_sharing_probe.json';write(path,dict(old=True),new=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.CreateFileW(str(path.resolve()),0x80000000,3,None,3,0x80,None);assert handle!=wintypes.HANDLE(-1).value,ctypes.get_last_error()
    value=dict(stage='test',updated='fixed_source_time',payload=42);failed=False;timer=None
    try:
        try:write(path,value)
        except PermissionError:failed=True
        assert failed,'The controlled no-delete-share handle must reproduce Windows replacement failure.'
        timer=threading.Timer(.2,lambda:kernel.CloseHandle(handle));timer.start();attempts=bounded_write(path,value);timer.join();assert attempts>1 and read(path)==value
    finally:
        if timer is None:kernel.CloseHandle(handle)
        elif timer.is_alive():timer.join()
    result=dict(passed=True,actual_windows_sharing_failure_reproduced=True,retry_attempts=attempts,value_and_original_timestamp_preserved=True)
    write(RECOVERY/'status_retry_test.json',result,new=True)


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK);assert not (RECOVERY/'started.json').exists()
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    phase('full_phase.json');ended=read(FULL/'children/cpu_b1ca2e5d-84cf9134.ended.json');assert ended['returncode']==1 and ended['all_owned_ended'] and ended['guardian']['reason'] is None
    identities={int(k):v for k,v in ended['guardian']['owned'].items()}
    for path in [FULL/'manager_started.json',RUN/'full_audit_wait_started.json',ROOT/'runs/singleton_age_20261108/started.json',ROOT/'runs/singleton_error_audit_20261108/started.json']:
        r=read(path);identities[r['pid']]=r['created']
    for pid,birth in identities.items():
        try:p=psutil.Process(pid);assert abs(p.create_time()-birth)>=.01 or not p.is_running(),('Original process still live',pid)
        except psutil.NoSuchProcess:pass
    completed=list((FULL/'completed').glob('*.json'));assert len(completed)==15;ready={}
    for path in completed:
        for name,h in read(path)['artifacts'].items():assert sha(ROOT/name)==h
        ready[path.relative_to(ROOT).as_posix()]=sha(path)
    video='b1ca2e5d-84cf9134';receipts={}
    for arm in ['GEO','THRESHOLD','SUPPORT','RANDOM','IMAGE','R1']:
        path=FULL/'rollouts'/arm/(video+'.receipt.json');r=read(path);assert r['passed']
        for name,h in {**r['artifacts'],**r['feature_files']}.items():assert sha(ROOT/name)==h
        receipts[arm]=sha(path)
    assert not (FULL/'rollouts/DIS'/(video+'.receipt.json')).exists() and not (FULL/'ordinary_motion'/(video+'.npz')).exists()
    assert not (RUN/'full_audit_phase.json').exists() and not (ROOT/'runs/singleton_age_20261108/comparison_phase.json').exists()
    failure_paths=[FULL/'manager.err',RUN/'full_audit_wait.err',ROOT/'runs/singleton_age_20261108/controller.err',ROOT/'runs/singleton_error_audit_20261108/controller.err',
        ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/c5ef78ef65f44146b0b6f57a614d6676/memory_guard.worker.log',
        ROOT/'runs/singleton_free_20261108/status.json.pending',FULL/'children/cpu_b1ca2e5d-84cf9134.ended.json']
    write(RECOVERY/'failure_receipt.json',dict(original_error='PermissionError: atomic service status replacement at DIS progress update.',all_original_owned_ended=True,identities=identities,
        preserved_failures={p.relative_to(ROOT).as_posix():sha(p) for p in failure_paths},completed_video_manifests=ready,ready_sixteenth_video_arms=receipts,
        unfinished='DISvectors were in RAM when progress writing failed. No completed DIScache/output exists for this video; only that incomplete stage and unstarted work are resumed.'),new=True)
    verify_status_retry()
    protocol=dict(registered=datetime.now(MSK).isoformat(),scientific_protocol_sha256=sha(FULL/'protocol.json'),scientific_sources_phase_sha256=sha(RUN/'full_phase.json'),
        changes='Bounded PermissionError retry for progress JSON only; route formerly shared old helper status to the current run. Scientific stage bodies, inputs, weights, dtype, scores, thresholds, solver, metrics, criteria and deadline unchanged.',
        continuation='Resume preserved manager over15sealed videos and6sealed arms of video16; execute its independent audit, then restore AGEand diagnosis sequentially. Original failed controllers/receipts/markers are retained, not relabeled as passed.',
        provenance='Original full_phase binds numeric sources; this additional phase binds the actual recovery entrypoints/status-only wrappers before execution. New root and all descendant workers are protected by the existing author deadline watch.',
        deadline='2026-11-08T14:00:00+03:00',original_deadline_markers='Historical markers retained. This recovery root must have its own verified deadline registration before any scientific child starts.')
    write(RECOVERY/'protocol.json',protocol,new=True)
    entry=(f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Технический отказ полного пилота первой проблемы на шестнадцатом видео: Windows запретила атомарную замену служебного `runs/singleton_free_20261108/status.json` при сообщении прогресса DIS. Это `PermissionError`, не исчерпание памяти и не научный отрицательный итог; минимум доступной памяти 14068731904 байта, причина контроллера памяти null (`runs/singleton_verify_20261108/full/children/cpu_b1ca2e5d-84cf9134.ended.json`). Прежний служебный писатель не имел ограниченного повтора такой ошибки. Все красные журналы и зависимые остановки сохранены, фактический конец прежних процессов подтверждён. Проверены хеши 15 готовых видео и шести готовых вариантов шестнадцатого; они не повторяются. Незаконченный DIS хранился только в памяти и не имеет готового кэша. Отдельная версия `research_singleton_verify_full_recovery1_20261108/` добавляет ограниченный повтор записи только прогресса и правильный адрес текущего статуса. На реальном Windows-дескрипторе воспроизведён запрет удаления, новая запись дождалась закрытия и сохранила исходное время сообщения (`full_recovery1/status_retry_test.json`). Численные функции, параметры, готовые результаты и срок 14:00 МСК не меняются; восстанавливается вся последовательная цепочка аудита, контроля давности и разбора ошибок. Паспорт `runs/singleton_verify_20261108/full_recovery1/protocol.json`, `failure_receipt.json`.\n\n")
    write(RECOVERY/'registration_journal.json',dict(entry=entry),new=True);plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    sources=read(RUN/'full_phase.json')['sources'].copy()
    for folder in [Path(__file__).parent,ROOT/'research_singleton_age_20261108',ROOT/'research_singleton_error_audit_20261108']:
        for path in folder.glob('*.py'):sources[path.relative_to(ROOT).as_posix()]=sha(path)
    for path in [RECOVERY/'protocol.json',RECOVERY/'failure_receipt.json',RECOVERY/'status_retry_test.json',ROOT/'research_singleton_verify_full_audit_20261108/audit.py',ROOT/'research_singleton_free_20261108/full_audit.py']:
        sources[path.relative_to(ROOT).as_posix()]=sha(path)
    write(RECOVERY/'phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,service_only_recovery=True),new=True)
    bounded_write(RECOVERY/'status.json',dict(stage='registered_before_first_child',updated=datetime.now(MSK).isoformat()))
    with (RECOVERY/'controller.log').open('x',encoding='utf-8') as out,(RECOVERY/'controller.err').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([sys.executable,'-u','research_singleton_verify_full_recovery1_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),script='research_singleton_verify_full_recovery1_20261108/controller.py');write(RECOVERY/'started.json',root,new=True)
    roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)
    write(RECOVERY/'deadline_registered.json',dict(passed=True,root=root,deadline=protocol['deadline']),new=True)


if __name__=='__main__':main()
