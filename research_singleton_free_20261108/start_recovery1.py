"""Register the unchanged scientific continuation after the service-test race."""
import sys,subprocess,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    phase();assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    for name in ['queue_started.json','next_stage_started.json','full_audit_wait_started.json']:
        r=read(RUN/name)
        if psutil.pid_exists(r['pid']):assert abs(psutil.Process(r['pid']).create_time()-r['created'])>=.01,'Previous own controller is still alive.'
    old=read(RUN/'children/catalog.ended.json');assert old['returncode']==1 and old['all_owned_ended']
    assert not (RUN/'technical.json').exists() and not (RUN/'selection.json').exists()
    sources={}
    for path in (ROOT/'research_singleton_free_20261108').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    log=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/78220fcecc1446bbbcad8b1540a9c059/memory_guard.worker.log'
    assert '57 passed' in log.read_text(encoding='utf-8') and 'test_r16_14_cache_invalidated_by_fingerprint' in log.read_text(encoding='utf-8')
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Служебная проверка каталога остановила очередь первой проблемы до научной калибровки. Валидатор прошёл;57тестов прошли, один тест неизменности отпечатка кэша не прошёл, поскольку Codex дописал итог вЕдиный план в05:43:14во время теста (лог `runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/78220fcecc1446bbbcad8b1540a9c059/memory_guard.worker.log`; время `runs/identity_assignment_20261108/parent_full_result_review.json`). Замороженные научные исходники, тест и красный результат сохранены. Возобновление `runs/singleton_free_20261108/recovery1_phase.json` повторяет служебную проверку при неизменных входах и затем выполняет только ещё не выполненные этапы. Готовые карты и полные научные прогоны не повторяются.\n\n"
    write(RUN/'recovery1_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(RUN/'recovery1_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,
        failure_log_sha256=sha(log),old_end_sha256=sha(RUN/'children/catalog.ended.json'),
        scientific_change=False,reason='Concurrent journal write during the cache-fingerprint test. Preserve failure and rerun stable checks before unfinished scientific steps.',
        input_snapshot={n:sha(ROOT/n) for n in ['experiments/project_history.json','experiments/metrics_glossary.json','dissertation/ЕДИНЫЙ_ПЛАН.md']}),new=True)
    with (RUN/'recovery1.log').open('x',encoding='utf-8') as out,(RUN/'recovery1.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_free_20261108/resume_recovery1.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_free_20261108/resume_recovery1.py')
    write(RUN/'recovery1_started.json',receipt,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(receipt)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'recovery1_deadline_registered.json',dict(passed=True,root=receipt,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
