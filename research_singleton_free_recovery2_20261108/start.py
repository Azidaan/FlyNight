"""Freeze and attach the minimal launch recovery without rerunning green checks."""
import sys,subprocess,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil
    phase();assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    receipt=read(RUN/'recovery1_started.json')
    if psutil.pid_exists(receipt['pid']):assert abs(psutil.Process(receipt['pid']).create_time()-receipt['created'])>=.01
    assert read(RUN/'children/technical.ended.json')['all_owned_ended'] and read(RUN/'catalog_check_recovery1.json')['passed']
    assert not (RUN/'technical.json').exists()
    log=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/9bc31e90c9d348c6839bf6987acd51db/memory_guard.worker.log'
    assert 'cannot import name' in log.read_text(encoding='utf-8') and 'queue.py' in log.read_text(encoding='utf-8')
    sources={}
    for directory in [ROOT/'research_singleton_free_20261108',Path(__file__).parent]:
        for path in directory.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — После восстановления зелёной проверки каталога(58тестов; `runs/singleton_free_20261108/catalog_check_recovery1.json`) технический этап остановился до выполнения тестов и нулевого прохода: локальный `queue.py` перекрыл стандартный модульPython при импортеTorch (`runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/9bc31e90c9d348c6839bf6987acd51db/memory_guard.worker.log`). Научные исходники не переписаны. Изолированная обёртка запуска `research_singleton_free_recovery2_20261108/entry.py` сохраняет её собственный путь при повторном запуске дочернего процесса контроллером памяти, загружает стандартныйqueueи исключает каталог с одноимённым файлом изпути поиска. Код опыта, входы, параметры, критерии и защиты прежние. Зелёный каталог повторно не пересчитывается; продолжаются только невыполненные технический и научные этапы.\n\n"
    write(RUN/'recovery2_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(RUN/'recovery2_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,scientific_change=False,
        old_failure_sha256=sha(log),catalog_reused_sha256=sha(RUN/'catalog_check_recovery1.json'),
        intervention='Isolated argv[0] and standard-library queue import; all original scientific code and guards unchanged.'),new=True)
    with (RUN/'recovery2.log').open('x',encoding='utf-8') as out,(RUN/'recovery2.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_free_recovery2_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_free_recovery2_20261108/controller.py')
    write(RUN/'recovery2_started.json',receipt,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(receipt)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'recovery2_deadline_registered.json',dict(passed=True,root=receipt,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
