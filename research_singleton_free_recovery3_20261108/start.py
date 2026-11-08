"""Preserve the invalid comparator and bind its format-only correction."""
import sys,subprocess,time,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_free_20261108.common import *


def main():
    import psutil,numpy as np
    phase();assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    r=read(RUN/'recovery2_started.json')
    if psutil.pid_exists(r['pid']):assert abs(psutil.Process(r['pid']).create_time()-r['created'])>=.01
    assert read(RUN/'children/technical_recovery2.ended.json')['all_owned_ended'];assert not (RUN/'technical.json').exists()
    log=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/e308709bd5a9460eb49e0aca77845ab5/memory_guard.worker.log'
    assert '5 passed' in log.read_text(encoding='utf-8') and "expected['outputs']" in log.read_text(encoding='utf-8')
    with np.load(BASE/'zero_native_outputs.npz',allow_pickle=False) as z:shape=list(z['outputs'].shape);count=len(z['offsets'])-1;first=z['outputs'][0].tolist()
    assert shape[1]==8 and first[0]>1 and first[-1]==0
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')}
    for n in ['research_singleton_free_recovery2_20261108/bootstrap.py','research_singleton_free_recovery2_20261108/entry.py']:
        sources[n]=sha(ROOT/n)
    for n in sources:ast.parse((ROOT/n).read_text(encoding='utf-8-sig'))
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Изолированный запуск устранил конфликтqueue;5тестов новой операции прошли. Затем обнаружен неверный формат сравнения нулевых выходов: архив `zero_native_outputs.npz` хранит8штатных колонок(последняя — индекс детекции), а границы{count}кадров — вoffsets; первая проверка ошибочно сравнивала с массивом, где вместо индекса был добавлен номер кадра. Это ошибка проверяющего кода, не установленное расхождение трекера. Исходный красный лог сохранён (`runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/e308709bd5a9460eb49e0aca77845ab5/memory_guard.worker.log`). Версия `research_singleton_free_recovery3_20261108/technical.py` сравнивает полные штатные выходы по каждому кадру и сохраняет проверяемый массив до утверждения. Готовые58тестов каталога и5тестов операции переиспользуются; научная калибровка ещё не начиналась.\n\n"
    write(RUN/'recovery3_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(RUN/'recovery3_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,scientific_change=False,
        old_failure_sha256=sha(log),original_archive_sha256=sha(BASE/'zero_native_outputs.npz'),archive_shape=shape,frames=count,
        correction='Compare eight native columns and frame offsets, preserving the local detection index; no tracker or scientific change.'),new=True)
    with (RUN/'recovery3.log').open('x',encoding='utf-8') as out,(RUN/'recovery3.err').open('x',encoding='utf-8') as err:
        process=subprocess.Popen([sys.executable,'-u','research_singleton_free_recovery3_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    receipt=dict(pid=process.pid,created=psutil.Process(process.pid).create_time(),script='research_singleton_free_recovery3_20261108/controller.py')
    write(RUN/'recovery3_started.json',receipt,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(receipt)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'recovery3_deadline_registered.json',dict(passed=True,root=receipt,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
