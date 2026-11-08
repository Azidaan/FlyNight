"""Bind already completed data and outputs before recovering their summary."""
import sys,time,subprocess,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_singleton_verify_20261108.common import *


def main():
    import psutil
    phase('comparison_phase.json');assert read(RUN/'children/comparison.ended.json')['all_owned_ended']
    for name in ['queue_started.json','comparison_wait_started.json','audit_wait_started.json']:
        r=read(RUN/name)
        if psutil.pid_exists(r['pid']):assert abs(psutil.Process(r['pid']).create_time()-r['created'])>=.01
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')}
    for n in sources:ast.parse((ROOT/n).read_text(encoding='utf-8-sig'))
    for path in [RUN/'selection.json',RUN/'calibration_rows.json',RUN/'scores_complete.json',RUN/'features_complete.json',*sorted((RUN/'outcomes').glob('*.json'))]:sources[path.relative_to(ROOT).as_posix()]=sha(path)
    assert len(list((RUN/'outcomes').glob('*.json')))==20
    failure=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/0c6297b1c8e44dddbd9cc5ed8c522999/memory_guard.worker.log'
    assert "KeyError: 'N'" in failure.read_text(encoding='utf-8') and '4 passed' in failure.read_text(encoding='utf-8')
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Для `singleton_own_template_v1_20261108` сохранены все589наборов:193готовыхнабора,134прежнихтекущихкэша и262новыепричинныепоследовательности (`runs/singleton_verify_20261108/features_complete.json`). Новая проверка наGPUпрошла независимый размер пакета, максимальное нормированное расхождение0,0000002253 (`first_new_feature_check.json`). Четыре теста операции прошли; пороги заморожены наfit20по73известнымпримерам(50своих/23чужих),83неизвестных исключены из настройки и учтены. Все20таблицисходов готовы. Сборка интервалов остановилась из-за отсутствующего поляNввидео без известных первичных меток. Красный лог сохранён (`runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/0c6297b1c8e44dddbd9cc5ed8c522999/memory_guard.worker.log`). Версия `research_singleton_verify_recovery1_20261108/aggregate.py` используетN=0длятакоговидео и читает только готовые исходы. Карты, настройка и обновления трекера не повторяются; условия опыта не меняются.\n\n"
    write(RUN/'summary_recovery1_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(RUN/'summary_recovery1_phase.json',dict(registered=datetime.now(MSK).isoformat(),sources=sources,scientific_change=False,
        failure_sha256=sha(failure),repair='Read completed outcomes; missingNmeans0known examples for that video. No new selection or scientific replay.'),new=True)
    with (RUN/'summary_recovery1.log').open('x',encoding='utf-8') as out,(RUN/'summary_recovery1.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_singleton_verify_recovery1_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    r=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_singleton_verify_recovery1_20261108/controller.py')
    write(RUN/'summary_recovery1_started.json',r,new=True);roots=read(BASE/'deadline_roots.json');roots['roots'].append(r)
    for attempt in range(5):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==4:raise
            time.sleep(.1)
    write(RUN/'summary_recovery1_deadline_registered.json',dict(passed=True,root=r,deadline='2026-11-08T14:00:00+03:00'),new=True)


if __name__=='__main__':main()
