import sys,time,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_identity_assignment_run_20261108.common import *
from research_singleton_verify_full_recovery1_20261108.status_io import for_run
RUN=ROOT/'runs/singleton_native_confirm_20261108';BASE=ROOT/'runs/identity_assignment_20261108'


def main():
    import psutil
    assert datetime.now(MSK)<datetime(2026,11,8,13,55,tzinfo=MSK)
    for path in Path(__file__).parent.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    catalog=read(ROOT/'experiments/project_history.json');id='singleton_native_confirm_inventory_v1_20261108';assert not any(e['id']==id for e in catalog['experiments'])
    catalog['experiments'].append(dict(id=id,title='Возврат старого ID после штатного подтверждения рамки — доступность сохранённых случаев',area='flyvis',state='planned',started='2026-11-08',
        question='Есть ли в сохранённых fit20 состояниях новые подтверждённые фрагменты, которым можно причинно вернуть потерянную личность?',hypothesis='Сначала штатное подтверждение присутствия рамки, затем отдельная проверка старой личности может избежать добавленных ложных рамок.',expected='Получить нижнюю оценку доступных случаев; не объявлять пользу трекеру по одной инвентаризации.',
        acceptance=[dict(criterion='Только fit20; настоящая первая выдача ранее неподтверждённого трека; известные правильные, чужие и неизвестные случаи отдельно; готовые расчёты не повторяются.',met=None)],obtained='',interpretation='',decision='',cannot_claim='Прежняя офлайн-сшивка уже существовала. Здесь проверяется доступность причинного решения без будущих кадров. Нулевая доступность в архиве запросов не означает её отсутствия между запросами. Проблемы 2–4 не исследуются.',
        evidence=[dict(kind='file',ref='research_singleton_native_confirm_20261108/fit_inventory.py',claim='Определения инвентаризации до подсчёта')]))
    write(ROOT/'experiments/project_history.json',catalog)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');stamp=datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')
    entry=f'\n\n**{stamp}** — Первая проблема: зарегистрирована инвентаризация штатного подтверждения.\n\n*Постановка (`{id}`).* Только на fit20 проверить сохранённые запросы: единственная допустимая свободная рамка уже выдана штатным трекером при первом подтверждении нового фрагмента. Сначала присутствие, затем старая личность; будущие кадры и правка уже выданных меток запрещены. Отличать от прежней офлайн-сшивки. Это нижняя оценка доступности, не проверка качества (`research_singleton_native_confirm_20261108/fit_inventory.py`). Готовые траектории, детектор и ядро не запускаются. Затем последовательная проверка каталога; проблемы 2–4 остаются отложенными.\n'
    marker='## Журнал изменений';assert text.count(marker)==1;plan.write_text(text.replace(marker,marker+entry,1),encoding='utf-8',newline='\n')
    for_run(RUN)('fit_inventory_registered',current_problem=1,deferred=[2,3,4])
    with (RUN/'controller.log').open('x',encoding='utf-8') as out,(RUN/'controller.err').open('x',encoding='utf-8') as err:
        p=subprocess.Popen([sys.executable,'-u','research_singleton_native_confirm_20261108/controller.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.IDLE_PRIORITY_CLASS)
    root=dict(pid=p.pid,created=psutil.Process(p.pid).create_time(),script='research_singleton_native_confirm_20261108/controller.py');write(RUN/'started.json',root,new=True)
    roots=read(BASE/'deadline_roots.json');roots['roots'].append(root)
    for attempt in range(10):
        try:write(BASE/'deadline_roots.json',roots);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.1)
    write(RUN/'deadline_registered.json',dict(passed=True,root=root,deadline='2026-11-08T14:00:00+03:00'),new=True)
    print(root)


if __name__=='__main__':main()
