"""Record already audited screening results while the full pilot is running."""
from research_singleton_age_20261108.common import *


def main():
    import psutil
    assert not (RUN/'comparison_phase.json').exists(),'Do not change catalog/chronology during its queued validation.'
    r=read(OWN/'comparison_report.json');audit=read(OWN/'independent_audit.json');assert audit['passed'] and audit['comparison_sha256']==sha(OWN/'comparison_report.json')
    assert read(OWN_FULL/'zero_parity.json')['passed'];assert not (OWN_FULL/'analyses_complete.json').exists()
    started=read(OWN_FULL/'manager_started.json');p=psutil.Process(started['pid']);assert abs(p.create_time()-started['created'])<.01 and p.is_running()
    path=ROOT/'experiments/project_history.json';catalog=read(path);c=next(v for v in catalog['experiments'] if v['id']=='singleton_own_template_v1_20261108');assert c['state']=='running'
    updates=dict(
        obtained='На 641 известном историческом запросе исходный трекер дал 41 правильное и 6 чужих назначений; проверка собственной карты FlyVis — 45 и 6. Простое изображение тоже дало 45 и 6, но один из четырёх добавленных правильных случаев у методов различается (runs/singleton_verify_20261108/comparison_report.json; outcomes/evaluate26_FLYVIS.json; outcomes/evaluate26_IMAGE.json). Условная разность J=правильные−чужие составила +4; 95% интервал прибавки на 100 запросов [0,142653;1,284404], интервал с поправкой на 11 сравнений включает ноль (comparison_report.json: comparisons.FLYVIS-B0). Независимая проверка 2156 оценок, 13937 строк исходов и 11088 матриц пройдена (independent_audit.json). Полный нулевой выход на 1209 кадрах совпал точно; полный пилот 20 видео продолжается (full/zero_parity.json; full/protocol.json).',
        interpretation='Проверка по собственной прошлой карте работает без второго трека-претендента и прошла предварительное условие перехода. Польза полного трекера ещё не установлена. Равные суммы с изображением не означают одинаковые случаи, а добавка именно FlyVis пока не подтверждена.',
        decision='Завершить уже начатый полный пилот и независимую сверку, затем выполнить зарегистрированный контроль по давности пропуска без изображения и FlyVis. Первую проблему пока не считать решённой; проблемы 2–4 остаются отложенными.')
    for field,value in updates.items():
        if c.get(field):c.setdefault('revisions',[]).append(dict(date='2026-11-08',commit='',field=field,old=c[field],reason='Промежуточный проверенный итог; полный пилот ещё выполняется.'))
        c[field]=value
    c['acceptance'][0].update(met=True,evidence='runs/singleton_verify_20261108/full/zero_parity.json')
    c['acceptance'][1].update(met=True,evidence='runs/singleton_verify_20261108/selection.json')
    for name,claim in [('comparison_report.json','Условный результат, не итог полного трекера'),('independent_audit.json','Независимая проверка условного результата'),('full/protocol.json','Полный пилот и критерии до его исходов'),('full/zero_parity.json','Точный нулевой полный выход')]:
        c['evidence'].append(dict(kind='file',ref='runs/singleton_verify_20261108/'+name,claim=claim))
    parent=next(v for v in catalog['experiments'] if v['id']=='identity_assignment_v2_run_20261108')
    parent['evidence'].append(dict(kind='file',ref='runs/identity_assignment_20261108/full_precision/independent_audit.json',claim='Отдельная сверка полного отрицательного результата без повторения трекера'))
    write(path,catalog)
    completed=len(list((OWN_FULL/'completed').glob('*.json')))
    entry=(f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Сохранён промежуточный статус первой проблемы. Независимый аудит условной проверки собственной карты пройден; её полный пилот продолжается, завершены {completed} из 20 видео (`runs/singleton_verify_20261108/full/completed/`). На 641 известном запросе FlyVis и изображение дали по 45 правильных и 6 чужих назначений против исходных 41 и 6; у трёх добавленных правильных случаев совпадают UID, ещё по одному различаются (`runs/singleton_verify_20261108/comparison_report.json`, `outcomes/evaluate26_FLYVIS.json`, `outcomes/evaluate26_IMAGE.json`). Вывода о полном качестве или дополнительной пользе FlyVis пока нет. Нулевой полный выход совпал на 1209 кадрах (`full/zero_parity.json`). Контроль давности без изображения и FlyVis зарегистрирован и поставлен после текущего полного расчёта и его аудита (`runs/singleton_age_20261108/protocol.json`, `started.json`); одновременный тяжёлый запуск не выполняется. Все условия и разрешённый срок 14:00 МСК сохраняются; проблемы 2–4 отложены.\n\n")
    write(RUN/'progress_journal.json',dict(entry=entry),new=True);plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')
    write(RUN/'progress_record.json',dict(passed=True,catalog_sha256=sha(path),plan_sha256=sha(plan),queued_catalog_check='After current full/audit; no mutation while validator/tests run.'),new=True)


if __name__=='__main__':main()
