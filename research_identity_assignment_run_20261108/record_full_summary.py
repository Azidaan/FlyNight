"""Append the completed parent pilot result without altering its frozen records."""
from research_identity_assignment_run_20261108.common import *


def main():
    full=RUN/'full_precision';completed=read(full/'analyses_complete.json');assert completed['passed']
    assert sha(full/'comparison_report.json')==completed['statistics_sha256']
    assert sha(full/'additional_report.json')==completed['additional_sha256']
    report=read(full/'comparison_report.json');additional=read(full/'additional_report.json')
    assert not report['accepted_geometry'] and not report['accepted_flyvis_system_and_increment']
    record=dict(recorded=datetime.now(MSK).isoformat(),source_sha256=sha(full/'comparison_report.json'),
        additional_sha256=sha(full/'additional_report.json'),result='Registered full acceptance failed; do not deploy.',
        pooled=report['pooled'],comparisons=report['comparisons'],returns={a:r['totals'] for a,r in additional['arms'].items()})
    write(RUN/'parent_full_result_review.json',record,new=True)
    entry=f"**{datetime.now(MSK).strftime('%d.%m.%Y, %H:%M MSK')}** — Завершён полный исторический20val-пилот исходной общей правки `identity_assignment_v2_run_20261108`.\n\n*Получено.* ИсходныйHOTA57,900829; геометрия58,021311; геометрия+FlyVis58,054615. ПоправкаFlyVis−исходный вариант: +0,153786п.п., исправленныйДИ99,375%[−0,019182;+0,361872]; сменIDстало на7больше, ложных рамок на56больше, пропусков на61меньше (`runs/identity_assignment_20261108/full_precision/comparison_report.json`, `comparisons.G_FLYVIS-B0`). В дополнительном счёте по доступной разметке исходный вариант549правильных/43чужих возврата; геометрия573/70; геометрия+FlyVis572/67. ДляFlyVis270возвратов сохранили верную цель на двух следующих доступных наблюдениях против263исходных; неизвестные и цензурированные случаи учитываются отдельно (`full_precision/additional_report.json`).\n\n*Вывод.* Прежняя общая правка не прошла зарегистрированные критерииHOTAи вреда. Условная прибавка на запросах не доказала полезности полного трекера. Дополнительный вкладFlyVisнад геометрией также не подтверждён. Измерение не закрывает всё пространственное считывание и не решает первую проблему.\n\n*Решение.* Сохранить отрицательный итог и исходные критерии. Продолжать только уже зарегистрированную ограниченную правку первой проблемы: штатные первые пары фиксируются, новые возвраты разрешаются только на свободном остатке, с отдельным контролем ослабления порога и теми же сигнальными контролями (`runs/singleton_free_20261108/protocol.json`). Остальные три проблемы отложены. Полный исходный расчёт повторно не запускать. Первичные хеши и краткая сверка: `runs/identity_assignment_20261108/parent_full_result_review.json`; независимый аудит публикации отдельно.\n\n"
    write(RUN/'parent_full_result_journal.json',dict(entry=entry),new=True)
    plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=plan.read_text(encoding='utf-8');anchor='## Журнал изменений\n\n';assert text.count(anchor)==1
    plan.write_text(text.replace(anchor,anchor+entry,1),encoding='utf-8')


if __name__=='__main__':main()
