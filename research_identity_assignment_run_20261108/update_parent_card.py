"""Publish the measured parent result in its existing experiment card."""
from research_identity_assignment_run_20261108.common import *


def main():
    path=ROOT/'experiments/project_history.json';catalog=read(path)
    card=next(c for c in catalog['experiments'] if c['id']=='identity_assignment_v2_run_20261108')
    report=read(RUN/'full_precision/comparison_report.json');assert not report['accepted_geometry'] and not report['accepted_flyvis_system_and_increment']
    assert card['state']=='running' and not card['obtained']
    card.update(state='completed_negative',done='2026-11-08',
        obtained='Нулевой полный выход на1204кадрах и1585восстановленных состояний совпали точно (runs/identity_assignment_20261108/zero_parity.json; restored_zero_parity.json). На641историческом запросе: B0 41правильный/6чужих/594отказа, G63/17/561, G+FlyVis60/15/566 (comparison_report.json). На20val HOTA B0=57,900829, G=58,021311, G+FlyVis=58,054615; FlyVis−B0 +0,153786п.п., ДИ99,375%[−0,019182;+0,361872], IDSw+7, FP+56, FN−61 (runs/identity_assignment_20261108/full_precision/comparison_report.json: pooled; comparisons.G_FLYVIS-B0). Известные чужие возвраты: B0=43, G=70, G+FlyVis=67 (full_precision/additional_report.json: arms.*.totals.wrong_returns).',
        interpretation='Условный выигрыш на запросах не перенёсся в принятую полезность полного трекера: исправленный интервал включает ноль, ограничения вреда нарушены. Добавка FlyVis над геометрией не подтверждена. Значительная часть возвратов не имеет доступной метки личности и учитывается как неизвестная.',
        decision='Эту общую правку не принимать в основной стенд. Продолжить только зарегистрированную ограниченную правку первой проблемы singleton_free_cost_v1_20261108: фиксировать штатные первые пары и работать со свободным остатком. Проблемы2–4отложены.')
    proofs=['runs/identity_assignment_20261108/zero_parity.json','runs/identity_assignment_20261108/export_complete.json',
        'runs/identity_assignment_20261108/selection.json','runs/identity_assignment_20261108/full_precision/comparison_report.json']
    for n,a in enumerate(card['acceptance']):a.update(met=n!=3,evidence=proofs[n])
    for name,claim in [('comparison_report.json','Исторические условные назначения'),('independent_audit.json','Независимые матрицы, метки и интервалы условной проверки'),
        ('full_precision/comparison_report.json','Полный20val-пилот не прошёл критерии'),('full_precision/additional_report.json','Чужие, правильные, неизвестные и устойчивые возвраты'),
        ('parent_full_result_review.json','Краткая сверка завершения и источников')]:
        card['evidence'].append(dict(kind='file',ref='runs/identity_assignment_20261108/'+name,claim=claim))
    newer=next(c for c in catalog['experiments'] if c['id']=='singleton_free_cost_v1_20261108');assert newer['state']=='planned';newer['state']='running'
    write(path,catalog)


if __name__=='__main__':main()
