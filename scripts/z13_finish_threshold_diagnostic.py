"""Preserve and record the bounded Z13 threshold replay result."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/z13_threshold_diagnostic_20261013'


def main():
    report = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
    ctl = json.loads((OUT / 'controller.json').read_text(encoding='utf-8'))
    end = json.loads((OUT / 'controller_result.json').read_text(encoding='utf-8'))
    elapsed = (datetime.fromisoformat(end['ended']) - datetime.fromisoformat(ctl['started'])).total_seconds()
    assert end['exit_code'] == 0 and elapsed <= 600
    assert set(report['arms']) == {'B', 'C', 'D'}
    assert all(r['tau'] is None and not r['rounding_changes_admission'] for r in report['arms'].values())
    report['elapsed_seconds'] = elapsed
    report['interpretation'] = {
        'rounding_defect_explains_no_threshold': False,
        'localization': 'Median normalized center error exceeds baseline on the same positive examples for all three models',
        'selection': 'Useful corrections exist (>50 in each model), but original presence confidence does not select a >=90% useful set with at least 50 interventions',
        'limits': 'Diagnostics on threshold train videos only, no bootstrap or HOTA. They isolate functional failures, not the architectural or data causes. Presence confidence and useful-correction confidence are different quantities.',
    }
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    synthetic = {'rounded_tau': None, 'exact_tau': .700049, 'successes': 50,
                 'interventions': 50, 'synthetic_only': True,
                 'setup': '50 successes at 0.700049; 7 failures at 0.700011; rounding merges both groups'}
    (OUT / 'synthetic_rounding_check.json').write_text(json.dumps(synthetic, indent=2) + '\n', encoding='utf-8')
    catalog_path = ROOT / 'experiments/project_history.json'
    data = json.loads(catalog_path.read_text(encoding='utf-8'))
    parent = next(e for e in data['experiments'] if e['id'] == 'fly_object_readout_z13_v1')
    card = next(e for e in data['experiments'] if e['id'] == 'z13_threshold_diagnostic_v1')
    source = 'runs/z13_threshold_diagnostic_20261013/report.json'
    obtained = f'{source}: B/C/D — точный τ отсутствует; округление не изменило допуск. На 5822 размеченных примерах (1786 положительных, 4036 отрицательных) максимальная доля успешных при ≥50 вмешательствах: B 4,93%, C 7,13%, D 13,89% (требуется 90%). Всего успешных оценок независимо от уверенности: 98/227/214. Медиана ошибки центра в долях большей стороны объекта на положительных примерах: B 0,781608, C 0,439317, D 0,433065; штатный прогноз 0,261313. Повтор завершён за {elapsed:.2f} с; dev и val не использованы.'
    interpretation = 'Округление — воспроизведённая ошибка полноты перебора, но на готовых B/C/D оно не объясняет отказ по исходному правилу. Функциональные ограничения: неточная локализация относительно штатного прогноза и недостаточный отбор полезных поправок по уверенности присутствия. У всех вариантов есть больше 50 удачных оценок, поэтому дело не только в отсутствии достаточного числа удачных случаев. Конкретную архитектурную или информационную причину эти измерения не устанавливают.'
    decision = 'Остановить эту реализацию без перебора порогов, переобучения и дорогой оценки отсутствующего действия на dev/MASA. Сохранить предсказания для следующей постановки модернизации считывания FlyVis. Исправление округления в будущей реализации допустимо как исправление измерения, но не спасает текущие модели.'
    cannot = 'Не доказаны HOTA, общее превосходство B/C/D, бесполезность FlyVis или объектного обучения. Цифры — диагностика на 26 train, без доверительных интервалов. Плохой центр и плохой отбор полезных поправок установлены как функциональные ограничения, но причинность конкретных узлов не доказана.'
    for target in (parent, card):
        values = {'obtained': obtained, 'interpretation': interpretation, 'decision': decision, 'cannot_claim': cannot}
        if target is parent:
            values['summary'] = 'Точный повтор подтвердил отсутствие рабочего τ; текущая объектная локализация и уверенность не дают безопасного вмешательства.'
        for field, value in values.items():
            old = target.get(field)
            if old and old != value:
                target.setdefault('revisions', []).append({'date': '2026-10-13', 'commit': 'f4c09b0', 'field': field, 'old': old, 'reason': 'Получен точный повтор выбора τ и разложение ошибок на 26 train; прежние формулировки сохранены'})
            target[field] = value
        target['state'] = 'completed_negative'; target['done'] = '2026-10-13'
        if not any(e.get('ref') == source for e in target['evidence']):
            target['evidence'].append({'kind': 'json', 'ref': source, 'locator': 'arms, interpretation', 'claim': 'Точный перебор не спас модели; ограничения локализации и отбора полезных поправок'})
    card['acceptance'][0].update(met=True, evidence=f'{source}; elapsed_seconds={elapsed:.2f}; B/C/D_predictions.json, B/C/D_exact_curve.json')
    card['acceptance'][1].update(met=False, evidence=f'{source}: все arms.*.rounding_changes_admission=false')
    catalog_path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    text = plan.read_text(encoding='utf-8')
    now = datetime.now().astimezone()
    entry = f'''**13.10.2026, {now:%H:%M} MSK** — З13: точный перебор не изменил итог; измерены ограничения локализации и уверенности

*Получено (`z13_threshold_diagnostic_v1`).* `{source}`: разрешённый повтор готовых B/C/D на прежних 26 train завершён за {elapsed:.2f} с, все предсказания и точные кривые сохранены. У B/C/D τ отсутствует и без округления; дефект округления не объясняет этот результат. Из 5822 размеченных примеров 1786 положительных, 4036 отрицательных; ещё 2504 неизвестных не входят в правило. Максимальная доля полезных поправок при минимум 50 вмешательствах: B 4,93%, C 7,13%, D 13,89%, вместо требуемых 90%. Всего успешных оценок независимо от уверенности 98/227/214. На положительных примерах медиана ошибки в долях большей стороны объекта 0,781608/0,439317/0,433065; штатный прогноз на тех же примерах 0,261313. Это диагностические числа на train, не оценка HOTA и не сравнение с доверительными интервалами.

*Вывод.* Есть два функциональных ограничения: центр в типичном положительном случае менее точен, чем штатный прогноз; уверенность присутствия не отбирает имеющиеся полезные поправки. Вероятность «объект где-то в области» (1−P[отказ]) и вероятность «поправка достаточно точна и лучше прогноза» — разные величины. Это устройство кода, а не доказательство, что другой считыватель обязательно поможет. Конкретная причина в представлении, пространственном считывании или данных этим опытом не установлена. Обучение D не устранило отказ по рабочему правилу; общего отсутствия пользы обучения ядра это не доказывает.

*Решение.* Возвращён отрицательный итог именно этой реализации; старые и промежуточные формулировки сохранены в revisions. Не ослаблять порог ради результата, не повторять обучение и не запускать dev/MASA без действия. Отложенные 75 val не использованы. Возможные следующие узлы — пространственное считывание и оценка качества поправки; новую реализацию сейчас не ставили. Литературная опора для будущего рассмотрения: Learning Localization-aware Target Confidence for Siamese Visual Tracking (arXiv:2204.14093) и Probabilistic Regression for Visual Tracking (arXiv:2003.12565); их перенос в FlyVis требует отдельной постановки. Сводка: HOTA не измерялась, процент улучшения стенда и доля оракула не оценены.

'''
    marker = '## Журнал изменений\n\n'
    if entry.splitlines()[0] not in text:
        plan.write_text(text.replace(marker, marker + entry, 1), encoding='utf-8')


if __name__ == '__main__':
    main()
