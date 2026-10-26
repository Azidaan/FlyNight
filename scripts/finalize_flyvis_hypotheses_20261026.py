"""Зафиксировать выполненную техническую подготовку, сохранив статус гипотез."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from register_flyvis_hypotheses_20261026 import DOC, IDS, ROOT, RUN, save


def main():
    now = datetime.now(timezone(timedelta(hours=3)))
    report = json.loads((RUN / 'technical_report.json').read_text(encoding='utf-8'))
    controller = json.loads((RUN / 'controller_result.json').read_text(encoding='utf-8-sig'))
    assert report['passed'] and controller['exit_code'] == 0 and controller['worker_ended'] and not controller['timed_out']
    path = ROOT / 'experiments/project_history.json'
    history = json.loads(path.read_text(encoding='utf-8'))
    for c in history['experiments']:
        if c['id'] in IDS:
            # Technical correction of these new, uncommitted references only.
            c['evidence'] = [dict(kind='file', ref=e, claim='Постановка до пилота') if isinstance(e, str) else e for e in c['evidence']]
        if c['id'] == IDS[0]:
            assert not c.get('done') and not c.get('obtained')
            c['done'] = ('Только техническая подготовка: прототип поправки и CPU-проба вычислительного пути через настоящую FlyVis. '
                         'Нулевой старт и отсутствие истории воспроизводят базу точно; градиент доходит в ядро после шага выходного слоя. '
                         'Источник: runs/flyvis_temporal_residual_20261026/technical_report.json. Научный пилот и HOTA не запускались.')
            c['files'] = ['tracking/flyvis_temporal_residual.py', 'scripts/flyvis_temporal_residual_preflight.py']
            c['evidence'].append(dict(kind='file', ref='runs/flyvis_temporal_residual_20261026/technical_report.json',
                                      claim='Работоспособность вычислительного пути, не научная приёмка'))
    save(path, json.dumps(history, ensure_ascii=False, indent=1) + '\n')
    doc_path = ROOT / DOC
    doc = doc_path.read_text(encoding='utf-8')
    assert '## 6. Выполненная техническая подготовка Б1' not in doc
    doc += (f'\n## 6. Выполненная техническая подготовка Б1 — {now:%d.%m.%Y, %H:%M} MSK\n\n'
            'Первая техническая проверка завершена. Прототип `tracking/flyvis_temporal_residual.py` сохраняет оценки базы при нулевой поправке и при отсутствии истории. '
            'Реальная FlyVis получила конечный ненулевой градиент во всех трёх группах свободных параметров после одного шага только выходного слоя; '
            'веса ядра не обновлялись. Нулевой градиент ядра на самом первом проходе ожидаем при нулевой выходной проекции; последующий проход подтвердил, что путь открывается.\n\n'
            f'Полная CPU-проба заняла {report["total_seconds"]:.2f} с; CUDA не инициализировалась. Источник: `runs/flyvis_temporal_residual_20261026/technical_report.json`, '
            '`total_seconds`, `zero_start_exact`, `no_history_exact`, `core_gradient_norms`, `core_weights_unchanged`; завершение подтверждено `controller_result.json`. '
            'Это один эпизод с реальным входом глаза и синтетическими масками кандидатов. Оценки сильной базы в пробе также синтетические. '
            'Работа с реальными областями и оценками MASA, перенос состояния внутри полного трекера, GPU-стоимость и HOTA ещё не проверены. '
            'Гипотеза остаётся запланированной, первый технический шаг выполнен; результат не включается в список проверенных научных гипотез.\n\n'
            'Следующее действие Б1 — соединить реальные области и наблюдаемую историю с оценками базы и воспроизвести её решения без поправки; затем замерить стоимость на GPU. '
            'Длительное обучение не запущено. HOTA не измерялась; процент к baseline и доля потолка не вычисляются.\n')
    save(doc_path, doc)
    plan_path = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md'
    plan = plan_path.read_text(encoding='utf-8')
    entry = (f'**{now:%d.%m.%Y, %H:%M} MSK** — Б1: технический путь поправки и градиента проверен, научный пилот не начат\n\n'
             '*Получено.* `runs/flyvis_temporal_residual_20261026/technical_report.json`: `passed=true`, `zero_start_exact=true`, `no_history_exact=true`, '
             f'`core_weights_unchanged=true`, полное время CPU {report["total_seconds"]:.2f} с (`total_seconds`), CUDA не инициализирована. '
             'После одного шага только выходного слоя градиент конечен и ненулевой во всех трёх группах параметров ядра (`core_gradient_norms`); '
             'ядро не обновлялось (`core_optimizer_steps=0`). `controller_result.json`: процесс завершён с кодом 0, без превышения бюджета.\n\n'
             '*Ограничение.* Реальный вход глаза, синтетические оценки базы и маски кандидатов. Это техническая проверка, не проверка узнавания областей. '
             'Б1–Б3 остаются непроверенными научными гипотезами. Следующий шаг — реальные области и воспроизведение базы, затем замер GPU-стоимости. HOTA не измерялась, процент к baseline не определён.\n\n'
             '*Техническая поправка записи.* У новых незакоммиченных карточек исправлен формат доказательств на объекты `kind/ref/claim`; первоначальная ошибка валидатора и прерванная сборка графа сохранены в `catalog_validation.log`. Научные источники и прежние карточки не исправлялись.\n\n')
    plan = plan.replace('## Журнал изменений\n', '## Журнал изменений\n\n' + entry, 1)
    save(plan_path, plan)
    roadmap = ROOT / 'FLYVIS_TRACKER_RESEARCH_ROADMAP.md'
    text = roadmap.read_text(encoding='utf-8')
    text += ('\n# 10. Дополнение 26.10.2026 — гипотезы Б1–Б3\n\n'
             'Прямое решение автора: начать Б1 «Сильная базовая ветвь и временная поправка мухи к сопоставлению объектов»; '
             'Б2 «Временная ветвь мухи перед обнаружением объектов» и Б3 «Использовать структуру связей мухи внутри новой памяти объектов» сохранить как альтернативы. '
             f'Полная постановка, литературные основания, контроли и критерии — `{DOC}`; карточки — в каталоге. '
             'Рассмотрение решений за прежней границей разрешено автором; выигрыш окружения без мухи отделяется от вклада FlyVis.\n\n'
             'Б1: выполнена только техническая проверка нулевого старта и обратного прохода в ядро; '
             'источник `runs/flyvis_temporal_residual_20261026/technical_report.json`. Научный статус всех трёх гипотез — план. '
             'Следующие зависимости Б1: реальные наблюдения и оценки сильной базы; точное воспроизведение базы; измеренная стоимость GPU; '
             'одна рабочая версия и HOTA полного трекера с согласованными контролями. Длительное обучение не запущено. '
             'Это дополнение не переписывает исторические результаты и не означает принятие ветви в стенд.\n')
    save(roadmap, text)
    print('Recorded technical preparation; all three scientific hypotheses remain planned.')


if __name__ == '__main__':
    main()
