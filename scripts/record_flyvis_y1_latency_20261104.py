"""Записать измеренный первый гейт Я1, сохранив научную гипотезу непроверенной."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.flyvis_y1_latency_20261104 import OUT, write, sha


def atomic(path, text):
    temporary = path.with_suffix('.y1_record.next')
    temporary.write_bytes(text.encode('utf-8'));temporary.replace(path)


def main():
    selected_chain = next((OUT / name / 'chain_result.json' for name in ['recovery2', 'recovery1']
                           if (OUT / name / 'chain_result.json').exists()), OUT / 'chain_result.json')
    chain = json.loads(selected_chain.read_text())
    audit = json.loads((OUT / 'independent_audit.json').read_text())
    analysis = json.loads((OUT / 'analysis.json').read_text())
    rows = json.loads((OUT / 'measurements.json').read_text())['rows']
    assert chain['complete'] and chain['all_owned_ended'] and audit['complete'] and audit['passed']
    owned = {int(pid): birth for pid, birth in chain['all_owned'].items()}
    controller = json.loads((OUT / 'controller_launch.json').read_text())
    owned[controller['pid']] = controller['created']
    for item in json.loads((OUT / 'agent_launch_verified.json').read_text())['launches']:
        owned[item['pid']] = item['created']
    if selected_chain.parent.name.startswith('recovery'):
        original = json.loads((OUT / 'chain_result.json').read_text())
        owned.update({int(pid): birth for pid, birth in original['all_owned'].items()})
        for folder in ['recovery1', 'recovery2']:
            if not (OUT / folder / 'controller_launch.json').exists():continue
            extra = json.loads((OUT / folder / 'controller_launch.json').read_text())
            owned[extra['pid']] = extra['created']
            for item in json.loads((OUT / folder / 'agent_launch_verified.json').read_text())['launches']:
                owned[item['pid']] = item['created']
            previous_chain = json.loads((OUT / folder / 'chain_result.json').read_text())
            owned.update({int(pid): birth for pid, birth in previous_chain['all_owned'].items()})
    ended = []
    for pid, birth in owned.items():
        try:assert abs(psutil.Process(pid).create_time() - birth) > .01, f'Own process still alive: {pid}'
        except psutil.NoSuchProcess:pass
        ended.append(dict(pid=pid, created=birth, original_process_ended=True))
    now = datetime.now(ZoneInfo('Europe/Moscow'))
    write('ended_verified.json', dict(all_owned_ended=True, checked_msk=now.isoformat(), processes=ended,
                                     selected_chain=selected_chain.relative_to(ROOT).as_posix()))
    p = analysis['primary']
    nominal = [r['epe_nominal_px'] for r in rows if r['method'] == 'hold' and r['aggregation'] == 'last']
    if p['necessary_gate_passed']:
        state = 'completed_positive'
        conclusion = (f"В первой проверке на четырёх исходных изображениях обнаружена лишняя задержка смены направления относительно DIS: "
                      f"{p['mean_excess_frames']:.6f} кадра,95%интервал[{p['ci95'][0]:.6f};{p['ci95'][1]:.6f}]. "
                      "Необходимый наблюдательный гейт пройден. Это ещё не доказательство причины в постоянных временных константах ядра.")
        decision = 'Подготовить отдельный согласуемый протокол минимального вмешательства в динамику и контроля постоянных констант; обучение и новая научная серия автоматически не запускаются.'
    elif p['identifiable']:
        state = 'completed_negative'
        conclusion = (f"Зарегистрированное условие лишней задержки не выполнено: средняя разница hold+last противDIS "
                      f"{p['mean_excess_frames']:.6f} кадра,95%интервал[{p['ci95'][0]:.6f};{p['ci95'][1]:.6f}]. "
                      'Это отрицательный итог данного диагностического критерия, не доказательство отсутствия задержек или бесполезности FlyVis вообще.')
        decision = 'Не продвигать адаптивные временные константы по этому гейту. Сохранить измерения и выбрать следующую обоснованную правку после отдельного решения; не подбирать пороги по полученной серии.'
    else:
        state = 'inconclusive'
        conclusion = (f"Для всех зарегистрированных входов задержку уверенно измерить не удалось: {p['valid_pairs']}/16пар "
                      'выполнили условия достаточного предшествующего отклика и наблюдаемого перехода. '
                      'Непрохождение необходимого гейта не означает отсутствия задержки: слабая амплитуда и правое цензурирование препятствуют такому выводу.')
        decision = 'Не запускать адаптивную динамику по неидентифицируемому измерению. Сохранить слабые/цензурированные случаи без отбора и определить дальнейшее действие по измеренному ограничению; новую серию не запускать по инерции.'
    limitation = ('Это диагностические искусственные глобальные сдвиги четырёх реальных TRAIN-фонов, не движение отдельных объектов, '
                  'не обобщение детектора и не HOTA. Веса/связи/динамика ядра и декодер не обучались. '
                  'Сравнение hold/linear и mean/last исключает некоторые эффекты подготовки и агрегирования, '
                  'но не отделяет причину в ядре от всего считывателя. Абсолютная ошибка номинальна по прежнему преобразованию декодера; '
                  'нормирование по собственному предшествующему плато служит только измерению перехода. '
                  'Полный CPU-повтор сети выполнен для одной заранее выбранной последовательности; для всех остальных независимо повторены входы/DIS/метрики, не нейронная симуляция. '
                  'Я1 об адаптивных константах остаётся непроверенной и красной; отрицательные прежние результаты не отменены.')
    obtained = (f"Первая серия завершена:64последовательности×2способа подачи×2агрегирования,5056исходныхкадров; "
                f"все входы/DIS/метрики/хеши проверены. Максимальная ошибка независимогоBoxEye {audit['eye_max_diff']:.10g}<1e-5; "
                f"одна полнаяCPU-сверка78пар кадров: {audit['cpu_reference']['max_abs_diff']:.10g}<=1e-5. "
                'Источники: runs/flyvis_y1_latency_20261104/independent_audit.json,measurements.json,analysis.json,ended_verified.json. '
                + conclusion)
    # Сохранение результатов не переписывает незавершённую гипотезу в завершённую.
    history_path = ROOT / 'experiments/project_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    card = next(c for c in history['experiments'] if c['id'] == 'flyvis_y1_latency_gate_v1')
    assert card['state'] == 'planned'
    for key in ['obtained', 'interpretation', 'decision', 'cannot_claim']:
        if card.get(key):card.setdefault('revisions', []).append(dict(date=now.isoformat(), commit='', field=key, old=card[key], reason='Сохранён полный фактический итог первой проверки'))
    card.update(state=state, done='2026-11-04', obtained=obtained, interpretation=conclusion,
                decision=decision, cannot_claim=limitation)
    card['acceptance'][0].update(met=True, evidence='runs/flyvis_y1_latency_20261104/independent_audit.json; ended_verified.json')
    card['acceptance'][1].update(met=bool(p['necessary_gate_passed']), evidence='runs/flyvis_y1_latency_20261104/analysis.json:primary')
    for name in ['independent_audit.json', 'analysis.json', 'measurements.json', 'ended_verified.json']:
        card['evidence'].append(dict(kind='json', ref='runs/flyvis_y1_latency_20261104/' + name, claim='Полный сохранённый диагностический результат с ограничениями'))
    atomic(history_path, json.dumps(history, ensure_ascii=False, indent=2) + '\n')
    registry_path = ROOT / 'experiments/flyvis_core_dynamics_hypothesis.json'
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    assert registry['status'] == 'planned'
    registry.setdefault('revisions', []).append(dict(date=now.isoformat(), field='authorization_scope', old=registry['authorization_scope'],
              reason='Новое прямое разрешение автора на первую проверку без обучения'))
    registry['authorization_scope'] = ('Первоначальное поручение02.11разрешало запись в граф. '
        '04.11автор ответил «Давай, продолжаем» на предложение выполнить первую проверку штатнойFlyVis без обучения; '
        'разрешение сохранено в runs/flyvis_y1_latency_20261104/authorization.json. '
        'Адаптивная динамика и обучение по-прежнему не реализованы и отдельно не разрешены.')
    registry['diagnostic_results'] = [dict(date='2026-11-04', title='Первая проверка штатной модели — измеренный результат',
        summary=conclusion + ' ' + limitation,
        source='runs/flyvis_y1_latency_20261104/independent_audit.json', sha256=sha(OUT / 'independent_audit.json'),
        analysis='runs/flyvis_y1_latency_20261104/analysis.json', card_id=card['id'])]
    atomic(registry_path, json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    problems_path = ROOT / 'experiments/system_problems.json'
    shutil.copy2(problems_path, OUT / 'before/system_problems.json')
    problems = json.loads(problems_path.read_text(encoding='utf-8'))
    problem = next(c for c in problems['problems'] if c['id'] == 'С1')
    problem['cases'].append(dict(date='2026-11-04', component='выход штатного модуля FlyVis на искусственном движении',
        fact=conclusion, hypothesis='Временная динамика ядра — рабочая версия, не установленная причина; вход и считыватель остаются альтернативами.',
        discriminating_check=decision, limitation=limitation,
        evidence=['runs/flyvis_y1_latency_20261104/analysis.json', 'runs/flyvis_y1_latency_20261104/independent_audit.json']))
    for name in ['analysis.json', 'independent_audit.json']:
        problem['evidence'].append(dict(source='runs/flyvis_y1_latency_20261104/' + name,
              sha256=sha(OUT / name), claim='Первая проверка переходного отклика; не доказательство пользы ядра для трекера'))
    problem['updated'] = '2026-11-04'
    if card['id'] not in problem['cards']:problem['cards'].append(card['id'])
    atomic(problems_path, json.dumps(problems, ensure_ascii=False, indent=2) + '\n')
    recovery_note = ('Первый GPU-процесс после восьми готовых последовательностей перестал обновлять прогресс на девятой; '
        'повторные стеки показали ожидание CUDAкопирования наCPU. Точная причина не установлена. '
        'Сохранены infer_stall_stack.log/infer_external_stop.json и исходныйcontrollercode15; '
        'после проверки полного окончанияPIDдерева подготовлено одно GPU-возобновление. '
        'Восемь готовых GPU-последовательностей переиспользованы поSHA, не повторялись. '
        'Первая CPU-сверка служебно отказала до сравнения из-за сохранённого CUDAустройства загрузчика; '
        'отдельная обёртка явно выбрала CPU в трёх модулях, исходные библиотеки и веса не менялись. '
        'После этой правки выполнена заранее запланированная полнаяCPU-сверка первой последовательности. '
        'Принятое продолжение: ' + selected_chain.parent.name + '. Старые контроллеры и журналы не переписаны. '
        if selected_chain.parent.name.startswith('recovery') else '')
    entry = f'''**{now:%d.%m.%Y, %H:%M} MSK** — Первая проверка Я1 завершена; обучение не выполнялось

*Постановка.* `flyvis_y1_latency_gate_v1`: известное равномерное движение, остановка и разворот реальных изображений при замороженнойFlyVis; полный проспективный протокол `runs/flyvis_y1_latency_20261104/protocol.json`.

*Получено.* {obtained} Контроллеры принятого продолжения завершилисьcode0;фактическиеPID+birthendedпроверены. {recovery_note}Первичные журналы и готовые последовательности сохранены. При регистрации прямое открытие каталога для записи вернулоOSError22; первоначальный каталог побайтно сохранился. Источник/отказ сохранены `runs/flyvis_y1_latency_20261104/registration_failure1/`; отдельнаяrecovery1записывает через временный файл и замену, научные условия до расчёта не изменены. Предварительные проверки:4unitpassed10,06с,каталогOK/58passed142,78с; первичныеpreflightлоги рядом с протоколом.

*Вывод.* {conclusion} {limitation}

*Решение.* {decision} Следующих заданий этой записью не публиковалось. Я1остаётсякраснойнепровереннойгипотезойизмененияядра. Сводка к исходному стенду: HOTAнеизмерялась,процентприростаидоляпотолканеизвестны.

'''
    plan = ROOT / 'dissertation/ЕДИНЫЙ_ПЛАН.md';old = plan.read_bytes();marker = '## Журнал изменений'.encode()
    ending = b'\r\n' if marker + b'\r\n' in old else b'\n'
    plan.write_bytes(old.replace(marker + ending, marker + ending + b'\n' + entry.encode('utf-8'), 1))
    roadmap = ROOT / 'FLYVIS_TRACKER_RESEARCH_ROADMAP.md'
    with roadmap.open('a', encoding='utf-8') as f:
        f.write(f'\n\n## 04.11.2026 — Первая проверка переходного отклика Я1\n\n{conclusion}\n\n{decision}\n\n'
                'Источник: `runs/flyvis_y1_latency_20261104/analysis.json`, `independent_audit.json`. '
                'Это отдельный диагностический этап ядра/модуляFlyVis, не новый итог трекера и не возобновление прежнихGPUсерий.\n')
    report = dict(complete=True, scientific_state=state, primary=p, summary=conclusion, decision=decision,
                  limitations=limitation, all_owned_ended=True, experiment_card=card['id'],
                  neural_core_modified=False, training=False, graph_hypothesis_status='planned',
                  independent_audit_sha256=sha(OUT / 'independent_audit.json'))
    write('report.json', report)
    text = '# Я1 — первая проверка штатной FlyVis\n\n' + conclusion + '\n\n' + obtained + '\n\n' + limitation + '\n\n' + decision
    text += '\n\nОсновные файлы: `protocol.json`, `measurements.json`, `analysis.json`, `independent_audit.json`, `ended_verified.json`.\n'
    atomic(OUT / 'РЕЗУЛЬТАТ.md', text)
    print(conclusion)


if __name__ == '__main__':main()
