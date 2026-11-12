import json
from pathlib import Path
from collections import Counter
OUT = Path(__file__).resolve().parent
result = json.loads((OUT / 'result.json').read_text(encoding='utf8'))
primary = [r for r in result['records'] if r['mode'] == 'complete_recovery']
outcomes = Counter(e['outcome'] for r in primary for e in r['evaluation'])
targets = [{key: r[key] for key in ['old_case', 'video', 'frame', 'target_tid', 'target_decision', 'target_evaluation_own', 'target_candidate_hits']} for r in primary]
controls = [{key: r[key] for key in ['old_case', 'mode', 'target_decision', 'removed_raw_ids']} for r in result['records'] if r['mode'] != 'complete_recovery']
summary = dict(variant='Registered primary tracked-or-lost states1/2', targets=targets, controls=controls, proposal_outcomes=dict(outcomes),
               criteria=dict(target574_correct_low_recovery=next(r['target_evaluation_own'] for r in primary if r['old_case'] == 574), no_foreign_or_unknown_proposal=not (outcomes['foreign'] or outcomes['unknown']),
                             native_assignments_unchanged=True, all5_absent_own_targets_no_continue=all(r['target_decision']['action'] != 'continue' for r in result['records'] if r['mode'] == 'remove_all_target_own'),
                             descriptor_permutation_no_proposals=all(not r['result']['proposals'] for r in result['records'] if r['mode'] == 'descriptor_permutation'),
                             missing_and_no_descriptors_defer=all(all(d['action'] == 'defer' for d in r['result']['decisions']) for r in result['records'] if r['mode'] in ['missing_competitors', 'no_descriptors'])),
               overall='completed_negative against strict allpool primary engineering criterion: one correct574recovery andthreeunknown extra proposals. Zero observed labeledforeign is not evidence all proposals are safe.',
               decision='Do not activate in full runtime or label as public-ID/HOTA fix. Retain true574 local recovery mechanism and explicit unknown cases. Lost-only secondary remains separately registered, with no gate tuning.',
               no_global_veto='Refuse means no excluded-low recovery proposal; it never deletes a track or forbids Census/native matching.',
               baseline_percent=dict(target574_correct_optional_proposals_before=0, target574_correct_optional_proposals_after=1, relative_percent=None, reason='Baseline0 prevents a finite relative percentage; full ID lifecycle/HOTA not evaluated.', ownless_target_false_continues=0, ownless_target_count=5, unknown_share_of_new_proposals_percent=75.0))
with (OUT / 'interpretation_primary.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(summary, ensure_ascii=False, indent=2).rstrip() + '\n')
text = '''1. Локальный результат и общий критерий

Зарегистрированная дополнительная ветвь нашла собственное продолжение строки574: raw35280 с уверенностью детектора0,07089663296937943. Косинусная близость MASA составляет0,9664591097828911, стоимость внешнего вида0,016770445108554466, пересечение с текущей предсказанной рамкой0,7469874622175492. Сопоставление строго взаимно ближайшее в полном пуле20строк этого случая. Рамка35292 стала её псевдонимом по заранее заданному подавлению перекрывающихся рамок с порогом0,7; эталонная идентичность в выборе не использовалась. Источник: result.json, records[mode=complete_recovery,old_case=574].

При удалении всех рамок, которые в оценочной разметке попадают в собственную идентичность574, ветвь отказывается от нового восстановления. При циклической перестановке описателей она тоже отказывается. При отсутствии описателей или полного конкурентного пула она откладывает решение. Значит, найденное продолжение не является простым выбором первого кандидата независимо от присутствия нужного объекта.

Общий строгий критерий не выполнен. По всем пяти полным пулам получено четыре новых предложения: одно однозначно собственное для574 и три с неопределённой правильностью. Это483→54963,55→4698 с псевдонимом4699 и563→35295 с псевдонимом35297. Для их прошлых образцов нет точного соответствия эталонному кадру в сохранённых часах; для двух текущих кандидатов нет попадания в эталонную рамку. Они сохранены как unknown (неопределённые), а не объявлены правильными. Ноль явно размеченных чужих предложений не устраняет эту неопределённость. Источник: result.json, records[*].evaluation.

2. Контроли исходных назначений

Строка70 отложена, поскольку уже имеет исходное назначение. Строка232 отложена, поскольку лучший кандидат принадлежит замороженному исходному назначению другого идентификатора. Для832 и1102 нет подтверждённого исключённого кандидата с низкой уверенностью. Такой отказ относится только к дополнительной ветви восстановления: он не запрещает исходное сопоставление или ветвь Census (локальное преобразование яркости), не удаляет траекторию и не меняет её идентификатор. Все исходные публичные назначения сохранены.

Все пять контролей с удалением собственных кандидатов не выдали новых продолжений целевым строкам. Циклическая перестановка описателей не выдала новых предложений во всех пяти случаях. Контроли отсутствия описателей и конкурентов отложили все решения. Источник: result.json, records соответствующих режимов.

3. Проверки и решение

Подготовлен действующий чистый модуль appearance_recovery_policy.py. Он получает только наблюдаемые рамки, уверенности, состояния, описатели и замороженные исходные назначения. Порог внешнего вида и геометрии заимствован без подбора из существующего BoT-SORT (трекера по движению и внешности); его перенос на MASA проверялся как инженерная гипотеза, а не как готовая вероятностная калибровка.

Независимый пересчёт подтвердил642решения и183проверки защиты исходных строк. Максимальное расхождение косинусной близости5,551115123125783e−16, пересечения рамок0. Пять целевых прошлых описателей сохранены точно; все222исходные близости воспроизведены с ошибкой не более3,3306690738754696e−16. Источники: audit.json и result.json, baseline.

Версия не принимается для включения в полный исполняемый трекер: строгий критерий отсутствия чужих и неопределённых новых предложений провален. Локальный результат574 сохранён отдельно от неудачи общего критерия. Вторая версия области применения, только для потерянных траекторий, зарегистрирована отдельно и не подменяет этот исход. Пороговые значения после результата не меняются.

Число правильных дополнительных предложений для574:0→1. Относительный процент к нулевой базе не определён; доля неопределённых среди четырёх новых предложений75%. В пяти контролях без собственных кандидатов ложных новых продолжений целевым строкам0из5. HOTA (метрика качества обнаружения и сопоставления траекторий), сохранность публичного идентификатора и полный жизненный цикл не измерены; исходное чужое предложение574 уже блокировалось глобальным сопоставлением.
'''
# Readable spacing for adjacent numerical facts in the drafted Russian report.
import re
text = re.sub(r'(?<=[А-Яа-я])(?=[0-9])|(?<=[0-9])(?=[А-Яа-я])', ' ', text)
with (OUT / 'РЕЗУЛЬТАТ_ОСНОВНОЙ.md').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(text.rstrip() + '\n')
