import json
from pathlib import Path
from collections import Counter
OUT = Path(__file__).resolve().parent

def read(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))

variants = []
for name, source in [('primary_tracked_or_lost', 'result.json'), ('secondary_lost_only', 'result_lost_only.json'), ('final_native_episode', 'result_episode.json')]:
    result = read(source)
    records = [r for r in result['records'] if r['mode'] == 'complete_recovery']
    counts = Counter(e['outcome'] for r in records for e in r['evaluation'])
    variants.append(dict(variant=name, source=source, proposals=result['proposals'], own=counts['own'], foreign=counts['foreign'], unknown=counts['unknown'],
                         whole_pool_engineering_gate=counts['foreign'] == 0 and counts['unknown'] == 0 and next(r['target_evaluation_own'] for r in records if r['old_case'] == 574)))
final = read('result_episode.json')
records = final['records']
primary = [r for r in records if r['mode'] == 'complete_recovery']
summary = dict(variants=variants, final_targets=[{key: r[key] for key in ['old_case', 'target_tid', 'eligible_tids', 'target_decision', 'target_evaluation_own', 'target_candidate_hits']} for r in primary],
               controls=[dict(mode=mode, target_continues=sum(r['target_decision']['action'] == 'continue' for r in records if r['mode'] == mode), all_pool_proposals=sum(len(r['result']['proposals']) for r in records if r['mode'] == mode)) for mode in ['disabled', 'descriptor_permutation', 'missing_competitors', 'no_descriptors', 'remove_all_target_own']],
               fixed=dict(complete_reference_rows=107, current_raw_candidates=222, appearance_thresh=0.8, appearance_cost_limit='1-0.8 from source', proximity_iou=0.5, nms_iou=0.7, native_high=0.25, native_low=0.1),
               final_scope='Only lost rows exposed by native observable pre-core recovery episode may emit; all107 rows remain nearest-neighbor competitors. Caller passes explicit eligible_tids from shared canonical helper. No GT anchor filter, no hardcoded track/raw ID values.',
               history='Primary and lost-only variants fail strict safety verification with3unknown proposals. Final observable-domain restriction is a late engineering response to that failure, separately registered and evaluated on the same5 cases. Do not present as independent heldout evidence.',
               decision='Final candidate passes this bounded5-case engineering gate and is available for parent experimental module/adapter integration. It is not enabled in the full tracker here. No further recovery rule/threshold variant is proposed.',
               cannot_claim=['HOTA or full tracker improvement', 'Public-ID continuity or lifecycle correction', 'General safety on new frames', 'Correctness of excluded broad unknown proposals', 'An advantage attributable to the FlyVis core'],
               baseline_percent=dict(correct_optional_target574_before=0, correct_optional_target574_after=1, finite_relative_percent=None, reason='Zero optional recovery baseline; full tracking ceiling not measured.', wrong_target_continues_without_own=0, absent_own_cases=5, final_unknown_new_proposals=0))
with (OUT / 'interpretation_final.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(summary, ensure_ascii=False, indent=2).rstrip() + '\n')
text = '''1. Действующая проверенная правка

Подготовлена отдельная ветвь подтверждения исключённых детекций с низкой уверенностью. Она не снижает исходные пороги детектора и не перехватывает занятые назначения. В проверенном эпизоде574 ветвь предлагает продолжить прежнюю строку по raw35280; raw35292 сохраняется как псевдоним той же группы перекрывающихся рамок. Детекторная уверенность35280 составляет0,07089663296937943, близость MASA0,9664591097828911, стоимость внешнего вида0,016770445108554466 и пересечение с предсказанной рамкой0,7469874622175492. Эталонная оценка связывает предложение с прежней идентичностью00115019. Источник: result_episode.json, proposals и records[mode=complete_recovery,old_case=574].

Правило требует одновременно абсолютных порогов внешнего вида и геометрии, строго взаимно ближайшего сопоставления в полном конкурентном пуле и отсутствия владельца у любого псевдонима кандидата. Пороги0,8для параметра внешнего вида,0,5для пересечения рамок и0,7для подавления дублирующих рамок взяты из существующих исходников без подбора. Параметр0,8в BoT-SORT (трекере по движению и внешности) означает стоимость (1−косинус)/2≤1−0,8; перенос этого правила на MASA остаётся инженерной гипотезой, а не вероятностной калибровкой. Исходные пороги0,25и0,1не изменены.

2. Область применения и конкуренция

В финальной версии новое предложение может выдать только потерянная строка, которая относится к уже существующему наблюдаемому эпизоду восстановления. Эти эпизоды определяются прежними проверками класса, истории, временного разрыва, двух региональных кандидатов с одним геометрически допустимым и достаточной пространственной опоры. Для сохранённых записей используется наличие геометрических признаков и оба исхода штатного считывателя, принятие и отказ. Значения активности, знак оценки и признак evidence_valid (допустимости нейронных признаков) не определяют допуск. Эталонные идентификаторы и наличие эталонного кадра также не используются.

Ограничивается только право выдать новое предложение. Все107прошлых строк остаются конкурентами при обратном поиске ближайшего образца; все222сырые текущие детекции остаются в исходном пуле до подавления дублирующих рамок. Ни известные идентификаторы случаев, ни эталонная правильность не исключают конкурентов. Чистый модуль принимает явный набор eligible_tids (идентификаторов строк, прошедших наблюдаемые условия) от вызывающей стороны. Источники: appearance_recovery_policy_v4.py, episode_addendum_v4.json и result_episode.json.

3. Все пять случаев и отрицательные контроли

В пяти полных пулах финальная версия выдала только одно новое предложение, для574. Однозначно своих предложений1, чужих0, неопределённых0. Строка232 отложена из-за занятого исходным назначением кандидата; строка70 сохраняет своё исходное назначение. Для832 и1102 дополнительная ветвь не нашла допустимого исключённого кандидата с низкой уверенностью. Последнее не является запретом их исходного сопоставления или другой ветви системы: модуль только воздерживается от добавления нового восстановления.

Во всех пяти контролях с удалением всех собственных текущих кандидатов новых предложений нет. При циклической перестановке описателей новых предложений тоже нет. При отсутствии описателей или полного конкурентного пула решения откладываются. Все исходные публичные назначения сохранены, включая назначения позднейших нативных стадий и новых текущих идентификаторов. Источник: result_episode.json, records соответствующих режимов.

4. Сохранённые неудачи

Первая зарегистрированная широкая версия и отдельная версия только для потерянных строк выдали четыре предложения: одно своё для574 и три неопределённых вне прежней области восстановления. Обе не прошли строгий критерий отсутствия чужих и неопределённых новых предложений. Файлы result.json и result_lost_only.json, исходные модули, протоколы и отчёты сохранены. Неопределённые предложения не объявлены ошибочными или правильными.

Финальная область применения была зарегистрирована позже как инженерная реакция на провал широких версий. Она использует независимо проверенные прежние условия допуска эпизода, не эталонный фильтр. Проверка выполнена на тех же пяти уже исследованных случаях; это не независимая оценочная выборка. Новых вариантов правил или порогового поиска после финального результата не предлагается.

Сохранён и технический сбой первой оценки: после частичного вычисления случая232 оценщик встретил прошлый сырой кадр533 без точной эталонной привязки. Техническая версия сохраняет такую оценку неопределённой, а не подбирает соседнюю разметку. История и уточнение прежней неполной формулировки «исходы не вычислялись» находятся в execution_failure_context.json.

5. Проверки и граница результата

Независимый пересчёт финальной версии подтвердил642решения строк и183проверки защиты исходных назначений. Максимальное расхождение косинусной близости5,551115123125783e−16, пересечения рамок0. Пять исходных целевых описателей сохранены точно,222исходные близости воспроизведены с ошибкой не более3,3306690738754696e−16. Источники: audit_episode.json и result_episode.json, baseline.

Модуль является действующим кандидатом для ограниченного адаптера. Его включение в полный исполняемый трекер и последующий жизненный цикл идентификаторов здесь не запускались. Первоначальное чужое предложение574 уже блокировалось глобальным сопоставлением; новый правильный локальный кандидат не доказывает исправление публичного идентификатора. HOTA (метрика качества обнаружения и сопоставления траекторий), обобщение на новые кадры и вклад ядра FlyVis не измерены.

Правильные дополнительные предложения для574:0→1; относительный процент к нулевой базе не определён. В пяти контролях без собственных кандидатов ложных новых продолжений0из5. Доля измеренного потолка полного трекера не определена, поскольку такой потолок здесь не измерялся.
'''
import re
text = re.sub(r'(?<=[А-Яа-я])(?=[0-9])|(?<=[0-9])(?=[А-Яа-я])', ' ', text)
with (OUT / 'ИТОГ_ВОССТАНОВЛЕНИЯ_574.md').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(text.rstrip() + '\n')
