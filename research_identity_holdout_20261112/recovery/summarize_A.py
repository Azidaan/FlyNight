"""Post-audit accounting only; no policy re-evaluation or threshold changes."""
from common import ROOT, OUT, read, sha, write, now
from collections import Counter, defaultdict
result = read(OUT / 'result.json')
audit = read(OUT / 'audit.json')
assert audit['passed']
runtime = {event['event_key']: event for event in read(OUT / 'runtime_cases.json')['cases']}
totals = defaultdict(Counter)
controls = defaultdict(Counter)
decisions = Counter()
gate_categories = Counter()
gate_rows = []
known = []
unknown_proposals = []
control_pools = defaultdict(set)
for receipt in result['events']:
    assert sha(OUT / receipt['path']) == receipt['sha256']
    event = read(OUT / receipt['path'])
    case = runtime[event['event_key']]
    current_exact = any(value['exact'] for value in event['current_evaluation'].values())
    primary = next(mode for mode in event['modes'] if mode['mode'] == 'fixed_repair')
    for mode in event['modes']:
        name = mode['mode']
        if not mode['valid']:
            controls[name]['invalid'] += 1
            continue
        totals[name]['evaluations'] += 1
        totals[name]['proposals'] += len(mode['policy']['proposals'])
        totals[name].update('primary_' + value['outcome'] for value in mode['evaluation'])
        totals[name].update('historical_' + value['outcome'] for value in mode['historical_evaluation'])
        if name.startswith('remove_all'):
            controls[name]['valid'] += 1
            tested = next(decision for decision in mode['policy']['decisions'] if decision['tid'] == mode['control_tid'])
            controls[name]['tested_' + tested['action']] += 1
            control_pools[name].add((event['event_key'], mode['control_tid'], tuple(mode['indices'])))
    for proposal in primary['policy']['proposals']:
        candidate = next(candidate for candidate in case['candidates'] if candidate['raw_id'] == proposal['raw_id'])
        literal = next(value for value in primary['evaluation'] if value['tid'] == proposal['tid'])
        historical = next(value for value in primary['historical_evaluation'] if value['tid'] == proposal['tid'])
        unknown_proposals.append(dict(event_key=event['event_key'], video=event['video'], frame=event['frame'], **proposal,
                                      detector_confidence=candidate['confidence'], literal_last_source=literal, historical_expected=historical,
                                      classification='unknown, not a demonstrated wrong or correct association', source=receipt['path']))
    for decomposition in event['decomposition']:
        decision = decomposition['decision']
        decisions[decision['reason']] += 1
        if decision['reason'] == 'no_excluded_low_candidate_passes_fixed_appearance_and_proximity':
            low = [trace for trace in decomposition['group_gate_trace'] if trace['same_class'] and trace['excluded_low']]
            appearance = [trace for trace in low if trace['appearance_pass']]
            proximity = [trace for trace in low if trace['proximity_pass']]
            joint = [trace for trace in low if trace['appearance_pass'] and trace['proximity_pass']]
            category = 'no_low_representative' if not low else 'no_low_appearance_pass' if not appearance else 'appearance_pass_exists_but_no_joint_proximity_pass'
            gate_categories[category] += 1
            gate_rows.append(dict(event_key=event['event_key'], tid=decomposition['tid'], low_groups=len(low), appearance_passing=len(appearance), proximity_passing=len(proximity), joint_passing=len(joint), accounting_category=category, source=receipt['path']))
        tid = decomposition['tid']
        reference = decomposition['reference_evaluation']
        history = event['historical_expected'][str(tid)]
        strict = current_exact and reference['exact'] and len(reference['hits']) == 1
        anchored = current_exact and history['valid']
        if strict or anchored:
            identity = reference['hits'][0]['identity'] if strict else history['expected_identity']
            own_candidates = [dict(candidate=candidate, current_evaluation=event['current_evaluation'][str(candidate['raw_id'])]) for candidate in case['candidates'] if any(hit['identity'] == identity for hit in event['current_evaluation'][str(candidate['raw_id'])]['hits'])]
            covered = [native for native in event['native_evaluation'] if any(hit['identity'] == identity for hit in native.get('candidate', {}).get('hits', []))]
            own_trace = [trace for trace in decomposition['group_gate_trace'] if any(hit['identity'] == identity for hit in trace['candidate_truth']['hits'])]
            known.append(dict(event_key=event['event_key'], tid=tid, primary_evaluable=strict, historical_evaluable=anchored,
                              literal_reference=reference, expected_history=history, decision=decision, own_raw_candidates=own_candidates,
                              existing_native_current_identity_claims=covered, same_tid_baseline=[native for native in event['native_historical_evaluation'] if native['tid'] == tid],
                              exact_gate_witness=own_trace, all_primary_own_raw_gate_observations=decomposition['all_raw_own_observations'], source=receipt['path']))
summary = dict(created=now(), events=result['count'], videos=len({case['video'] for case in runtime.values()}), row_occurrences=sum(len(case['rows']) for case in runtime.values()), raw_candidate_observations=sum(len(case['candidates']) for case in runtime.values()),
               totals={key: dict(value) for key, value in totals.items()}, controls={key: dict(value) for key, value in controls.items()},
               unique_counterfactual_pools=len(set().union(*control_pools.values())), duplicate_primary_historical_counterfactual_pools=len(control_pools['remove_all_own'] & control_pools['remove_all_historical_own']),
               eligible_episode_decision_reasons=dict(decisions), absolute_gate_accounting=dict(gate_categories), absolute_gate_rows=gate_rows,
               identifiable_cases=known, new_proposals=unknown_proposals, audit=audit,
               conclusion='Inconclusive for new-case identity benefit/safety: all6new proposals areunknown in bothregistered tasks. No demonstrated wrong newpair; no demonstrated correct newpair. The1primary/2historical evaluable episodes offer no identified uncovered low-confidence continuation opportunity.',
               known_case_2179='GT00122657 iscurrentlycoveredbypublic2445. Previous2179 isnotrestored. Fixedbranch refusal explainedbyNMS/nativeownership pluslowappearance/proximity; snapshotalone cannotestablish thecauseof earlierIDtransition. Saved-native trace separatelylocalizes2179assignmentloss1160 andGTownerinterval1159to1165.',
               non_conclusions=['No evidence of algorithmic impossibility from absence of an observable positive opportunity', 'Unknown6 are not observed errors or successes', 'No fulltracker/HOTA/publicIDlifecycle improvement', 'No GT labelling of unannotated lastreference fromhistoricalexpectedID', 'No B descriptor/Census/policy outcomes'],
               source_hashes={name: sha(OUT / name) for name in ['result.json', 'audit.json', 'runtime_cases.json']})
write('summary_A.json', summary)
write('criteria_A.json', dict(state='inconclusive', criteria=[dict(criterion='Technical independent audit and immutable frozen rule', met=True, evidence='audit.json'),
                                                            dict(criterion='At leastone correct newpair andzero unknown/foreign in primary task', met=False, evidence=dict(correct=0, foreign=0, unknown=6, source='summary_A.json:totals.fixed_repair')),
                                                            dict(criterion='Same positive evidence in separate historicalexpectedID task', met=False, evidence=dict(correct=0, foreign=0, unknown=6, source='summary_A.json:totals.fixed_repair')),
                                                            dict(criterion='No testedrowcontinues in valid remove-own controls', met=True, evidence='summary_A.json:controls;1primary/2historical tests, only2distinctcounterfactualpools'),
                                                            dict(criterion='Allselectedcases andcompetitors retained; unfavorable/unknown outcomes preserved', met=True, evidence='result.json/event receipts andsummary_A.json')],
                              interpretation='Unmetpositiveevidence criteria arise fromannotationreadiness/absenceofnewobservablepositiveopportunity, not demonstrated algorithmicfailure. Do not tune thresholds to fix missing GT.'))
text = '''1. Что показала проверка на новых для выбора правки видео

На54эпизодах из пяти видео фиксированная ветвь восстановления выдала6новых предложений. Все6остались неопределёнными как при проверке последнего образца, так и в отдельной задаче продолжения исторического идентификатора. У всех шести последний образец не имеет точной разметки; у пяти нет точной разметки текущего кадра, а у шестого текущая рамка не попала в эталонную рамку при пороге пересечения0,5. Это не шесть доказанных ошибок и не шесть успехов. Перестановка описателей не выдала предложений. Источники: summary_A.json, totals/new_proposals; result.json и файлы событий.

Из54целевых решений28сохранили штатное назначение,2отложены из-за занятой группы рамок,18отказали по абсолютным условиям дополнительной ветви. Среди этих18в11случаях ни один представитель с низкой уверенностью не прошёл порог внешнего вида; в7были кандидаты, прошедшие внешний вид, но не нашлось кандидата, одновременно прошедшего геометрию. Это точный разбор условий отказа, а не доказательство неправильности отказов без разметки. Источник: summary_A.json, absolute_gate_accounting/absolute_gate_rows.

2. Два оцениваемых случая

Кадр1165 видео b1ca2e5d-84cf9134, прежняя строка2179 — единственный случай с точной разметкой последнего образца и текущего кадра. Объект00122657 уже покрыт штатной детекцией110260 с уверенностью0,855911 под публичным идентификатором2445. Собственные низкоуверенные рамки110318 и110366 имеют уверенности0,030377 и0,011606; обе вошли в группу представителя110260, которая уже занята. Их исходные близости0,296442 и0,310172, пересечения с предсказанной рамкой0,269077 и0,270883 также ниже неизменных порогов. Ветвь отказалась от нового восстановления:2179 не возвращён, владение2445 сохранено. Это не доказывает исправность прежней смены идентификатора во всей системе. Причина предыдущего перехода не устанавливается одним этим снимком. Источник: summary_A.json, identifiable_cases; отдельная сохранённая хронология — identity_transition_2179_2445.json.

Кадр649 того же видео, строка1111 оценивается только по исторической задаче. Точно размеченное наблюдение643 и текущая детекция46466 относятся к00122615; последний образец внешности получен на неразмеченном кадре644 и не получает перенесённую метку. Детекция46466 с уверенностью0,474425 уже назначена1111 штатным алгоритмом. Существующее правильное историческое продолжение сохранено. Новый низкоуверенный кандидат для исправления не нужен. Источник: summary_A.json, identifiable_cases.

Таким образом, среди одного основного и двух исторических оцениваемых эпизодов не обнаружена новая проверяемая возможность вернуть непокрытый собственный объект через исключённую низкоуверенную детекцию. Отсутствие доказанного выигрыша на этих данных не доказывает неработоспособность алгоритма.

3. Контроли, проверка и граница вывода

Основной контроль удаления собственных кандидатов допустим в1случае, исторический — в2. Во всех случаях новые предложения отсутствуют; остальные53и52контроля соответственно явно недопустимы из-за разметки. Один контроль совпадает между определениями, поэтому это2различных набора, а не3независимых испытания. Контроли отсутствия описателей и конкурентов откладывают решения. Сохранены все1211вхождений строк и4916сырых рамок, включая конкурентов вне области выдачи новых предложений.

Независимый пересчёт подтвердил6168решений,109191скалярное сравнение описателей и6929записей разметки; максимальное численное расхождение8,881784197001252e−16. Результат для качества идентичности неопределённый. Пороги и производственный модуль не менялись; полного трекера, HOTA и жизненного цикла идентификаторов здесь нет. Доля измеренного потолка не определена. Источник: audit.json.

Резерв B исследован только по метаданным разметки:25эпизодов, основной и исторический оцениваемый случай по1, ожидаемая историческая идентичность в текущей разметке этого случая отсутствует. Описатели, Census и политика для B не вычислялись. Недостаток эталонной информации нельзя исправить изменением порога. Источник: B_metadata_readiness.json.
'''
import re
text = re.sub(r'(?<=[А-Яа-я])(?=[0-9])|(?<=[0-9])(?=[А-Яа-я])', ' ', text)
text = re.sub(r'(?<=[А-Яа-я]),(?=\d)', ', ', text)
write('B_metadata_receipt.json', dict(created=now(), script_sha256=sha(OUT / 'B_metadata_readiness.py'), result_sha256=sha(OUT / 'B_metadata_readiness.json'),
                                       initial_schema_error='B_videos key absent; corrected to exactfrozen B_reserved_videos before GT metadata processing; nooutput/model/policy run atfailure.',
                                       reserve_policy_outcomes_computed=False, A_fixed_policy_unmodified=True))
with (OUT / 'ИТОГ_A.md').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(text.rstrip() + '\n')
print('A_SUMMARY_READY')
