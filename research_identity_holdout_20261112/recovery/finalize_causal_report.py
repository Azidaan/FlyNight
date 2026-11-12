"""Final posthoc interpretation of the last registered diagnostics; no tests."""
from common import ROOT, OUT, read, sha, write, now
formula = read(OUT / 'assignment_loss_result_v2.json')
column = read(OUT / 'column109699_metadata.json')
mask = read(OUT / 'classmask_result.json')
trace = read(OUT / 'identity_transition_2179_2445.json')
metadata = read(OUT / 'native_class_metadata_A.json')
summary_a = read(OUT / 'summary_A.json')
ready_b = read(OUT / 'B_metadata_readiness.json')
conditions = [dict(condition='native_baseline', owner=formula['raw109699_baseline_owner'], target2179_assigned=False),
              dict(condition='remove2445_only', owner=formula['raw109699_without2445_owner'], target2179_assigned=bool(formula['target2179_without2445_assignment']))]
conditions += [dict(condition=record['variant'], owner=record['raw109699_owner'], target2179_assigned=bool(record['target2179_assignment'])) for record in mask['records']]
summary = dict(created=now(), A_recovery=dict(events=54, new_proposals=6, primary=dict(own=0, foreign=0, unknown=6), historical=dict(own=0, foreign=0, unknown=6),
                                              eligible_decision_reasons=summary_a['eligible_episode_decision_reasons'], absolute_gate_accounting=summary_a['absolute_gate_accounting'], conclusion='Inconclusive identity benefit, not6observed errors; no identifiable newuncoveredpositiveopportunity in1primary/2historical evaluableepisodes.'),
               recorded_transition=dict(last_reliable_anchor=1159, lost_assignment_frame=1160, recorded_lost_state_frame=1161, current_anchor=1165,
                                        GT_at1159={'2179': '00122657', '2445': '00122746'}, GT_at1165={'2445': '00122657'},
                                        missing_saved_frames_in1150to1165=[frame['frame'] for frame in trace['frames'] if not frame['event_saved']],
                                        identity_error_interval=[1159,1165], exact_first_GT_error_frame_unknown=True,
                                        object2445_preexisted_loss=trace['firsts']['2445'],
                                        scope='Assignmentloss1160 is observed, not proof109699 shouldbelongto2179 atthatunannotatedframe. GTownerchange2445 acrossanchors confirmed; physical/neural rootcause remainsunproven.'),
               native_formula=dict(formula='1-IoU(predicted_xyxy,current_native_box)*detection_confidence', direct_appearance_in_base_cost=False, direct_history_in_base_cost=False, history_via_saved_Kalman_mean=True,
                                   shared_detection_confidence=formula['rows'][0]['detection_confidence'], rows=formula['rows'], full_matrix_entries=41*23,
                                   full_float32_formula_error=formula['full_matrix_float32_formula_max_error'], neural_correction_changes=formula['changed_by_neural_readout']),
               full_column_ranking_source='column109699_metadata.json', target2179_rank_all=column['target2179_rank_all'], target2179_only_admissible_high_edges=column['target2179_high_edges_within_native_limit'],
               counterfactual_conditions=conditions, interpretation='Inthisregisteredfourconditionmatrixcheck, neitherremoving2445alone norclassmaskalone returns2179; theircombination does. Thisrejects2445as thesole self-sufficientassignment-loss explanation andshows cross-class alternativecompetition. Itdoesnotproveown109699 orsafeuseofdeletingactive2445.',
               class_metadata=dict(actual_first_pairs=metadata['actual_first_pairs'], cross_class=metadata['cross_class_pairs'], events_with_cross_class=metadata['events_with_cross_class'],
                                   percentage=100*metadata['cross_class_pairs']/metadata['actual_first_pairs'], directions=metadata['class_pair_counts'],
                                   meaning='Saved detector-class labels differ, not proof ofGTidentity errors. Noactual0->2pairs inthese54A; the0->2rival2316 isonlythiscounterfactual.'),
               solver_checks=dict(baseline_reproduced=True, max_independent_objective_error=max(formula['independent_solver_objective_max_error'], mask['baseline_objective_error'], *(record['objective_error'] for record in mask['records'])),
                                  compatible_costs_unchanged=mask['surviving_compatible_costs_unchanged']),
               B_metadata_only=ready_b['summary'], B_repair_outcomes_computed=False, no_more_intervention_variants=True,
               technical_failure_preserved=dict(first_proof_failed_error=2.1457672119140625e-6, unchanged_acceptance_tolerance=1e-6,
                                                 fix='Actualnative xyxy->xywh->tlwh->xyxy initialization arithmetic, not threshold relaxation.', corrected_typed_error=formula['full_matrix_float32_formula_max_error']),
               cannot_claim=['Fulltracker/HOTA improvement','CorrectGTidentityof109699 at1160','Deployabledeletionofactive2445','Classmaskalonefixes2179','All6detectorclassmismatchesareGTerrors','Bperformance','Uniquephysical/neuralcauseofearlierIDchange'],
               evidence={name: sha(OUT / name) for name in ['summary_A.json','B_metadata_readiness.json','identity_transition_2179_2445.json','assignment_loss_protocol.json','assignment_loss_protocol_v2.json','assignment_loss_result_v2.json','column109699_metadata.json','native_class_metadata_A.json','classmask_protocol.json','classmask_result.json']})
write('final_causal_summary.json', summary)
write('final_diagnostic_criteria.json', dict(scope='Late registered source/matrixdiagnosis onA; no newrepair choice orfullmodel evaluation.',
                                            acceptance=[dict(criterion='Reproduce all943savedbase costs fromactualnative geometry/confidence', met=True, evidence='assignment_loss_result_v2.json: full_matrix_float32_formula_max_error=0'),
                                                        dict(criterion='Originalfirstassignment exact andidentitycopy same', met=True, evidence='assignment_loss_result_v2.json: baseline_pairs'),
                                                        dict(criterion='Removeonly2445 returns2179', met=False, evidence='assignment_loss_result_v2.json: owner2316; thisnegativecriterion preserved'),
                                                        dict(criterion='Classcompatibilitymaskalone returns2179', met=False, evidence='classmask_result.json: classmask_only owner2445'),
                                                        dict(criterion='Combinedclassmask+remove2445 returns2179 inthisartificialmatrix', met=True, evidence='classmask_result.json: classmask_and_remove2445'),
                                                        dict(criterion='Unchanged compatiblecosts/costlimit andindependentoptimalobjectives', met=True, evidence='classmask_result.json andassignment_loss_result_v2.json'),
                                                        dict(criterion='ReportnoGT1160 andno deployable/fulltracker improvement', met=True, evidence='final_causal_summary.json: cannot_claim')]))
text = '''1. Где зафиксирована потеря прежнего назначения

На кадре1159 видео b1ca2e5d-84cf9134 публичный идентификатор2179 соответствует эталонному объекту00122657, а2445 — другому объекту00122746. На сохранённом кадре1160 строка2179 не получает назначения, на1161 уже записана как потерянная. На точно размеченном кадре1165 тот же публичный2445 соответствует00122657. Поэтому подтверждён переход аннотированной принадлежности2445 в интервале1159–1165. Точный первый ошибочный кадр неизвестен: на1160 нет точной разметки. Потеря назначения2179 на1160 сама по себе не доказывает, что именно рассматриваемую детекцию ему следовало назначить.

Идентификатор2445 не создан в ответ на эту потерю: его назначение сохранено уже на сыром кадре1106, публичная выдача — на1107. Архив не содержит события каждого кадра; из1150–1165 записаны9событий. Сохранённые1159,1160и1161 позволяют точно локализовать шаг потери назначения, но не восстановить ненаблюдавшиеся эталонные состояния. Источник: identity_transition_2179_2445.json.

2. Что создало разницу стоимостей

Первая штатная матрица рассчитывает стоимость как1−IoU×уверенность детектора, где IoU — доля пересечения рамок относительно их объединения. Для одной и той же сырой детекции109699 уверенность равна0,9255288243. Предсказанная рамка2179 имеет IoU0,307515234 и стоимость0,715385795; рамка2445 имеет IoU0,852321506 и стоимость0,211151898. Значит, разница в этом столбце создана геометрией предсказанных рамок при общей уверенности. Внешность напрямую в штатную стоимость не входит, история влияет через уже сохранённое состояние фильтра Калмана. Нейронный считыватель не изменил ни одного элемента всей матрицы этого кадра.

Независимое воспроизведение фактического пути преобразования координат дало точное совпадение всех943стоимостей. Первый технический вариант ошибочно использовал сырые координаты до преобразования через центр и размер рамки; он не прошёл исходный допуск. Исправлена только эта подготовка числового входа, допуск не ослаблялся. Первый сбой, исходник и протокол сохранены. Источник: assignment_loss_result_v2.json.

3. Что показали все зарегистрированные контроли

В исходной матрице детекцию109699 получает2445. После удаления только строки2445 её получает2316 со стоимостью0,697521091, а2179 остаётся без назначения. У2316 сохранён класс0, у текущей детекции — класс2; штатная стоимость не запрещает такую межклассовую конкуренцию. Это различие меток детектора, а не утверждение о физической идентичности на неразмеченном кадре.

Сам по себе запрет рёбер между разными классами не меняет исходное назначение:2445 продолжает выигрывать. Только сочетание этого запрета с удалением2445 отдаёт109699 строке2179. Следовательно, в заданной четырёхвариантной проверке обе одиночные меры недостаточны, а их сочетание меняет исход. Это опровергает объяснение, в котором2445 является единственной самодостаточной причиной отказа. Удаление действующей траектории — диагностическое вмешательство, а не предлагаемая рабочая правка. Правильность назначения109699 строке2179 на1160 не установлена. Источники: assignment_loss_result_v2.json и classmask_result.json.

Для2179 рассматриваемая детекция — единственное ребро первой стадии дешевле неизменного предела0,8; следующее имеет стоимость0,866131187. Строка2179 занимает третье место по стоимости в столбце после2445 и2316. Её ребро проходит порог, но проигрывает глобальной конкуренции. Матрица второй, низкоуверенной стадии в архиве не сохранена;13таких сырых кандидатов перечислены без выдуманного результата второй стадии. Источник: column109699_metadata.json.

На всех54эпизодах A найдено6межклассовых пар из611фактических первых назначений, около0,982%. Это переходы меток автомобиль/автобус/грузовик; они не являются шестью доказанными ошибками эталонной идентичности. Пара2316→автомобиль возникает только в диагностическом удалении2445, а не среди этих611фактических назначений. Источник: native_class_metadata_A.json.

4. Что можно заключить для всей проверки

Новая ветвь восстановления на A выдала6неопределённых предложений. Ни их правильность, ни ошибочность разметкой не доказаны. В единственном основном и двух исторических оцениваемых эпизодах не обнаружена новая непокрытая низкоуверенная возможность, на которой можно доказать выигрыш. Поэтому результат для пользы новой ветви остаётся неопределённым, а не доказательством её неработоспособности. Случай2179 при этом выявил отдельную проблему сохранения идентификаторов в исходной системе; её нельзя объявлять решённой тем, что дополнительная ветвь сохранила занятое назначение2445.

Контрольные задачи линейного назначения подтверждены независимым решателем с расхождением целевой функции не более3,553e−15. Пороги, веса и производственные модули не изменялись. Полный трекер, жизненный цикл и HOTA не пересчитывались. На B выполнена только проверка метаданных, без вычисления описателей или политик. Исследования этой серии завершены; новых вариантов не предложено. Относительный прирост к исходной модели и доля измеренного потолка не определены.
'''
import re
text = re.sub(r'(?<=[А-Яа-я])(?=[0-9])|(?<=[0-9])(?=[А-Яа-я])', ' ', text)
text = re.sub(r'(?<=[А-Яа-я]),(?=\d)', ', ', text)
with (OUT / 'РАЗБОР_ПОТЕРИ_2179.md').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(text.rstrip() + '\n')
write('causal_text_review.json', dict(report_sha256=sha(OUT / 'РАЗБОР_ПОТЕРИ_2179.md'),
                                      checks=dict(method_or_object='Readout, geometry, solver andmatrixintervention consistentlyidentified astools/steps; trackedinstances andIDcontinuity aretheinvestigatedbehavior.',
                                                  adjacent_tautology='Timeline, costmechanism, counterfactualcontrols andlimits have distinctroles; no unsupportedrepetition ofa uniquephysicalcause.',
                                                  actual_work='Everyassertion checkedagainst savedtrace, exactcostreconstruction andregisteredcounterfactuals; missingGT1160/unknown6/noHOTA preserved.'),
                                      author_decision_needed=False, no_further_variants=True))
print('FINAL_CAUSAL_REPORT_READY')
