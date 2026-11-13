from common import *
import shutil, psutil
p=read(OUT/'protocol.json'); audit=read(OUT/'audit.json'); trace=read(OUT/'trace_v2.json'); assert audit['passed'] and read(OUT/'science3_queue_execution.json')['passed']
variants=[]
for policy in p['policies']:
    for alpha in p['alphas']:
        rows=[x for x in audit['summary'] if x['policy']==policy and x['alpha']==alpha]
        known=[x for x in rows if x['frame']!=1160]
        variants.append(dict(policy=policy,alpha=alpha,correct=sum(x['correct'] for x in known),wrong=sum(x['wrong'] for x in known),target2179_at1160=next(x['target2179'] for x in rows if x['frame']==1160),target2179_at1165=next(x['target2179'] for x in rows if x['frame']==1165),lost_known_correct=sum(len(x['lost_correct']) for x in known),new_known_wrong=sum(len(x['new_wrong']) for x in known)))
real=[x for x in variants if x['policy'] not in ['native','latest','swap_latest_negative']]
assert all(not x['target2179_at1160'] and not x['target2179_at1165'] for x in real)
source=next(o for o in trace['observations'] if o['tid']==2445 and o['frame']==1159)
target=lambda rows:next(x for x in rows if x['identity']=='00122657')
coverage=dict(filtered=target(source['filtered_ownership'])['covered_object_fraction'],raw=target(source['raw_ownership'])['covered_object_fraction'])
save('summary.json',dict(passed=True,assignment_matrices=len(audit['summary']),variants=variants,real_memory_policies=6,real_memory_target_recoveries=0,history_observations=len(trace['observations']),coverage_target_in_competitor_box1159=coverage,old_descriptor_reproduction_max_error=read(OUT/'feature_result.json')['old_descriptor_max_error'],production_changes=0,core_calls=0,full_tracker_calls=0,promoted_variant=None,known_correct_change_percent=0,scope=p['scope'],conclusions=['Raw-versus-filtered box substitution alone is insufficient in the tested snapshots.','Freeze2445 before the loss is insufficient: later updates are not necessary for this frozen-decision failure.','Earliest retained observations still favor competitor in target column; minimum-over-history retains the strongest misleading match.','Descriptor-swap negative control changes global ownership at alpha1 but adds a wrong2445assignment at1159; cannot be a repair.','Exact first physical switch and unique cause of descriptor confusion remain unproven; box overlap is measured, not alone a causal segmentation experiment.']))
for n,h in p['inputs'].items(): assert sha(ROOT/n)==h,n
execution=OUT/'execution';execution.mkdir()
for name in ['science_queue_execution.json','science2_queue_execution.json','science3_queue_execution.json']:
    queue=read(OUT/name)
    for job in queue['jobs']:
        for pid,birth in job['result']['owned'].items():
            try: alive=psutil.Process(int(pid)).create_time()==birth
            except psutil.NoSuchProcess: alive=False
            assert not alive,pid
        for file in [Path(job['report']),Path(job['report']).with_name('memory_guard.worker.log')]:
            if file.exists():shutil.copyfile(file,execution/(job['token']+'_'+file.name))
save('provenance.json',dict(passed=True,all_registered_inputs_unchanged=True,all_scientific_workers_ended=True,production_changes=0,no_core_calls=True,no_full_tracker_calls=True,failures_preserved=['science_queue_execution.json','science2_queue_execution.json'],empty_failed_trace_preserved=True,technical_gpu_recovery_required=True))
report='''1. Проверка памяти2179

Проверены54полные матрицы назначения на трёх прежних состояниях1159/1160/1165: исходный вариант, прежняя внешность, шесть правил выбора памяти и отрицательный контроль с перестановкой образцов. Веса0,25/1и порог0,8 прежние. Изменялись только исторические признаки2179/2445, остальные строки и кандидаты сохранены. Источники: protocol.json; result.json; summary.json.

Ни одно из шести правил памяти не восстановило2179на1160или1165. Не помогли исходная рамка детектора вместо сглаженной, самый ранний из сохранённых образцов, минимум расстояния по набору прошлых образцов и заморозка образца2445на1158/1159. При весе0,25сохраняются18известных правильных назначений и1неправильное; при весе1остаются18правильных и3неправильных, как у прежней внешности с этим весом. Новые ошибки большого веса не вызваны заменой памяти: они уже есть в согласованном контроле. Источник: summary.json/variants.

2. Что установлено о механизме

Архивный Planner.after записывает в историю сглаженную t.xyxy, затем адаптер добавляет raw_id, не заменяя рамку. Оба файла совпали с контрольными суммами паспорта исходного запуска. Это реальный путь формирования входа памяти. Источник: source_localization.json.

На точно размеченном1159номер2179сопоставлен машине00122657, а2445—другой машине00122746. Однако рамка2445геометрически покрывает большую часть прямоугольника2179. Покрытие исходной и сглаженной рамкой измерено отдельно в summary.json/coverage_target_in_competitor_box1159. Это пересечение прямоугольников разметки, не измерение доли видимых пикселей и не доказательство единственной причины ошибки признаков.

На1165стоимость внешности для2179→110260равна0,359247, а для2445→110260—0,012743. Заморозка2445на1159увеличивает вторую стоимость до0,108202, но предпочтение2445и итоговое назначение сохраняются. Следовательно, последующие обновления усилили предпочтение конкурента, но не являются необходимым условием этого отказа на сохранённом состоянии. Замена только сглаженной рамки исходной также недостаточна. Источник: audit.json/summary, policy freeze2445_1159 и latest_raw.

Набор прошлых образцов с выбором минимального расстояния сохраняет самый похожий неподходящий образец. Поэтому добавление истории само по себе не устраняет неверное предпочтение. Deep SORT использует набор прошлых признаков и минимальное косинусное расстояние; статья обосновывает такой контроль, но не обещает устойчивость к неверно сопоставленным образцам. Источник: https://arxiv.org/html/1703.07402, раздел2.2; наш результат audit.json, policy gallery_raw.

Перестановка последних образцов двух номеров при весе1передаёт целевую рамку2179на1160и1165, но добавляет ошибочное назначение2445на1159. Это отрицательный контроль: он подтверждает влияние признаков на результат, но не является допустимым исправлением. Точной разметки1160нет; неизвестное назначение не объявляется успехом. Источник: summary.json, policy swap_latest_negative.

3. Проверки и технические отказы

Независимый скалярный пересчёт косинусных расстояний и другой решатель назначения проверили все54матрицы; максимальное расхождение стоимости2,980067526081598e-8, целевой функции3,552713678800501e-15. Исходные и прежние матрицы внешности воспроизведены точно. Источник: audit.json.

Первый технический отказ вызван записью numpy.float32в JSON. Второй—изменением численного результата MASAпри другом размере пакета рамок: максимум0,00010757893323898315 превысил прежний допуск0,00001. После восстановления исходного размера пакета и позиции рамки совпадающие признаки воспроизведены с ошибкой0. Допуск не ослаблялся. Оба отказа сохранены: trace_recovery_protocol.json; feature_recovery_protocol.json; feature_result.json.

4. Решение и предел вывода

Шесть проверенных замен памяти не принимать. Рабочие настройки, веса моделей и прежние результаты сохранены; ядро и полный трекер не запускались. Причина не сводится к возрасту последнего образца или только сглаживанию рамки. На этом состоянии неверное предпочтение уже присутствует в сопоставлении по полным рамкам до последующего обновления памяти. Единственная физическая причина смешения признаков и первый кадр переноса номера пока не доказаны.

Следующая содержательная правка должна изменить сравниваемое содержимое области: отделить признаки доступной части машины от соседа/фона с согласованным контролем полного прямоугольника. Повторять сетку весов или просто увеличивать память не обосновано. Это направление следующей регистрации, не выполненный опыт и не доказанное решение.

По числу подтверждённых правильных назначений:18→18, изменение0процентов к исходному варианту для шести правил памяти. Восстановлений2179в двух целевых состояниях0→0; относительный процент при нулевой базе не определяется. Доля измеренного потолка неизвестна. Это локальная проверка сопоставления, не HOTAи не качество полного трекера.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8',newline='\n') as f:f.write(report)
save('text_review.json',dict(passed=True,checks=['Each scientific sentence checked for method versus object, duplication, and correspondence to actual evidence.','No visible pixel fraction inferred from GT rectangle overlap.','No unique root cause or unannotated first-switch claim.','Diagnostic swap excluded from accepted repair.']))
base=OUT.relative_to(ROOT).as_posix();cat=read(ROOT/'experiments/project_history.json');c=next(x for x in cat['experiments'] if x['id']=='history_representation_2179_20261113')
c.update(state='completed_negative',done='2026-11-13',obtained='Шесть замен исторических образцов не восстановили2179на1160/1165; всего54матрицы,18известных правильных назначений остаются18. Источник '+base+'/summary.json. Исходная/сглаженная рамка, ранний образец, набор прошлых образцов и заморозка2445проверены при прежних весах0,25/1. Отрицательный контроль меняет целевое назначение, но добавляет ошибку2445на1159. Два технических отказа сохранены; восстановление размера пакета MASAдаёт точное совпадение повторных признаков.',interpretation='Последующие обновления усиливают предпочтение2445, но заморозка до потери номера не исправляет его: обновление после1159не является необходимой причиной отказа на этих состояниях. Замена только сглаженной рамки исходной также недостаточна. Факт пересечения прямоугольников не доказывает единственную причину смешения признаков. Источник '+base+'/audit.json; trace_v2.json.',decision='Не продвигать проверенные варианты памяти. Отказаться от этой простой замены образцов; следующий кандидат должен менять сравниваемое содержимое области и отделять доступную часть объекта от соседа/фона. Полный трекер остаётся запрещён.',cannot_claim='Один знакомый случай, три фиксированных состояния; без развёртывания жизненного цикла и без HOTA. Нет точной разметки1160, единственная физическая причина и первый неверный кадр переноса не установлены. Новые назначения неизвестной личности не считаются успехом. Было два технических запуска извлечения признаков вместо одного, повтор устранял нарушение воспроизводимости без ослабления допуска.')
for i,item in enumerate(c['acceptance']): item.update(met=(i!=5),evidence=base+('/feature_recovery_protocol.json; '+base+'/provenance.json' if i==5 else '/audit.json; '+base+'/summary.json'))
c['evidence'].extend(dict(kind='file',ref=base+'/'+n) for n in ['summary.json','audit.json','source_localization.json','trace_v2.json','provenance.json','РЕЗУЛЬТАТ.md'])
(ROOT/'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode())
problem=ROOT/'experiments/system_problems.json'; problems=read(problem);s5=next(x for x in problems['problems'] if x['id']=='С5')
s5.setdefault('history',[]).append(dict(date='2026-11-13',component='Память и внешнее сопоставление всей системы',fact=c['obtained'],hypothesis='Пересечение области соседа/фона может смешивать признаки полного прямоугольника; это рабочая версия, не единственная доказанная причина.',discriminating_check='Шесть замен памяти, изолированная заморозка2445до потери, отрицательная перестановка, полные матрицы при прежних весах. Следующая правка—сравнение доступной части объекта с контролем полного прямоугольника.',limitation=c['cannot_claim'],evidence=[base+'/summary.json',base+'/audit.json'],source_sha256=sha(OUT/'summary.json')))
problem.write_bytes((json.dumps(problems,ensure_ascii=False,indent=2)+'\n').encode())
journal('Итог (`history_representation_2179_20261113`). Шесть правил памяти не восстановили2179, проверены54полные матрицы. Исходная рамка вместо сглаженной, ранний образец, набор прошлых образцов и заморозка2445до потери недостаточны. Последующие обновления усиливают предпочтение2445, но отказ сохраняется без них; единственная физическая причина не доказана.18правильных→18,0процентов. Источники '+base+'/summary.json; audit.json; source_localization.json. Рабочие настройки не менялись, ядро/полный трекер не запускались; два технических отказа сохранены, размер пакета MASAвосстановлен без ослабления допуска. Решение: эту замену памяти не принимать; следующий кандидат меняет содержимое сравниваемой области.')
print('MEMORY_RESULT_RECORDED',flush=True)
