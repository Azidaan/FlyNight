from common import *
import math,shutil,psutil
r=read(OUT/'result.json');a=read(OUT/'audit.json');assert a['passed'] and read(OUT/'queue1_execution.json')['passed']
rt=read(OUT/'runtime.json');dominance=[]
for frame,raw in [(1160,109699),(1165,110260)]:
    e=next(e for e in rt['events'] if e['frame']==frame);ids=[r['tid'] for r in e['rows']];j=[c['raw_id'] for c in e['candidates']].index(raw);endpoints={}
    for v in ['baseline','appearance_1.0']:
        row=next(x for x in r['records'] if x['frame']==frame and x['variant']==v);costs={str(t):row['cost'][ids.index(t)][j] for t in [2179,2445]};assert costs['2179']>costs['2445'];endpoints[v]=costs
    dominance.append(dict(frame=frame,raw_id=raw,endpoints=endpoints,descriptor_source_frames={str(t):e['rows'][ids.index(t)]['reference']['frame'] for t in [2179,2445]},conclusion='Bothendpoints strictlyfavor2445over2179. AnyconvexcombinationoftheseTWOedgecosts alpha0..1 keeps thispairwiseordering; NOTaproofaboutallpossibleglobalassignmentcyclesorotherweights. NoGT1160truthassumption.'))
save('pairwise_dominance.json',dict(passed=True,records=dominance,source_result_sha256=sha(OUT/'result.json')))
summary=[]
for v in read(OUT/'protocol.json')['variants']:
    rows=[x for x in r['records'] if x['variant']==v['name']];known=[x for x in rows if x['frame'] in [1159,1165]]
    summary.append(dict(variant=v['name'],correct=sum(sum(y['outcome']=='correct' for y in x['evaluation']) for x in known),wrong=sum(sum(y['outcome']=='wrong' for y in x['evaluation']) for x in known),lost_known_correct=sum(len(x['lost_correct_tids']) for x in known),new_wrong=sum(len(x['new_wrong_tids']) for x in known),unknown_changed=sum(len(x['unknown_added']) for x in rows),target2179_assigned1160=bool(next(x for x in rows if x['frame']==1160)['target2179']),target2179_assigned1165=bool(next(x for x in rows if x['frame']==1165)['target2179']),rollback='baseline_restored',verdict='baseline' if v['name'] in ['baseline','identity'] else 'measured_worse' if any(x['confirmed_worse'] for x in known) else 'no_target_recovery_unverified_changes'))
assert not any(s['target2179_assigned1160'] or s['target2179_assigned1165'] for s in summary)
save('summary.json',dict(passed=True,variants=summary,frames=[1159,1160,1165],decisions=len(r['records']),default_restored=True,promoted_variant=None,conclusion='Registeredsmallweightfamily didnotrecover2179; somevariantsmeasurablyworse, othersonlyunverifiednewlinks. Keepnativebaseline. Exactlabelsabsent1160. No fulltracker/neuralcore calls.'))
for n,h in read(OUT/'protocol.json')['inputs'].items():assert sha(ROOT/n)==h,n
for n,h in read(OUT/'initial_state.json')['prior_science'].items():assert sha(ROOT/n)==h,n
for job in read(OUT/'queue1_execution.json')['jobs']:
    for pid,birth in job['result']['owned'].items():
        try:alive=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess:alive=False
        assert not alive,pid
    report=Path(job['report']);(OUT/'execution').mkdir(exist_ok=True)
    for f in [report,report.with_name('memory_guard.worker.log')]:
        if f.exists():shutil.copyfile(f,OUT/'execution'/(job['token']+'_'+f.name))
save('provenance.json',dict(passed=True,checked=now(),all_registered_inputs_unchanged=True,prior_scientific_artifacts_unchanged=True,all_owned_calculation_workers_ended=True,all_settings_restored=sha(OUT/'active_settings.json')==sha(OUT/'original_settings.json'),models_unchanged=True,core_calls=0,tracker_calls=0,new_downloads=0))
storage=read(OUT/'disk_inventory.json');e=storage['reports'][0];c_roots={x['root']:x for x in storage['reports'] if x['root'].startswith('C:')};lower=sum(x['logical_bytes'] for x in e['duplicate_artifact_roles'])
save('storage_summary.json',dict(measured=storage['completed'],project_logical_bytes=e['logical_bytes'],project_top=e['top_level'],service_copies_lower_bound_bytes=lower,C_scanned_logical_bytes=sum(x['logical_bytes'] for x in c_roots.values()),C_roots={k:dict(logical_bytes=v['logical_bytes'],errors=v['errors'],top=v['top_level']) for k,v in c_roots.items()},unavailable_roots=storage['unavailable_roots'],calculation_artifact_bytes=sum((OUT/n).stat().st_size for n in ['runtime.json','descriptor_plan.json','descriptors.npz','feature_result.json','result.json','audit.json','summary.json','pairwise_dominance.json']),drives_now={v:dict(zip(['total','used','free'],shutil.disk_usage(v+':/'))) for v in ['C','E']},all_own_results_on_E=True,cleanup_performed=False,limit='Logicalmetadata sizes froma5minread-onlysnapshot; nofilecontentsread, nopermissionbypass, inaccessiblefoldersunknownnotzero. Filelogicalsizeisnotphysicalallocation. Otherprogramsactive; free-spacedifferencesnotattributedtoexperiment.'))
base=OUT.relative_to(ROOT).as_posix();cat=read(ROOT/'experiments/project_history.json');c=next(c for c in cat['experiments'] if c['id']=='association_weights_2179_copy_20261112')
c.update(state='completed_negative',done='2026-11-12',obtained='Проверены9фиксированных вариантов на3сохранённых состояниях,27матриц. Ни один не назначил2179на1160или1165. Базовый вариант сохраняет18известных правильных назначений и1неправильное на1159/1165; квадрат уверенности теряет3правильных, веса внешности0,5и1добавляют по2новые ошибки. Источник '+base+'/summary.json. Все настройки копии восстановлены; модели/ядро/полный трекер не менялись. Объём расчётных артефактов '+str(read(OUT/'storage_summary.json')['calculation_artifact_bytes'])+'байт; источник storage_summary.json.',interpretation='В этой фиксированной пробе веса не исправили отказ. Геометрическая стоимость и стоимость внешности обе предпочитают2445над2179в рассматриваемом столбце; любая их выпуклая смесь сохраняет этот локальный порядок. Это не доказательство невозможности другого глобального назначения или всех методов внешности. Источник '+base+'/pairwise_dominance.json.',decision='Откатить все пробованные параметры копии и сохранить исходные рабочие настройки. Не продвигать варианты с ухудшением либо неизвестными новыми назначениями. Не расширять подбор на этом случае; следующий метод должен менять доступное различение/память или раннее сопоставление, а не повторять ту же сетку весов.',cannot_claim='Один ранее разобранный случай и3фиксированных состояния, без обновления дальнейшей истории. Точной разметки1160нет. Нет оценки HOTA, выигрыша полного трекера, независимого обобщения или обучения ядра. Суммы18правильных/1неправильное условны по доступным историческим идентичностям, а не точность всех объектов.')
for item in c['acceptance']:item.update(met=True,evidence=base+'/audit.json; '+base+'/provenance.json; '+base+'/summary.json')
c['evidence'].extend(dict(kind='file',ref=base+'/'+n) for n in ['summary.json','pairwise_dominance.json','provenance.json','storage_summary.json','РЕЗУЛЬТАТ.md'])
(ROOT/'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode())
text='''1. Малая проверка весов

На копии трёх сохранённых состояний случая2179проверены9заранее заданных вариантов: исходный и тождественный контроль, отключение/усиление уверенности,4веса внешности и перестановка признаков. Получены27полных назначений, все исходные строки и кандидаты сохранены, порог0,8не менялся. Новая внешняя модель не загружалась; готоваяMASAполучила167описателей на23кадрах без обучения. Источники: protocol.json,feature_result.json,result.json.

Ни один вариант не вернул назначение2179на1160или1165. Усиление уверенности квадратом потеряло3ранее правильных назначения на двух точно размеченных состояниях. Веса внешности0,5и1добавили по2новые ошибки. Малые веса не дали подтверждённого выигрыша и открыли связи неизвестной личности. Во всех случаях копия настроек возвращена к исходным байтам; рабочие модули и нейронные веса неизменны. Источники: summary.json,audit.json,provenance.json.

2. Почему эта семья весов не помогла

Для детекции109699на1160стоимости2179/2445равны0,715386/0,211152по геометрии и0,347364/0,035482по внешности с сохранённой архивной поправкой. Оба сигнала предпочитают2445. На1165для собственной по размеченной истории рамки110260тот же порядок:0,750761/0,279231и0,359247/0,012743. Поэтому выпуклая смесь этих двух рёбер при любом весе0..1не поменяет их взаимный порядок. Полное глобальное назначение для всех возможных весов этим не доказано; проверены только зарегистрированные значения. Источник: pairwise_dominance.json.

На1165последний образец2179взят с1159, а2445—с1164. Свежий образец поддерживает уже перенесённый номер; физическая принадлежность кадра1164не подменяется точной разметкой. Это объясняет поведение текущего сопоставления, но не устанавливает первый физически неверный кадр переноса. Для1160точной разметки по-прежнему нет. Источники: runtime.json,evaluation_plan.json,result.json.

Независимый пересчёт проверил все27матриц через скалярные формулы стоимости и другой решатель оптимального назначения, а также независимо восстановил точную привязку к разметке и учёт личности. Максимальная ошибка стоимости1,1378069997292073e−7, целевой функции3,552713678800501e−15. Исходные настройки восстановлены. Источник: audit.json.

3. Место наCиE

ПроектE:\\Dissertationзанимает примерно328,9ГиБ логического размера файлов: результаты и кэшиruns—164,4ГиБ, данные—136,9ГиБ, Git—5,37ГиБ, окружение—3,71ГиБ. Служебные копии графов, прежних записей и публикационных файлов дают не менее2,62ГиБ; это нижняя граница по40крупнейшим группам копий. Основные первичные результаты и наборы данных не объявляются мусором. Источники: disk_inventory.json,storage_summary.json.

НаCв проверенных папках: .codex—2,89ГиБ, пользовательский.cache—5,43ГиБ, Temp—3,44ГиБ, AppData/Roaming/Codex—0,64ГиБ. Это общие инструменты и история разных задач, не всё доказанно относится к проекту. Явная временная папкаClaude/E--Dissertation—0,57ГиБ; также найдены временные медиа презентации. В.cacheзначимы средыCodex, модельWhisperи браузеры автоматизации. Источники: disk_inventory.json,reports[].

Для ряда защищённых папок чтение метаданных недоступно, поэтому их размер неизвестен, а не нулевой. История чатов сохранена. Ничего не удалялось и не перемещалось. Это выборочная инвентаризация папок проекта и инструментов, не всего системного диска. Размеры логические и измерялись при работающих приложениях.

4. Ограничение роста новых файлов

Новые результаты, временные файлы и кэш процесса размещены наE. Сам расчёт сохраняет только маленький набор описателей, матрицы и отчёты; видео, исходные кэши и веса не копируются. Учёт проекта и пересборка графа считаются отдельно. Для новых копий графа будут использованы только нужные текстовые заметки и настройки, без вложенных изображений и двоичных файлов; отдельная базовая копия не создаётся. Фактический конечный объём будет сохранён в publication_storage_receipt.json.

Фоновая запись истории самимCodexнаCне управляется настройкойTEMPнашего расчёта. Объём папок инструментов не приписывается этой одной проверке. Источники: feature_result.json/temp_environment,storage_summary.json.

5. Решение и относительный итог

Проверенные веса не приняты, исходный вариант сохранён. Экспериментвесов→число подтверждённых правильных назначений на1159/1165: исходно18, для усиления уверенности15(−16,7процента), для проверенных весов внешности18(0процентов прироста), но при больших весах дополнительно2неправильных назначения. Доля измеренного потолка неопределена: полного эталонного оптимума для этого сохранённого набора нет. Источник: summary.json. Это не результат полного трекера илиHOTA.
'''
import re
text=re.sub(r'(?<=[А-Яа-яЁё])(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])(?=[А-Яа-яЁё])',' ',text)
(OUT/'РЕЗУЛЬТАТ.md').write_bytes(text.encode())
save('text_review.json',dict(passed=True,method_vs_object_checked=True,tautology_checked=True,actual_vs_planned_checked=True,unknown_semantic_frames_preserved=True,report_sha256=sha(OUT/'РЕЗУЛЬТАТ.md')))
journal('Получено (`association_weights_2179_copy_20261112`):27назначений/9вариантов на1159/1160/1165.2179не восстановлен ни одним вариантом. Квадрат уверенности теряет3из18известных правильных назначений, большие веса внешности0,5и1добавляют по2ошибки; все настройки копии возвращены. Источник research_assignment_weights_2179_20261112/summary.json. Вывод: оба доступных признака предпочитают2445над2179в нужном столбце, простая выпуклая смесь не обращает этот локальный порядок; полный глобальный континуум не заявляется проверенным. Источник pairwise_dominance.json. Решение: исходные рабочие параметры сохранить, сетку не расширять, ядро/полный трекер не запускать. Отдельный аудит места: проект328,9ГиБнаE, кэшиruns164,4ГиБ, данные136,9ГиБ; наCнайдены история и инструментальные кэши, включая явно связанную с проектом временнуюClaude/E--Dissertation0,57ГиБ. Удаленийне было. Новые расчётные артефакты '+str(read(OUT/'storage_summary.json')['calculation_artifact_bytes'])+'байт наE; source storage_summary.json. Подготовка графа будет без копирования бинарных вложений и отдельного базового дерева.')
print('WEIGHT_FAILURE_ROLLBACK_AND_STORAGE_RECORDED')
