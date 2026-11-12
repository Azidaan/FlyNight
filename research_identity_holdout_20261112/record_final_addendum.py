from common import *
import psutil,re
r=read(OUT/'recovery/classmask_result.json');q=read(OUT/'queue5c_execution.json');assert q['passed']
assert r['original_baseline_reproduced'] and r['surviving_compatible_costs_unchanged'] and r['no_GT1160_claim']
for record in r['records']:assert record['objective_error']<1e-9
p=read(OUT/'recovery/classmask_protocol.json')
for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
assert sha(OUT/'recovery/classmask_probe.py')==p['worker_sha256']
for job in q['jobs']:
    for pid,birth in job['result']['owned'].items():
        try:alive=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess:alive=False
        assert not alive,pid
metadata=read(OUT/'recovery/native_class_metadata_A.json');allpairs=[pair for e in metadata['event_records'] for pair in e['saved_actual_first']]
assert len(allpairs)==metadata['actual_first_pairs']==611
assert sum(x['cross_class'] for x in allpairs)==metadata['cross_class_pairs']==6
for n,h in metadata['inputs'].items():assert sha(ROOT/n)==h,n
save('audit_addendum.json',dict(passed=True,checked=now(),scientific_result_unchanged=True,class_probe_sha256=sha(OUT/'recovery/classmask_result.json'),metadata_sha256=sha(OUT/'recovery/native_class_metadata_A.json'),checks='Allregisteredinputs/sources,exactbaseline,unchangedcompatiblecosts,independentsolverobjectives,ownedworkersended,611pairrecords/6classmismatches. ClassmetadataisnotGTidentityerror.',original_audit_sha256=sha(OUT/'audit.json')))
base=OUT.relative_to(ROOT).as_posix();cat=read(ROOT/'experiments/project_history.json');c=next(c for c in cat['experiments'] if c['id']=='identity_class_compatibility_diagnostic_20261112')
c.update(state='inconclusive',done='2026-11-12',obtained='Проверка классов отдельно не меняет владельца109699:остаётся2445,2179без назначения. Только сочетание удаления2445с проверкой класса передаёт109699к2179; удаление2445без проверки класса передавало его2316другого класса. Все совместимые стоимости и порог0,8сохранены. Источники '+base+'/recovery/classmask_result.json и assignment_loss_result_v2.json. Во всех54Aнайдено6межклассовых исходных назначений из611; это метаданные детектора, не6доказанных ошибок личности; источник recovery/native_class_metadata_A.json.',interpretation='Локализован механизм конкуренции в сохранённой геометрической матрице. Одна проверка класса не исправляет исходный отказ; сочетание с удалением активного конкурента не является допустимым рабочим исправлением.',decision='Завершить ограниченный разбор без новых вариантов, изменений классового допуска, весов или полного трекера. РезультатAсохранён, резервBне использован для выбора.',cannot_claim='Нет точной разметки1160, поэтому109699не объявляется своей рамкой2179. Не доказана полезность жёсткого классового ограничения или улучшение личности на последовательности.')
for item in c['acceptance']:item.update(met=True,evidence=base+'/recovery/classmask_result.json; '+base+'/audit_addendum.json')
c['evidence'].append(dict(kind='file',ref=base+'/audit_addendum.json'))
(ROOT/'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode())
addition='''
9. Последнее дополнение: класс конкурента не равен истинной личности

Во всех54новых событиях найдены6исходных назначений с различающимися сохранённым классом траектории и текущим классом детектора из611пар. Это не6доказанных ошибок личности: класс детектора и независимая разметка объекта различаются по смыслу. Источник: recovery/native_class_metadata_A.json.

В кадре1160после удаления2445рамку109699получал2316с сохранённым классом0, хотя у детекции класс2. Отдельная проверка совместимости классов исходное назначение не меняет:2445побеждает,2179остаётся без пары. Только сочетание этой проверки с удалением2445даёт2179→109699. Это два заранее зарегистрированных контроля, а не включённое исправление; активного конкурента в рабочей системе не удаляли. Точные стоимости совместимых рёбер и порог0,8сохранены, независимая оптимальная стоимость подтверждена. Источники: recovery/classmask_protocol.json,classmask_result.json и audit_addendum.json.

Подтверждённый механизм текущего отказа—полная конкуренция предсказанных рамок по геометрической стоимости с допуском других классов. Порог0,8само целевое ребро2179не отсекает: его стоимость0,715385795меньше порога. Но точная физическая принадлежность детекции1160и первый неверный кадр переноса номера по-прежнему не доказаны. Новых вариантов после этих двух контролей не запускали.
'''
report=OUT/'РЕЗУЛЬТАТ.md';raw=report.read_text(encoding='utf8');(OUT/'РЕЗУЛЬТАТ_до_дополнения.md').write_bytes(raw.encode());s=raw.rstrip()+'\n'+addition
# Typography only: preserve the exact non-whitespace character sequence.
clean=re.sub(r'(?<=[А-Яа-яЁё])(?=[0-9A-Za-z])|(?<=[0-9A-Za-z])(?=[А-Яа-яЁё])',' ',s)
assert re.sub(r'\s','',s)==re.sub(r'\s','',clean)
report.write_bytes((clean.rstrip()+'\n').encode())
save('text_review_addendum.json',dict(reviewed=now(),each_sentence_method_or_object_checked=True,adjacent_tautology_checked=True,actual_vs_planned_checked=True,unknown_physical_truth_preserved=True,typo_only_nonwhitespace_exact=True,previous_text_preserved='РЕЗУЛЬТАТ_до_дополнения.md',report_sha256=sha(report)))
journal('Получено последнее дополнение (`identity_class_compatibility_diagnostic_20261112`): проверка класса отдельно не меняет2445→109699; удаление2445без неё даёт2316, с ней—2179. Совместимые стоимости и порог0,8не менялись; это причинный контроль, не применённая правка и не доказательство правильной рамки1160. Во всех54Aсохранено611исходных пар,6межклассовых по метаданным детектора; они не объявлены6истинными ошибками. Источники research_identity_holdout_20261112/recovery/classmask_result.json,native_class_metadata_A.json,audit_addendum.json. Новых вариантов не запускаем. Сохранение личности на новойAостаётся неопределённым; подтверждённый прежний574зелёный. Итоговый отчёт дополнен без удаления прежней редакции; веса/полный трекер не тронуты.')
print('FINAL_BOUNDED_DIAGNOSTIC_RECORDED')
