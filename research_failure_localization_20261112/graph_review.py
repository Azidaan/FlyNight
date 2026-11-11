"""Compare the author's five tasks to the actual generated section7, preserving P1-P11."""
from common import *
g=read(ROOT/'ObsidianFlyVis/.graph-visible.json')
mapping=[dict(task=1,description='Принадлежность единственного допустимого продолжения',nodes=['fly:P8','sys:С2','sys:С3'],current='Ложные отказы 232/832 и ложное разрешение574; готовая политика порог0 не содержит отдельного состояния ожидания.'),
 dict(task=2,description='Конкуренция объектов и общее назначение',nodes=['fly:P5','sys:С3'],current='Подстановка разрешения на 29 полных матрицах; принадлежность пары не определяет победителя между номерами одной машины.'),
 dict(task=3,description='Сопоставление одной физической части при изменении рамки',nodes=['fly:P8','sys:С5'],current='Неизменяющие контроли выполнены; физических ключевых точек нет, причинная проверка соответствий не завершена. Отдельного явно названного узла для этой задачи в текущем разделе нет.'),
 dict(task=4,description='Достаточность поля наблюдения',nodes=['fly:P3'],current='В пяти нынешних шаблонах97–149ячеек; старая формулировка об1–2колонках не описывает эту геометрию. Число ячеек не равно числу независимых физических деталей.'),
 dict(task=5,description='Непрерывность личности и память',nodes=['sys:С5','sys:С2','fly:P5'],current='Изучены конкурирующие номера и история1102; полный жизненный цикл не запускался.')]
for r in mapping:r['paths']=[g['nodes'][k]['path'] for k in r['nodes']]
source_nodes={k:v for k,v in g['nodes'].items() if v['section']=='Проблемы мухи'}
save('graph_review.json',dict(created=now(),root=g['nodes']['Проблемы мухи'],mapping=mapping,
 source_nodes=source_nodes,source_hashes={v['path']:sha(ROOT/'ObsidianFlyVis'/v['path']) for v in source_nodes.values()},
 primary_observable_failure='При фиксированном пороге считыватель карт отвергает собственные232/832 и принимает чужую574. Неверные оценки воспроизведены; их физическая причина внутри зрительного пути не локализована.',
 evaluation_confusion='Не разделялись достаточно явно: принадлежность физическому объекту, выбор представителя среди нескольких ID и действие общего назначения. С3 уже содержала пример такого расхождения от07.11.',
 outside_five=['sys:С7 — динамическое запаздывание FlyVis','sys:С6 — исполнение','fly:P6,fly:P7,fly:P11 — обоснованность и область выводов'],
 boundary='Five tasks cover identity tracking branch, not every FlyVis failure. P1-P11 not rewritten; no graph restructuring.'))
# New measured evidence belongs in existing problems, not a new speculative root.
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw)
backup=OUT/'before/system_problems.json'
assert not backup.exists();backup.write_bytes(raw)
for ident in ['С1','С3','С5']:
    item=next(x for x in d['problems'] if x['id']==ident)
    item['cases'].append(dict(date='2026-11-12',component='Полный модуль FlyVis и сопряжение со всей системой',
       fact='Независимо проверены29прямых связей; правильное разрешение меняет назначение232/832 и6из12своих связей за допуском. На252у1102 и1072 одновременно однаGTцель; на258её продолжает1072. Это подтверждение проблемыС3 на новых событиях, не уникальная причина всех отказов. Источники research_failure_localization_20261112/stage1.json, stage2.json, ownership.json.',
       hypothesis='Точная физическая причина неверных зрительных оценок пока не разделена; новых причин без вмешательства не объявлено.',
       discriminating_check='Для проверки физических соответствий нужны независимые ключевые точки с погрешностью; подстановка координат рамки не заменяет эту разметку. Для261у1102 не сохранена матрица и нет прямой метки.',
       limitation='Не выполнена физическая подстановка третьего этапа; нет нового полного трекера, HOTA или улучшения ядра.',
       evidence=['research_failure_localization_20261112/РЕЗУЛЬТАТ.md','research_failure_localization_20261112/audit.json','research_failure_localization_20261112/graph_review.json'],source_sha256=sha(OUT/'РЕЗУЛЬТАТ.md')))
    if ID not in item['cards']:item['cards'].append(ID)
    item['updated']='2026-11-12'
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Сверка пяти задач автора с разделом7 графа. Пять задач относятся к сохранению личности; запаздываниеС7 и исполнениеС6 — отдельные проблемы. Основной наблюдаемый отказ текущего считывателя:232/832 отвергаются,574принимается; причина внутри зрительного пути пока не локализована. С3 уже описывала расхождение своей пары и действия назначения. Источник research_failure_localization_20261112/graph_review.json. ФормулировкиP1–P11 и структура графа не менялись.')
print('GRAPH_REVIEW_RECORDED')
