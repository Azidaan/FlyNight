import json,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'result.json').read_text(encoding='utf8'));rows=r['records'];summary=[];oracle=[]
for c in r['controls']:
 z=next(x for x in rows if all(x[k]==c[k] for k in ['old_id','representation','head','mode']));random=[x for x in c['scores'][c['cyclic_count']:] if x is not None];cyclic=[x for x in c['scores'][:c['cyclic_count']] if x is not None]
 summary.append({k:c[k] for k in ['old_id','representation','head','mode','count','parent_count']}|dict(score=z['score'],valid_random=len(random),valid_cyclic=len(cyclic),random_median=statistics.median(random) if random else None,random_fraction_lower=sum(x<z['score'] for x in random)/len(random) if random and z['score'] is not None else None,random_rejected=sum(x<=0 for x in random)))
for x in r['oracle_records']:
 if not x['is_own']:continue
 y=next(y for y in r['oracle_records'] if all(y[k]==x[k] for k in ['representation','head','mapping','support']) and not y['is_own']);oracle.append({k:x[k] for k in ['representation','head','mapping','support','count']}|dict(own_score=x['score'],false_score=y['score'],own_minus_false=None if x['score'] is None or y['score'] is None else x['score']-y['score']))
s=dict(control_summaries=summary,oracle_differences=oracle,acceptance=dict(original_and_identity_parity=True,independent_audit=True,exclusive_support_causal_correction=False,physical_ownership_proven=False,mapping_robust_C34_order=False),conclusion='Other-instance rectangle overlap is observed, but exclusive-center support drops to2points at original query and1/2points at center/affine oracle queries. Cannot estimate its contribution to falsepermission. DualGT retains the falsepermission. C34 true-vs-false rank reverses between center and affine GT mappings; no mapping-robust C34identity conclusion. R134spatial favors true at both oracle mappings. No deployment improvement established.',cannot_claim=['pixel ownership','unique core defect','overlap caused falsepermission','deployable GToracle correction','HOTA gain','actual ID switch'],baseline_change=dict(metric='574 C34native falsepermission count on sole archived foreignpair',baseline=1,dual_gt=1,relative_change_percent=0,exclusive=None,ceiling_fraction=None,reason='Exclusive score undefined, no comparable physical-cause ceiling; full tracker not evaluated.'))
with (OUT/'interpretation.json').open('x',encoding='utf8') as f:json.dump(s,f,ensure_ascii=False,indent=2)
text='''1. Результат рамочной проверки

На сохранённой паре574 ограничение обеих конечных точек собственными аннотированными рамками не устранило ложное разрешение. Для C34 и исходного считывателя оценка изменилась с0,5463815125420641 при120точках до0,5825807562695111 при42точках. После исключения точек, попадающих в рамки других экземпляров хотя бы на одном конце, осталось2точки. По заранее заданному минимуму3 результат неопределённый. Удаление перекрывающихся областей поэтому нельзя назвать исправлением ошибки или доказательством её причины. Первичный источник:result.json,records; краткая сводка:interpretation.json,control_summaries.

Для ограничения двумя рамками199случайных контролей с теми же42точками сохранили положительную оценку C34. Медиана0,5500662703217201; оценка рамочной пробы выше140из199контролей. Для эксклюзивной опоры все контроли имеют те же2точки и тоже не оцениваются. Это отделяет отсутствие измерения от положительного результата.

2. Сравнение истинного и чужого партнёров

Истинный экземпляр00115019 и чужой00115011 присутствуют в одном сохранённом текущем глазе. При одинаковых84исходных точках и неизменных7геометрических признаках C34 зависит от правила соответствия. Перенос по центрам рамок даёт истинному партнёру−0,022700939670518117, чужому+0,14757397034632613. Аффинное соответствие нормированных координат рамок даёт истинному+0,010655184930704942, чужому−0,09841905406700868. Значит, C34 содержит различающий сигнал при одном из заданных соответствий, но устойчивый к этим двум правилам порядок не подтверждён. Первичный источник:result.json,oracle_records; все разности:interpretation.json,oracle_differences.

Обычное пространственное представление R134_spatial ставит истинного партнёра выше чужого при обоих правилах; временное R134_temporal меняет порядок вместе с правилом соответствия. На общей эксклюзивной опоре остаётся1точка при переносе по центрам и2при аффинном преобразовании. Оценки по ней не вычисляются.

3. Граница вывода

Наложение всех ближайших аннотированных рамок сохранено вnative_context_574.png; исходные кадры показаны рядом без дорисовки. Аннотации помечают GT00115010,00115011 и00115019 как частично закрытые другими объектами на обоих концах. Прямоугольники не задают видимые маски; исключение центров также не исключает смешивание в рецептивном поле. Поэтому физическая принадлежность похожих фрагментов и уникальная причина ложного разрешения остаются непроверенными.

Независимая проверкаaudit.json пересчитала18508оценок, максимальное расхождение1,7763568394002505e−15. Нейронная сеть, трекер, веса, пороги и производственный код не менялись. Это проверка считывания сохранённых карт, а не HOTA или качества идентификаторов полного трекера. Число ложных разрешений исходного C34 на единственной рассматриваемой чужой паре:1→1, изменение0%; для эксклюзивной пробы число не определено из-за недостаточной опоры. Доля измеренного физического потолка не определена.
'''
with (OUT/'РЕЗУЛЬТАТ_ОСНОВНОЙ.md').open('x',encoding='utf8') as f:f.write(text)
