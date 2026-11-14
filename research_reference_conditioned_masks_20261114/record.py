from common import *
import shutil
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
from data import load_record,NEAREST,SUPPORT
assert all(read(OUT/n)['passed'] for n in ['audit.json','capacity_audit.json','diagnostics_audit.json'])
s=read(OUT/'summary.json');p=read(OUT/'protocol.json')
report='''Выделение машины по прошлому образцу: результат и границы

Основной заранее заданный линейный считыватель с регуляризацией 1 не прошёл проверку качества. На 2179 полнота цели составила 0,7705, но доля ложного выделения известных нецелевых пикселей внутри рамки — 0,3419 при ограничении 0,1. Среднее пересечение масок, IoU, по трём запросам составило 0,2730 для FlyVis и 0,1838 для согласованного контроля по входу без FlyVis. Однако геометрия дала 0,2757, цвет с геометрией — 0,3033. Источник: summary.json и result.json. Преимущество над одним контролем не означает прохождения всей проверки.

Найдена ошибка оценки: в двух остальных запросах внутри рамки нет размеченных отрицательных пикселей. Исходный вычислитель заменял нулевой знаменатель единицей и выдавал ноль. Это неопределённая величина, а не отсутствие ошибок. Исходный result.json сохранён; audit.json и summary.json явно помечают эти случаи как непроверяемые. Следующие диагностические результаты используют null. Общий отрицательный исход не изменяется.

Причина части ошибки считывателя подтверждена на 2179: из 563 чистых фоновых ячеек исходный линейный вариант отнёс 96 к цели. При снятии регуляризации таких ячеек стало 36; небольшой нелинейный считыватель правильно разделил все 598 чистых обучающих ячеек. На следующем кадре ошибки остались: IoU нелинейного FlyVis составил 0,1714, ложное выделение внутри рамки 0,1502. Следовательно, неспособность простого считывателя описать прошлый образец объясняет часть ошибки, но не весь перенос. Нелинейная проба одновременно меняет семейство функций и функцию потерь; приписывать эффект только нелинейности нельзя. Источники: capacity_result.json, capacity_audit.json.

Дискретная сетка сама по себе не объясняет плохой результат 2179. Оптимальная по известной разметке маска на той же сетке могла бы дать IoU 0,8696. Эта верхняя оценка использует правильные ответы текущего кадра и не является рабочим методом. Источник: capacity_result.json и сертификат в capacity_audit.json.

Цвет даёт дополнительный сигнал на 2179: при одинаковом нелинейном считывателе вариант RGB дал IoU 0,2531, а переобученный вариант с удалённым цветом — 0,1843; доля ложного выделения выросла с 0,0720 до 0,2283. На других двух случаях такой устойчивой прибавки нет. Удаление цвета из проверенного входного пути до ядра подтверждено исходниками и 703 совпавшими сохранёнными входными векторами. Цветовая диагностическая ветвь не воспроизводит весь путь Lamina и не доказывает, что цвет объясняет все ошибки ядра. Источники: color_result.json, diagnostics_audit.json, alignment_audit.json предыдущей проверки входа.

Добавление текущей активности FlyVis к цвету в одинаковый считыватель из 76 входов ухудшило результат на всех трёх запросах относительно нулевых каналов FlyVis. На 2179 IoU снизился с 0,3029 до 0,2352. Это измеренный отрицательный результат данного объединения признаков, а не доказательство отсутствия полезной информации в активности. Источник: color_result.json.

Подстановка прошлой активности в уже обученный считыватель резко улучшила результат 2179: IoU 0,6935 вместо 0,2352. Но объясняющий контроль показал сильную альтернативу: простое копирование прошлой ручной маски без чтения нового кадра даёт 0,8722, а фиксированный код координат с цветом — 0,7848. Следовательно, высокая точность с прошлым образцом здесь может возникать из сохранения положения. Растровая ручная маска имеет более высокое разрешение, чем сеточная верхняя оценка, поэтому 0,8722 и 0,8696 не противоречат друг другу. Источники: tensor_result.json и position_result.json.

Три первичных пересчёта подтвердили коэффициенты, прогнозы и показатели; diagnostics_audit.json воспроизводит 66 диагностических прогнозов с максимальным отклонением около 0,000000507. Исходные входы и веса не изменены. Один ночной образец оставлен непригодным: в нём только одна чистая целевая ячейка при неизменном минимуме три. Источники: audit.json, capacity_audit.json, diagnostics_audit.json, result.json → readiness.

Решение: основная проверка завершена отрицательно. Рабочая система не меняется. Серия диагностик этой реализации завершена; следующий практический шаг — перенос прошлой маски при изменившемся перекрытии с обязательным сравнением с её простым копированием. Нужны также автоматическое получение начальной маски и проверка на других эпизодах. Сохранение личности на последовательности и HOTA не измерялись; полный трекер и ядро FlyVis не запускались. Три пары и ручная инициализация не доказывают обобщения на новые видео.
'''
(OUT/'РЕЗУЛЬТАТ.md').write_text(report,encoding='utf8')
# Render measured masks, never illustrations of an assumed successful output.
color=np.load(OUT/'color_scores.npz');pos=np.load(OUT/'position_scores.npz');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
for c in [x for x in p['cases'] if x['primary']]:
 ref=load_record(next(r for r in p['records'] if r['id']==c['reference']));q=load_record(next(r for r in p['records'] if r['id']==c['query']))
 variants=[('Текущий кадр',None),('Цвет, без FlyVis',color[c['id']+'__ZeroFlyRGB76__query'].argmax(1)[NEAREST]==1),('Цвет + FlyVis',color[c['id']+'__FlyRGB76__query'].argmax(1)[NEAREST]==1),('Копия прошлой маски',ref['truth']==1)]
 sheet=Image.new('RGB',(1564,473),'#20232a');draw=ImageDraw.Draw(sheet)
 for j,(label,m) in enumerate(variants):
  im=cv2.cvtColor(q['image'],cv2.COLOR_BGR2RGB)
  if m is not None:
   m=m&SUPPORT;im[m]=(im[m]*.45+np.array([30,230,60])*.55).astype('uint8')
  sheet.paste(Image.fromarray(im),(j*391,40));draw.text((j*391+6,10),label,font=font,fill='white')
 draw.text((8,438),'Зелёное — выделенная цель. Последняя панель не использует текущий кадр. Разметка частичная.',font=font,fill='white');sheet.save(OUT/(c['id']+'_final_controls.png'))
cat=read(ROOT/'experiments/project_history.json');card=next(c for c in cat['experiments'] if c['id']=='reference_conditioned_masks_20261114')
card.update(state='completed_negative',done='2026-11-14',obtained='Основной считыватель не прошёл качество: 2179 insideFPR=0,3419; средний IoU FlyVis/вход/геометрия/RGB=0,2730/0,1838/0,2757/0,3033 (research_reference_conditioned_masks_20261114/summary.json). В двух запросах внутри рамки нет известных отрицательных пикселей, FPR неопределён. Последующие диагностические результаты и их ограничения: РЕЗУЛЬТАТ.md; capacity_result.json; color_result.json; tensor_result.json; position_result.json.',interpretation='Подтверждены недостаточная выразительность/регуляризация исходного считывателя на прошлом образце, вклад цвета на2179 и ошибка нулевого знаменателя. Перенос на следующий кадр остаётся проблемой. Высокий результат прошлых признаков объясним позиционной памятью: копия прошлой ручной маски на2179 IoU0,8722 (position_result.json).',decision='Завершить диагностическую серию этой реализации; не переносить её в рабочий трекер. Далее проверять рабочий перенос маски при изменении перекрытия и автоматическую инициализацию. Сохранить исходный отрицательный критерий.',cannot_claim='Все причины ошибок, общее отсутствие информации в FlyVis, сохранение ID/HOTA, независимый успех на новых видео, автоматическое получение начальной маски. Последующие проверки являются объясняющими, не заменяют первичный исход.')
for a,v in zip(card['acceptance'],[False,True,True,True]):a.update(met=v,evidence='research_reference_conditioned_masks_20261114/summary.json; audit.json; result.json → readiness')
card['evidence'] += [dict(kind='file',ref=rel(OUT/n)) for n in ['РЕЗУЛЬТАТ.md','summary.json','audit.json','capacity_audit.json','diagnostics_audit.json','position_result.json']]
(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
topics=read(ROOT/'experiments/obsidian_experiment_topics.json');topics['topics'][card['id']]=['Сохранение личности','Содержание области наблюдения','Маска по прошлому образцу'];(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
addition='''**6.17. Нулевой знаменатель не означает отсутствие ошибок.** Если в оцениваемой области нет размеченных отрицательных пикселей, доля ложного выделения не определена. Записывать null и число наблюдений; такой случай не проходит критерий разделения. Обнаружено при reference_conditioned_masks_20261114, источники: research_reference_conditioned_masks_20261114/audit.json и summary.json.

**6.18. Перенос маски сравнивается с её копированием.** Хорошая маска на близком следующем кадре может отражать сохранение положения, а не узнавание объекта. При ручном прошлом образце обязательно измерять копию прежней маски без чтения нового кадра и отдельно проверять изменение перекрытия или положения. Обнаружено при reference_conditioned_masks_20261114, источник: research_reference_conditioned_masks_20261114/position_result.json. Ручная инициализация и автоматическое получение первой маски — разные задачи.

'''
anchor='## 7. Действующий план\n';save('method_addition.json',dict(anchor=anchor,addition=addition));plan=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';raw=plan.read_bytes();text=raw.decode();nl='\r\n' if '\r\n' in text else '\n';assert text.count(anchor.replace('\n',nl))==1;plan.write_bytes(text.replace(anchor.replace('\n',nl),(addition+anchor).replace('\n',nl),1).encode())
journal('Завершена reference_conditioned_masks_20261114: основной критерий качества не выполнен; подтверждены ошибка нулевого знаменателя, ограничение исходного считывателя, частная польза цвета на2179 и сильный контроль сохранения положения. Копия прошлой ручной маски даёт IoU0,8722 на2179 без текущего наблюдения, источник research_reference_conditioned_masks_20261114/position_result.json. Диагностическая серия этой реализации завершена; дальше нужен перенос при изменении перекрытия. Полный трекер и ядро не запускались. В раздел6 внесены правила6.17–6.18; первичные отрицательные результаты сохранены.')
# Prepare established, narrowly scoped publication helpers.
src=ROOT/'research_depth_transfer_20261114'
for n in ['checks.py','verify_staging.py']:shutil.copyfile(src/n,OUT/n)
checks=(OUT/'checks.py').read_text(encoding='utf8');(OUT/'checks.py').write_text(checks.replace('depth_transfer_final_tests','reference_conditioned_final_tests'),encoding='utf8')
pub=(src/'publish.py').read_text(encoding='utf8').replace("identity='depth_transfer_20261114';expected='выполнено' if read(D/'result.json')['scientific_passed']","identity='reference_conditioned_masks_20261114';expected='выполнено' if read(D/'summary.json')['scientific_passed']")
pub=pub.replace("['audit.json','result.json','holdout_maps.npz','benchmark_maps.npz','evaluation_arrays.npz']","['audit.json','result.json','scores.npz','readers.npz','evaluation_arrays.npz','color_scores.npz','position_scores.npz']")
pub=pub.replace('assert bp==ap and at.endswith(bt);extra=at[:-len(bt)]',"change=read(D/'method_addition.json');assert ap==bp.replace(change['anchor'],change['addition']+change['anchor'],1) and at.endswith(bt);extra=at[:-len(bt)]")
pub=pub.replace("headplan.replace(marker,marker+extra,1)","headplan.replace(change['anchor'],change['addition']+change['anchor'],1).replace(marker,marker+extra,1)")
(OUT/'publish.py').write_text(pub,encoding='utf8')
for name,entry,args in [('checks','checks.py',[]),('publication','publish.py',['prepare'])]:save('queue_'+name+'.json',dict(runner_sha256=sha(OUT/'run_queue.py'),deadline='2026-11-14T08:55:00+03:00',inputs={rel(OUT/n):sha(OUT/n) for n in [entry,'common.py']},jobs=[dict(entry=rel(OUT/entry),args=args,timeout=600)]))
print('RECORDED_AND_PREPARED',flush=True)
