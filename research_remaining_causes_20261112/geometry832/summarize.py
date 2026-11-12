import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
for n in ['independent_audit.json','coordinate_audit.json','source_support_audit.json','factorial_audit.json']:assert read(n)['passed'],n
base=read('result.json');factor=read('factorial_result.json');supp=read('source_support_result.json')
def row(d,old_id,rep,qa,var,head,pair=None,support='common'):
 return next(r for r in d['records'] if r['old_id']==old_id and r['representation']==rep and r.get('query')==qa and r['variant']==var and r['head']==head and r['support']==support and (pair is None or r['pair']==pair))
interactions=[]
for oid in [232,70,1102,574,832]:
 for rep in ['C34','R134_temporal','retina32']:
  for qa in ['C34','R134_temporal']:
   for var in ['gt_center_saved_offset','gt_affine_saved_offset','gt_center_zero_offset','gt_affine_zero_offset']:
    for head in ['native','agreement']:
     for pair in [['terminal_hold','index'],['index','terminal_hold'],['terminal_hold','terminal_hold']]:
      A=row(factor,oid,rep,qa,'detector_affine_saved_offset',head,['index','index']);B=row(factor,oid,rep,qa,var,head,['index','index']);C=row(factor,oid,rep,qa,'detector_affine_saved_offset',head,pair);D=row(factor,oid,rep,qa,var,head,pair);interactions.append(dict(old_id=oid,representation=rep,query=qa,variant=var,head=head,pair=pair,base_score=A['score'],coordinate_only_score=B['score'],state_only_score=C['score'],combined_score=D['score'],interaction_score=None if D['score'] is None else D['score']-B['score']-C['score']+A['score'],interaction_mean_similarity=D['mean_similarity']-B['mean_similarity']-C['mean_similarity']+A['mean_similarity']))
retinaerr=0.
for r in factor['records']:
 if r['representation']=='retina32':
  z=row(factor,r['old_id'],'retina32',r['query'],r['variant'],r['head'],['index','index']);retinaerr=max(retinaerr,max(abs(a-b) for a,b in zip(r['channels'],z['channels'])))
assert retinaerr==0.
summary=dict(passed=True,record_counts=dict(geometry=len(base['records']),source_support=len(supp['records']),factorial=len(factor['records'])),retina_state_invariance_error=retinaerr,interactions=interactions,conclusion='C34 geometry-only and expanded GT source support fail; GT geometry plus held reference state rescues832 at fixed heads. Temporal maps need only GT geometry. This is a bounded causal interaction in cached readout, not a unique neuronal root cause or deployable tracker repair.')
with (OUT/'summary.json').open('x',encoding='utf8') as f:json.dump(summary,f,ensure_ascii=False,indent=2);f.write('\n')
primary=next(x for x in interactions if x['old_id']==832 and x['representation']=='C34' and x['query']=='C34' and x['variant']=='gt_center_saved_offset' and x['head']=='native' and x['pair']==['terminal_hold','terminal_hold'])
refonly=next(x for x in interactions if x['old_id']==832 and x['representation']=='C34' and x['query']=='C34' and x['variant']=='gt_center_saved_offset' and x['head']=='native' and x['pair']==['terminal_hold','index'])
report=f'''1. Установленный результат.

У случая832 обнаружено взаимодействие двух узлов полного модуля FlyVis: переноса координат между наблюдениями и состояния ответа на исходное изображение. Ни замена координат отдельно, ни удержание изображения отдельно не исправляют решение исходной головыC34. Их совместное вмешательство исправляет её решение на неизменных149исходных ячейках. Проверены360строк первого опыта,270строк проверки исходной опоры и1200строк факторной проверки; все исходные данные, веса,семь геометрических признаков головы и нулевой порог сохранены. Источники: result.json, source_support_result.json, factorial_result.json в research_remaining_causes_20261112/geometry832/.

2. Геометрия и отрицательный первичный критерий.

Raw-кадры (номера кадров исходного видео)534 и540 соответствуют кадрам разметки90 и91; в обоих найден объект00118997. Для149из149исходных ячеек перенос по центрам рамокGT (эталонной разметки) требует поправку поxот−12,2661до−11,6966пикселя, а перенос долейGT-рамки — от−12,3173до−10,7556. Весь исходный диапазон поиска ограничен±4,25575пикселя. Следовательно, эти два заданныхGT-переноса недоступны даже непрерывному поиску внутри ограничивающего прямоугольника; это доказано для всей области, а не только19сдвигов сетки. Перенос долейGT-рамки не является доказанным соответствием физических частей объекта. Источник: coordinate_audit.json, evidence[old_id=832].

Первичный критерий спасенияC34 одними координатами не выполнен: native-голова (исходная голова решения)−0,398459784→−0,685717186 приGT-центре и→−0,671422393 приGT-аффинном переносе; порог остаётся0. Расширение исходной опоры с149до183ячеек добавило34ячейкиGT-объекта, ноC34 также остаётся отрицательной:−0,705089678 и−0,660006315 соответственно. Поэтому нехватка исходной площади не является достаточным объяснением этого отказа. Источники: result.json и source_support_result.json, old_id=832, representation=C34, head=native; для первого опыта query=C34,support=common, для второго support=reference_gt.

3. Совместная проверка состояния и координат.

Для832, C34, исходногоC34-сдвига и native-головы четыре согласованных условия дали: исходная геометрия/исходные состояния−0,398459784;GT-центр/исходные состояния−0,685717186;исходная геометрия/удержание обоих концов−0,598086542;GT-центр/удержание обоих концов+0,173103824. Взаимодействие по разности разностей равно{primary['interaction_score']:.12f};по среднему пространственному коэффициенту корреляции{primary['interaction_mean_similarity']:.12f}. ПриGT-аффинном переносе совместный score (выход головы до порога)+0,185689161. Дляagreement-головы (головы согласованности средних и масштаба) совместные значения+0,815062175 и+0,840279103. Положительное взаимодействие есть и до головы решения, поэтому эффект нельзя объяснить одной сменой её проекции. Источники: factorial_result.json и summary.json, interactions.

Одного удержания исходного конца вместе сGT-центром достаточно: native+0,135427472 иagreement+0,277410675; при удержании только текущего конца native−0,558172440. Это сужает локализацию до исходного состояния в совместной проверке; отдельный нейрон или единственная причина не установлены. Удержание было ранее сохранённым синтетическим повторением конечного изображения в течение1с, а не новым реальным видеонаблюдением и не доказанным равновесием. Retina32 (подготовленный вход глаза в32-битной точности) не меняется между условиями состояний: максимальная разность каналов0. Источники: factorial_result.json и summary.json, retina_state_invariance_error.

4. Контрольные случаи и границы результата.

ПриGT-центре и удержании только исходного концаC34 принимает70,1102,832, отвергает чужую574, но по-прежнему отвергает свою232:4из5правильных локальных решений вместо исходных2из5. При удержании обоих концов теряется своя1102:3из5. Поэтому удержание обоих концов не является общей полезной заменой. Оба варианта состояний были включены до расчёта, и выбирать один по этим пяти случаям как готовый алгоритм нельзя. Источник: factorial_result.json,representation=C34,query=C34,variant=gt_center_saved_offset,pair=[terminal_hold,index] либо[terminal_hold,terminal_hold].

Дляобычных временных картR134temporal достаточно однойGT-геометрии: на исходномR134-сдвиге native832−0,150395584→+0,195296409,agreement−0,833304268→+0,929215411 приGT-аффинном переносе. В этом заранее включённом вторичном вариантеagreement сохраняет принятие232,70,1102 и отвергает чужую574(−0,696090001):5из5против исходных3из5. Native оставляет574положительной(+0,024810295):4из5. Это диагностическая проверка по известнымGT-рамкам, не рабочий алгоритм получения координат без разметки. Источник: result.json,representation=R134_temporal,query=R134_temporal,variant=gt_affine_saved_offset,support=common.

5. Проверка и относительные величины.

Независимые пересчёты используютSciPy-интерполяцию (восстановление значений между узлами) и скалярные формулы каналов и голов; все ошибки меньше10⁻⁹. Независимая скалярная проверка координат дала максимальную ошибку0. Общими остаются сохранённые исходники карт и разметка, а динамика ядра заново не вычислялась. Источники: independent_audit.json,coordinate_audit.json,source_support_audit.json,factorial_audit.json.

Сводка относительно исходного сравнения: отрицательный первичный геометрический тестC34 —0исправленных решений832;GT-геометрия+удержание только исходного концаC34 —40%→80%правильных решений на этих пяти случаях,+40п.п.,+100%относительно исходных40%;GT-аффинный перенос с обычными временными картами иagreement —60%→100%,+40п.п.,+66,67%относительно исходных60%. Относительные проценты знакопеременных выходов головы не имеют содержательного смысла; измеренного независимого потолка, полного восстановления траектории иHOTA здесь нет.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write(report)
print('SUMMARY',len(interactions),primary,refonly)
