import json,hashlib,csv,io
from pathlib import Path
OUT=Path(__file__).resolve().parent
def read(n):return json.loads((OUT/n).read_text(encoding='utf8'))
def write(n,x):
    data=json.dumps(x,ensure_ascii=False,indent=2).rstrip()+'\n'
    with (OUT/n).open('xb') as f:f.write(data.encode('utf8'))
r=read('result.json');a=read('audit.json');assert r['passed'] and a['passed']
rows=[z for z in r['records'] if z['representation']=='C34' and z['head']=='native']
compact=[]
for reg in r['registrations']:
    oid=reg['old_id'];values={z['coordinate']+'_'+z['reference_mode']:z['score'] for z in rows if z['old_id']==oid};compact.append(dict(old_id=oid,translation=reg['translation'],searched_translations=len(reg['candidate_translations']),source_cells=next(z['count'] for z in rows if z['old_id']==oid),**values))
target=next(z for z in compact if z['old_id']==832);interaction=target['census_terminal_hold']-target['census_index']-target['original_terminal_hold']+target['original_index']
out=dict(passed=True,primary_accepted=False,reason='832 repaired locally but own1102 lost and foreign574 still accepted; overall nativeC34 accuracy2/5 unchanged.',local_832_repaired=True,all5=compact,interaction832_score=interaction,baseline_accuracy=.4,repair_accuracy=.4,relative_accuracy_change_percent=0.,new_core_calls=0,gt_used_for_geometry=False,operational_recommendation='Do not deploy universally. Preserve existing active owner773, defer duplicate832; a positive head vote alone must not steal active ownership. Parent unified appearance/assignment policy must evaluate proposal under protected active pairs.',limits='No full saved-matrix assignment or public trajectory repaired by this component; assignment review is separate. Five selected cases, cached held responses, no independent generalization or HOTA.')
write('summary.json',out)
votes=[dict(case_index=z['case_index'],old_id=z['old_id'],representation=z['representation'],head=z['head'],coordinate=z['coordinate'],reference_mode=z['reference_mode'],score=z['score'],accepted=z['accepted'],count=z['count']) for z in r['records'] if z['score'] is not None]
write('votes_for_assignment.json',dict(source='result.json',votes=votes,policy='Scores are proposals, not public assignments. Keep original alpha0.2/cost gates and protect existing active owners in actual matrix replay.'))
stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n');writer.writerow(['old_id','translation_dx','translation_dy','candidates','source_cells','original_index','original_referenceheld','census_index','census_referenceheld'])
for z in compact:writer.writerow([z['old_id'],*z['translation'],z['searched_translations'],z['source_cells'],z['original_index'],z['original_terminal_hold'],z['census_index'],z['census_terminal_hold']])
with (OUT/'all5_nativeC34.csv').open('xb') as f:f.write(stream.getvalue().rstrip().encode('utf8')+b'\n')
text='''1. Исполнимая правка без эталонной геометрии.

Реализована одна общая замена: перенос исходного шаблона оценивается по Census-признаку сырых изображений (знакам 24 локальных сравнений яркости в окрестности 5 × 5), затем замороженная голова считывает текущую карту по этим координатам и ранее сохранённую исходную карту после синтетического удержания. Оценка переноса не использует GT (эталонную разметку), номер объекта, значения нейронной головы или вручную подставленную поправку. Generic API (общая функция для нового случая) register_translation() находится в registration.py; аргументы — два изображения, рамки и вырезки детектора, исходные точки карты и геометрия глаза.

Поиск перечисляет все целочисленные переносы, при которых полный пиксельный шаблон с окрестностями 5 × 5 помещается в текущую вырезку, а все исходные точки карты — в текущий выпуклый контур глаза. Порядок выбора фиксирован до расчёта: медиана расстояния Хэмминга (числа различающихся битов), сумма расстояний, квадрат отклонения от перемещения центров детекторных рамок, затем dx и dy. После исходов область и правило выбора не менялись. Основание для Census: R. Zabih, J. Woodfill, Non-parametric Local Transforms for Computing Visual Correspondence, ECCV 1994, DOI 10.1007/BFb0028345, https://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf . Окрестность 5 × 5 и медианное агрегирование — наши заранее фиксированные инженерные условия; статья не обещает исправление наших случаев.

2. Первичный критерий не выполнен.

На 832 реальная оценка переноса без GT равна (−6; −2) пикселя, выбрана из 688 допустимых переносов. На тех же 149 исходных точках native C34 (исходная голова по активности ядра) даёт: исходные координаты и история −0,398459784; исходные координаты и удержание образца −0,668466625; Census и исходная история −0,258058941; Census и удержание образца +0,461949902. Порог во всех четырёх условиях остаётся 0. Это устраняет локальный отказ головы 832 без GT-геометрии. Значение переноса не доказывает движение физической детали: это результат выбранного фотометрического сопоставления. Источники: result.json, records и registrations; audit.json, registration_checks.

Однако универсальная замена не принята: она теряет правильное принятие своей 1102 и продолжает принимать чужую 574. Общая доля правильных native C34 решений остаётся 2 из 5. Это заранее зарегистрированные условия отказа правки, а не второстепенные оговорки. Поиск, признаки и пороги после результата не подбирались.

3. Все пять случаев, одинаковая последовательность четырёх условий.

Ниже порядок чисел един для всех случаев: исходные координаты / исходная история; исходные координаты / удержание только образца; Census / исходная история; Census / удержание только образца. Все числа — выход одной замороженной native C34 головы до неизменного нулевого порога.

Свой 232: −0,426740283; −0,510476678; −0,661891999; −0,663714794. Перенос (5; −3), 1500 допустимых переносов, 97 исходных точек. Отказ сохраняется.

Свой 70: +0,590723643; +0,588490235; +0,389565662; +0,990283082. Перенос (0; 0), 2610 допустимых переносов, 143 исходные точки. Принятие сохраняется.

Свой 1102: +0,204126092; +0,004096874; −0,482394587; −0,488785954. Перенос (20; −2), 1737 допустимых переносов, 98 исходных точек. Правильное принятие потеряно.

Чужой 574: +0,546381606; +0,460200815; +0,657793833; +0,959217323. Перенос (−1; −1), 686 допустимых переносов, 120 исходных точек. Ложное принятие сохраняется и усиливается по выходу головы.

Свой 832: −0,398459784; −0,668466625; −0,258058941; +0,461949902. Перенос (−6; −2), 688 допустимых переносов, 149 исходных точек. Локальный отказ головы исправлен только совместным вмешательством.

Источники для всех строк: result.json, representation=C34, head=native; компактные точные числа — all5_nativeC34.csv и summary.json. Все обычные временные карты и обе головы также сохранены, всего 120 строк.

4. Действие в сопоставлении и принадлежность активному треку.

Положительный голос 832 сам по себе не восстанавливает прежний публичный номер. В сохранённом кадре 540 текущая рамка raw31468 уже принадлежит активному треку 773 того же размеченного объекта. Поэтому корректное операционное действие при запрете перехвата активного владельца — сохранить 773 и отложить дублирующую заявку 832. Насильно назначать raw31468 старому 832 на основании нового положительного голоса нельзя. Полная проверка сохранённой матрицы с прежними коэффициентом 0,2 и порогами назначения выполняется отдельным аудитором родительской политики; этот компонент её результат не подменяет. Голоса переданы в votes_for_assignment.json.

Универсальная регистрация Census не включается как безусловная замена. Её функция и отрицательный опыт сохраняются; родительская политика может рассматривать предложение координат вместе с полным внешним видом и защитой активного владельца. Это допустимое продолжение всей авторской системы, но его возможную пользу нельзя заранее приписывать нейронному ядру FlyVis.

5. Проверки и границы результата.

Независимый аудитор повторно перечислил всю область поиска и восстановил каждую стоимость с прямыми логическими Census-сравнениями вместо битовой упаковки рабочего кода. Все выбранные минимумы совпали, максимальная ошибка стоимостей и координат — 0. Интерполяция независимо проверена через SciPy, каналы и головы — скалярными формулами; максимальная ошибка 1,56 × 10⁻¹⁵. Исходное считывание сопоставлялось с теми же сохранёнными картами повышенной точности, а не с более старым побитово отличающимся повтором. Источник: audit.json.

Новые проходы ядра, обучение, полный трекер и изменение весов или порогов — 0. Использованы ранее проверенные состояния после синтетического удержания; выполнение такого пути для нового произвольного наблюдения в полном трекере этой проверкой не измерено. Пять случаев были выбраны ранее; обобщение, HOTA и независимый потолок не установлены.

Относительная сводка: Census + удержание образца — 40% → 40% правильных локальных решений, изменение 0 п.п. и 0% относительно исходных 40%; принятые свои — 2 → 2 (0%), ложные разрешения чужим — 1 → 1 (0%). Состав ошибок изменился: 832 исправлен, 1102 потерян. Процент знакопеременного выхода головы не интерпретируется как качество или выигрыш полного трекера.
'''
with (OUT/'РЕЗУЛЬТАТ.md').open('xb') as f:f.write((text.rstrip()+'\n').encode('utf8'))
write('text_review.json',dict(passed=True,checks=['Operations described as methods; archived objects and controlled outputs remain objects of evaluation.','No duplicate result sections without a different purpose; all5 parallel outputs grouped explicitly.','Every outcome from completed audited120row artifact; operational matrix/trajectory/HOTA remain separately unmeasured.'],report_sha256=hashlib.sha256((OUT/'РЕЗУЛЬТАТ.md').read_bytes()).hexdigest()))
print('FINISHED',out)
