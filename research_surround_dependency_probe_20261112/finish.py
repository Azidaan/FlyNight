from common import *
import shutil,psutil
a=read(OUT/'audit.json');scores=read(OUT/'scores.json');base=read(OUT/'baseline_gate.json');assert a['passed'] and a['ordinary_controls_exact'] and not a['primary_any_both_gate'] and not scores['decision_changes']
model=read(OUT/'loaded_model.json');guardpath=Path(model['supervisor']['supervisor_report']);guard=read(guardpath);assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors'];owned=[]
for pid,birth in guard['owned'].items():
    try:same=psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:same=False
    assert not same;owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
save('execution_receipt.json',dict(passed=True,guard=guard,all_owned_ended=True,process_checks=owned));shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/'worker.log');rel=OUT.relative_to(ROOT).as_posix()
obtained=('На пяти случаях проверены восемь сочетаний вмешательств и три представления, 120 оценок. Обе обычные оценки во всех условиях совпали точно с исходными. Решения FlyVis также сохранились: две свои пары приняты, одна чужая принята. '
 'Для232 исходный балл −1,452065; серое окружение у обоих концов −1,376530, замороженное −1,327592. Для832: −1,396481 → −1,383704 / −1,363531; для574: +1,843682 → +1,756685 / +1,866990. '
 'Основной критерий исправления232 и вторичный критерий исправления всех трёх ошибок не пройдены. Источники '+rel+'/scores.json и audit.json.')
interpret=('Изменение только входов вне точных зависимостей обычных признаков меняет оценку FlyVis при неизменности обычных оценок. Это причинный эффект окружения через нейронную обработку, но ни один вариант не исправил решения. '
 'Проверенные замены внешней области не являются достаточным исправлением232. Они не исключают влияние окружения внутри защищённой области или иных способов взаимодействия. Защищённая область не является физической маской машины.')
decision=('Не внедрять отсечение или замораживание окружения, не подбирать радиус и заполнитель по этим случаям. '
 'Следующая проверяемая версия — изменение координат входного окна до рекуррентной обработки; наблюдение о разных вырезках сохранено отдельно и не объявляется причиной. Нужен контроль входа в одних координатах при неизменных конечных окнах и сохранении полезного движения.')
cannot=('Не установлены конкретный виновный нейрон, дефект весов или корневая причина всех отказов. Серый заполнитель может менять адаптацию и создавать искусственную границу. '
 'Сохранённые области определены вычислительными зависимостями, могут содержать фон и исключать части объекта. Пять повторно использованных случаев не являются независимым итоговым тестом. Нет улучшения ID или HOTA; трекер не запускался.')
maxmap=max(v for r in base['records'] for v in r['parity_errors'].values())
lines=['# Влияние окружения при неизменных обычных признаках','','1. Результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Граница вмешательства','',
 'В образце защищены все непосредственно считываемые ячейки. В кандидате защищены вершины всех треугольников интерполяции для сохранённых сдвигов трёх представлений. К обоим наборам добавлены соседи, от которых зависит обычный пространственный контроль. Эти входные ячейки сохранены на протяжении всей истории. Все остальные заменены серым значением 0,5 либо первым значением соответствующей ячейки.','',
 'Для232 сохранены141ячейка у образца и162у кандидата. Остальные случаи:70 —195/307;1102 —142/177;574 —168/280;832 —199/277. Источник protocol.json, cases.protected. Эти числа описывают вычислительную опору и не являются площадью видимой машины.','',
 'История изменена отдельно у образца, отдельно у кандидата и у обоих; исходный и неизменяющий варианты составляют контроли. Обычные карты за пределами опоры могут измениться, но непосредственно сравниваемые обычные признаки и итоговые оценки должны совпадать точно. Это требование проверено независимо.','',
 '3. Надёжность проверки','',f'Все исходные случаи проверены до вмешательств. Максимальное отличие исходных карт ядра от архива {maxmap:.9g}, допуск0,0001; обычные исходные карты совпали точно. Неизменяющий контроль точный, веса и входные файлы неизменны. Источники baseline_gate.json, audit.json, execution_receipt.json.','',
 'Перед запуском обнаружено и исправлено несоответствие унаследованного выражения вторичного критерия: оно требовало хотя бы одну свою232/832 вместо обеих. Исправление сделано в отдельном audit_secondary_v2.py; исходный протокол и оценщик сохранены. Условие протокола не менялось, новых выходов на момент исправления не было. Источник protocol.json, implementation_correction_before_replay.','',
 'Независимая проверка заново перечисляет вершины интерполяции, строит защищённую область и входы по каждой ячейке; использует другой интерполятор, формулу согласия через средний квадрат различий и скалярное суммирование. Общие карты и слои раскрыты; динамика ядра независимой реализацией не проверена.','',
 '4. Все оценки FlyVis','']
for r in scores['records']:
    if r['representation']=='C34':lines += [f"Номер {r['old_id']}; вмешательства у образца и кандидата {r['mode']}: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines+=['5. Наблюдение для следующего различающего контроля','',
 'Архивные входные окна меняют положение и размер. У232 перед конечным образцом окно меняется с[957,507,1040,590] на[956,507,1040,591], тогда как последние поисковые окна одинаковы. Код подаёт соседние входные массивы по одинаковым индексам сетчатки; отдельного приведения к неподвижным координатам перед рекурсией в этом пути нет. Это наблюдение о подготовке входа, не доказательство ошибки. Источник crop_observation.json; изменения вырезки не равны движению машины.','',
 '6. Литературное основание','',
 'Lappalainen и соавторы (2024), Connectome-constrained networks predict neural activity across the fly visual system, https://doi.org/10.1038/s41586-024-07939-3. Раздел о пространственных и временных рецептивных полях и описание гексагональной свёрточной сети обосновывают проверку входных зависимостей за пределами непосредственно сравниваемых колонок. Статья не доказывает причину наших ошибок.','',
 'Изменение числа правильных решений:0 % относительно исходного условия. Исправлено0из3ключевых ошибок (0 %). Измеренный потолок полного трекера отсутствует.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,False]):item.update(met=value,evidence=rel+'/audit.json')
c['evidence'] += [dict(kind='file',ref=rel+'/'+n) for n in ['scores.json','audit.json','replay.json','baseline_gate.json','execution_receipt.json','crop_observation.json','РЕЗУЛЬТАТ.md']]
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1');problem['cases'].append(dict(date='2026-11-12',component='Окружение входных колонок полного модуля FlyVis',fact=obtained,hypothesis='Влияние окружения на баллы подтверждено, его достаточная роль в ошибке232 не подтверждена. Изменение входных координат остаётся отдельной непроверенной версией.',discriminating_check=decision,limitation=cannot,evidence=[rel+'/audit.json',rel+'/scores.json',rel+'/crop_observation.json'],source_sha256=sha(OUT/'audit.json')));problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);s=next(x for x in d['sources'] if x['id']=='lappalainen2024_recurrent_history_control')
for field,extra in [('application','\n\n12.11.2026. Дополнительно проверены входные зависимости вне области обычного считывания: две замены окружения меняют баллы FlyVis при точном сохранении обычных оценок, но не исправляют решения. Роль источника — устройство пространственных связей и рецептивных полей.'),('sections',' Дополнительно проверены раздел Spatio-temporal, spatial and temporal receptive fields и описание гексагонального свёрточного представления.')]:
    s.setdefault('revisions',[]).append(dict(date=now(),field=field,old=s[field],reason='Добавлена фактическая роль первоисточника в проверке окружения; прежний текст сохранён.'));s[field]+=extra
s['evidence'] += [rel+'/protocol.json',rel+'/audit.json'];s['related'].append('exp:'+ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Порог, веса и трекер не менялись. Отдельное наблюдение о вырезках: '+rel+'/crop_observation.json.')
print('RECORDED_NEGATIVE OWNED_WORKERS_ENDED')
