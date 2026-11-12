from common import *
import shutil,psutil
a=read(OUT/'audit.json');scores=read(OUT/'scores.json');base=read(OUT/'baseline_gate.json');assert a['passed'] and not a['primary_any_both_gate'] and not scores['decision_changes']
model=read(OUT/'loaded_model.json');guardpath=Path(model['supervisor']['supervisor_report']);guard=read(guardpath);assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors'];owned=[]
for pid,birth in guard['owned'].items():
    try:same=psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:same=False
    assert not same;owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
save('execution_receipt.json',dict(passed=True,guard=guard,all_owned_ended=True,process_checks=owned));shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/'worker.log')
rel=OUT.relative_to(ROOT).as_posix()
obtained=('На пяти известных случаях проверены восемь сочетаний историй и три представления: 120 оценок. Ни одно решение не изменилось. Во всех сочетаниях FlyVis принимает две свои пары и одну чужую; оба обычных контроля принимают три свои и одну чужую. '
 'При замене обеих предысторий первым входом / предпоследним входом оценки FlyVis: 232 −1,310997 / −1,201388; 832 −2,181159 / −1,270674; 574 +2,361382 / +2,381563. '
 'Исходные оценки соответственно −1,452065, −1,396481, +1,843682. Источник '+rel+'/scores.json, records/counts/decision_changes.')
interpret=('При неизменных первом и последних двух входах, числе обновлений и весах изменение более раннего содержимого причинно меняет оценки. '
 'Однако ни односторонние, ни совместные замены не устранили отказы 232/832 или неверное разрешение 574. '
 'Ни одна из двух фиксированных версий постоянной предыстории не прошла критерий. Это не доказывает отсутствие влияния всей памяти: начальный разгон и первый вход не менялись, а обучение считывателя осталось прежним.')
decision=('Не внедрять постоянную предысторию и не подбирать её длину либо содержимое по пяти случаям. Сохранить отдельно отрицательный вывод об этой правке. '
 'Для дальнейшей локализации нужен контроль, отделяющий формирование карты от пространственного сравнения; текущий опыт не установил причину неправильного зрительного порядка.')
cannot=('Нет доказательства дефекта определённых нейронов или весов, общей вредности памяти, улучшения сохранения ID либо HOTA. '
 'Это пять повторно использованных разработческих случаев. Постоянная предыстория искусственна; возможен сдвиг распределения относительно обучения. '
 'Карты двух концов после вмешательства могут происходить из разных искусственных историй. Точная причина всех отказов недели остаётся открытой.')
maxmap=max(v for r in base['records'] for v in r['parity_errors'].values());maxscore=max(r['error'] for r in a['original_score_parity'])
lines=['# Содержимое предыстории и ошибки личности','','1. Проверенный результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Что изменялось','',
 'Для каждого конца пары сохранены первый вход, последние два входа, длина последовательности и все временные параметры. Только промежуточные ранние входы заменены первым либо предпоследним входом той же последовательности. Состояние сети не подменялось напрямую: оно получено повтором изменённого входа. Начальный разгон сети одинаков в каждой паре условий. Никаких меток или будущих кадров модель не получала.','',
 'Условия: исходное; неизменяющая копия; первая постоянная предыстория только у образца, только у кандидата и у обоих; вторая постоянная предыстория в тех же трёх сочетаниях. Порог, сдвиги совмещения, нормировки, веса ядра и считывателя сохранены.','',
 'Для образца 1102 обе замены фактически сохраняют входы из-за повторов в исходном префиксе; для образца 574 неизменна первая замена. Эти нулевые вмешательства видны в replay.json, changed_earlier_values; они не являются доказательством нечувствительности ядра.','',
 'Прежние проверки z13_anchor_state_v1 и common_state_memory_20261109 не повторялись: первая убирала поисковую рекурсию, вторая меняла возраст образца. Здесь сохранён последний переход каждого конца пары. Источники прежних протоколов и отличия записаны до прогона.','',
 '3. Проверки','',f'Все десять исходных карт ядра совпали с архивом в пределах {maxmap:.9g}, заранее установленный допуск 0,0001; обычные карты совпали точно. Максимальное расхождение исходных баллов {maxscore:.9g}. Все исходные случаи проверены до вмешательств. Неизменяющий контроль точный. Источники baseline_gate.json и audit.json.','',
 'Независимый расчёт заново построил каждый входной префикс, использовал другой интерполятор, формулу согласия через средний квадрат различий и скалярное суммирование. Общие карты, геометрия, метки и слои раскрыты; динамика ядра независимой реализацией не проверена. Контрольные суммы исходников и входов сохранены, веса неизменны; рабочие процессы завершились. Источники protocol.json, audit.json, execution_receipt.json.','',
 '4. Все оценки FlyVis','']
for r in scores['records']:
    if r['representation']=='C34':lines += [f"Номер {r['old_id']}; история образца и кандидата {r['mode']}: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines += ['5. Основание и границы','',
 'Lappalainen и соавторы (2024), Connectome-constrained networks predict neural activity across the fly visual system, https://www.nature.com/articles/s41586-024-07939-3. Рекуррентное устройство модели и временное объединение зрительной информации обосновывают раздельный контроль предыстории и последнего перехода. Статья не устанавливает причину наших ошибок; её архитектура заново не внедрялась.','',
 'Изменение числа правильных решений относительно исходного условия: 0 %. Устранено 0 из 3 ключевых ошибок (0 %). Доля общего достижимого качества не измерена.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,False]):item.update(met=value,evidence=rel+'/audit.json')
c['evidence'] += [dict(kind='file',ref=rel+'/'+n) for n in ['scores.json','audit.json','replay.json','baseline_gate.json','execution_receipt.json','РЕЗУЛЬТАТ.md']]
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1');problem['cases'].append(dict(date='2026-11-12',component='Входная предыстория полного модуля FlyVis',fact=obtained,hypothesis='Проверенные постоянные префиксы не объяснили три ошибки как устранимое следствие раннего содержимого. Начальный разгон не исследован.',discriminating_check=decision,limitation=cannot,evidence=[rel+'/audit.json',rel+'/scores.json'],source_sha256=sha(OUT/'audit.json')));problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);ident='lappalainen2024_recurrent_history_control';assert not any(s['id']==ident for s in d['sources'])
d['sources'].append(dict(id=ident,title='Рекуррентная зрительная динамика — Lappalainen и соавторы',problem_group='Сохранение личности',method_group='Временная история',publication='Janne K. Lappalainen et al. Connectome-constrained networks predict neural activity across the fly visual system. Nature 634, 1132–1140, 2024.',doi='10.1038/s41586-024-07939-3',url='https://www.nature.com/articles/s41586-024-07939-3',sections='Описание рекуррентного устройства и временного объединения зрительной информации для оптического потока.',established='Авторы строят рекуррентную модель по связям зрительной системы и обучают её вычислению оптического потока.',problem='С1 — требуется отделить вклад предыстории состояния от последнего зрительного перехода.',application='В проверке содержимого предыстории сохранены первый и последние два входа, временная схема и веса; ранние входы меняются отдельно у образца и кандидата.',limitation='Наши фиксированные замены не изменили решений. Статья не доказывает причину отказов и не даёт готового правила сохранения личности.',evidence=[rel+'/protocol.json',rel+'/audit.json'],related=['exp:'+ID,'sys:С1']))
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Полный трекер не запускался, пороги и веса сохранены.')
print('RECORDED_NEGATIVE OWNED_WORKERS_ENDED')
