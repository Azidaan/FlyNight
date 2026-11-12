from common import *
import shutil,psutil
a=read(OUT/'audit.json');scores=read(OUT/'scores.json');base=read(OUT/'baseline_gate.json');render=read(OUT/'render_manifest.json');assert a['passed'] and not a['primary_any_both_gate'] and not scores['decision_changes'];executions={}
for name,supervisor in [('render',read(OUT/'render_supervisor.json')),('core',read(OUT/'loaded_model.json')['supervisor'])]:
    guardpath=Path(supervisor['supervisor_report']);guard=read(guardpath);assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors'];owned=[]
    for pid,birth in guard['owned'].items():
        try:same=psutil.Process(int(pid)).create_time()==birth
        except psutil.NoSuchProcess:same=False
        assert not same;owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
    executions[name]=dict(guard=guard,process_checks=owned,all_owned_ended=True);shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/('render_process.log' if name=='render' else 'worker.log'))
save('execution_receipt.json',dict(passed=True,executions=executions,all_owned_ended=True));rel=OUT.relative_to(ROOT).as_posix()
obtained=('Из видео точно восстановлены45исходных наблюдений; всего получены74различных входа глаза. На пяти случаях проверены восемь сочетаний и три представления,120оценок. '
 'Ни одно решение не изменилось: FlyVis принимает две свои пары и одну чужую, оба обычных контроля три свои и одну чужую. '
 'Для232 исходный балл −1,452065; неподвижное окно последнего перехода у обоих концов −1,492523, всей истории −1,604931. '
 'Для832: −1,396481 → −1,885031 / −2,759268; для574: +1,843682 → +1,843682 / +2,147785. '
 'Основной критерий исправления232 и вторичный критерий всех трёх ошибок не пройдены. Источники '+rel+'/retinal_baseline_gate.json, render_manifest.json, scores.json, audit.json.')
interpret=('Смена координат входного окна влияет на карты и оценки при сохранении конечных изображений. Но приведение последнего перехода или всей истории к конечному окну не исправило решения и ухудшило оценки232/832. '
 'Нельзя объявлять изменение окна достаточной причиной этих отказов или считать фиксированное окно принятым исправлением. '
 'Неподвижное окно может менять ранее видимую часть объекта, начальную адаптацию и распределение признаков; это не независимая разметка физических соответствий.')
decision=('Не внедрять неподвижное окно и не подбирать его положение по пяти случаям. Сохранить отрицательный результат этой конкретной правки. '
 'Следующий различающий вопрос — сохраняется ли дополнительный отказ232 в установившемся ответе на то же конечное изображение; он отделит переходную динамику от статического представления. Эта проверка ещё не выполнена.')
cannot=('Не установлена корневая причина всех отказов и нет доказанного дефекта определённых нейронов. Пять повторно использованных случаев не дают независимой оценки обобщения. '
 'Общее движение камеры сохраняется; вычисление прежних кадров в конечном окне требует повторного извлечения входов. Нет нового обучения, изменения порога, полного трекера или улучшения HOTA.')
maxmap=max(v for r in base['records'] for v in r['parity_errors'].values())
lines=['# История в неподвижном входном окне','','1. Результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Что изменялось','',
 'Для каждого конца пары сохранены список исходных кадров и конечное окно. Первый вариант заново вырезает только предпоследний наблюдаемый кадр в координатах конечного окна. Второй заново вырезает в этих координатах всю историю. Конечный вход совпадает с исходным точно. Изображения взяты из настоящих кадров с прежней полноэкранной предварительной обработкой; сдвиги между деталями не оценивались.','',
 'История меняется отдельно у образца, отдельно у кандидата и у обоих. Каждый вариант применяется также к обычным признакам. Геометрия сравнения, сдвиги, нормировки, веса и временная схема сохранены. Прежние проверки совмещали уже готовые карты; здесь меняются входы до рекуррентной обработки.','',
 'Последние поисковые окна у всех пяти случаев уже были одинаковы, поэтому замена только последнего перехода у кандидата является нулевым вмешательством. У574 также не меняется последнее окно образца. Эти случаи сохраняются как фактические контроли и не являются свидетельством нечувствительности ядра. Источники render_manifest.json, schedules, и replay.json, changed_values.','',
 '3. Проверки исполнения','',
 f'Все45исходных наблюдений восстановлены точно до получения изменённых входов. Альтернативное суммирование окрестностей дало максимальное расхождение {render["max_alternative_error"]:.9g} при заранее заданном допуске0,001. Оно использует NumPy и другой порядок суммирования; декодер, предварительная обработка и изменение размера общие. Источник render_manifest.json.','',
 f'Десять исходных карт ядра совпали с архивом в пределах {maxmap:.9g}, допуск0,0001; десять обычных карт совпали точно. Все исходные случаи проверены до вмешательства в ядро. Неизменяющий контроль точный. Независимо восстановлены списки кадров и окон; оценки проверены другим интерполятором и формулой согласия через средний квадрат различий. Источники baseline_gate.json, audit.json.','',
 'Оба вычислительных процесса завершены; CPU-подготовка закончилась до GPU-этапа. Входные видео, состояния предварительной обработки, веса и исходники связаны контрольными суммами с протоколом. Отдельно сохранены исходные массивы входов и карты. Источники protocol.json, render_manifest.json, replay.json, execution_receipt.json.','',
 '4. Все оценки FlyVis','']
for r in scores['records']:
    if r['representation']=='C34':lines += [f"Номер {r['old_id']}; режимы образца и кандидата {r['mode']}: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines+=['5. Литературное основание','',
 'Lappalainen и соавторы (2024), Connectome-constrained networks predict neural activity across the fly visual system, https://www.nature.com/articles/s41586-024-07939-3. В работе модель получает зрительные последовательности и обучается оптическому потоку. Это обосновывает необходимость различать изменение входной последовательности и последующее совмещение готовых карт. Статья не доказывает причину наших ошибок. Алгоритм Lucas–Kanade, найденный при поиске источников, в проверке не использовался: окно задано сохранёнными координатами, а не оценено по изображениям.','',
 'Изменение числа правильных решений относительно исходного условия:0 %. Исправлено0из3ключевых ошибок (0 %). Доля общего достижимого качества не измерена.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,False]):item.update(met=value,evidence=rel+'/audit.json')
c['evidence'] += [dict(kind='file',ref=rel+'/'+n) for n in ['scores.json','audit.json','render_manifest.json','retinal_baseline_gate.json','replay.json','baseline_gate.json','execution_receipt.json','РЕЗУЛЬТАТ.md']]
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1');problem['cases'].append(dict(date='2026-11-12',component='Координаты входной истории полного модуля FlyVis',fact=obtained,hypothesis='Неподвижное окно не устранило дополнительный отказ232. Его достаточная причинная роль не подтверждена; переходная динамика и статическое представление ещё не разделены.',discriminating_check=decision,limitation=cannot,evidence=[rel+'/audit.json',rel+'/scores.json',rel+'/render_manifest.json'],source_sha256=sha(OUT/'audit.json')));problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);s=next(x for x in d['sources'] if x['id']=='lappalainen2024_recurrent_history_control');field='application';s.setdefault('revisions',[]).append(dict(date=now(),field=field,old=s[field],reason='Добавлена роль первоисточника при проверке входной геометрии; прежний текст сохранён.'));s[field]+='\n\n12.11.2026. Проверено изменение координат реальной входной истории до рекуррентной обработки: конечные наблюдения сохранены, последние переходы или вся история извлечены в неподвижном окне. Решения не улучшились. Роль статьи — постановка обработки последовательности и обучения оптическому потоку, не доказательство нашей причины.';s['evidence'] += [rel+'/protocol.json',rel+'/audit.json'];s['related'].append('exp:'+ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Порог и веса неизменны, полный трекер не запускался.')
print('RECORDED_NEGATIVE BOTH_WORKERS_ENDED')
