from common import *
import shutil,psutil,numpy as np
a=read(OUT/'audit.json');scores=read(OUT/'scores.json');base=read(OUT/'baseline_gate.json');rep=read(OUT/'replay.json');coord=read(OUT/'coordinate_audit.json')
assert a['passed'] and not a['primary_any_both_gate'] and coord['passed']
assert not [r for r in scores['decision_changes'] if r['representation']=='C34']
guardpath=Path(read(OUT/'loaded_model.json')['supervisor']['supervisor_report']);guard=read(guardpath)
assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors'];owned=[]
for pid,birth in guard['owned'].items():
    try:same=psutil.Process(int(pid)).create_time()==birth
    except psutil.NoSuchProcess:same=False
    assert not same;owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/'worker.log')
save('execution_receipt.json',dict(passed=True,guard=guard,process_checks=owned,all_owned_ended=True))
# Recompute whole-network state differences and saved residual summaries from raw arrays.
assert sha(OUT/'states.npz')==rep['states_sha256'];residuals=[];contrasts=[]
with np.load(OUT/'states.npz',allow_pickle=False) as states:
    for r in rep['records']:
        key=f"{r['mode']}_{r['case']}_{r['endpoint']}";delta=states[key+'_last_delta'];full=states[key]
        assert delta.shape==full.shape==(1,coord['nodes']) and np.isfinite(full).all() and np.isfinite(delta).all()
        maximum=float(np.max(np.abs(delta)));rms=float(np.linalg.norm(delta.astype(float))/np.sqrt(delta.size))
        assert maximum==r['last_state_delta_max'] and abs(rms-r['last_state_delta_rms'])<1e-12
        residuals.append(dict(old_id=r['old_id'],endpoint=r['endpoint'],mode=r['mode'],max_abs=maximum,rms=rms))
    for ci,case in enumerate(read(OUT/'protocol.json')['cases']):
        for ei in [0,1]:
            delta=states[f'history_hold_{ci}_{ei}'].astype(float)-states[f'terminal_hold_{ci}_{ei}'].astype(float)
            contrasts.append(dict(old_id=case['case']['old_id'],endpoint=ei,max_abs=float(abs(delta).max()),rms=float(np.linalg.norm(delta)/np.sqrt(delta.size))))
for old,new in zip(rep['history_contrasts'],contrasts):
    assert old['old_id']==new['old_id'] and old['endpoint']==new['endpoint'] and abs(old['max_abs']-new['max_abs'])<1e-12 and abs(old['rms']-new['rms'])<1e-10
save('state_audit.json',dict(passed=True,residuals=residuals,contrasts=contrasts,convergence_claim=False,reason='Finite1s hold; residuals and history dependence are small but nonzero. No arbitrary convergence threshold chosen after outcomes.'))
rel=OUT.relative_to(ROOT).as_posix()
obtained=('Проверено точное соответствие координат 721 колонки для 34 выходных и 8 входных типов реально загруженной модели; сохранённая геометрия совпала. '
 'На пяти случаях, восьми сочетаниях предысторий и трёх представлениях пересчитаны 120 оценок. Ни одно решение FlyVis не изменилось. '
 'Для 232 исходный балл −1,452065, после удержания конечных изображений с исходной предысторией −1,214943, с постоянной предысторией −1,212381. '
 'Для 832 соответственно −1,396481 / −2,071548 / −2,072943; для чужой 574: +1,843682 / +1,994636 / +1,993948. '
 'Основной и вторичный критерии не пройдены. У обычных контролей есть шесть изменений решений в отдельных сочетаниях: четыре исправления 832 во временном представлении и два ухудшения 232 в пространственном представлении? '
 'Источники: '+rel+'/coordinate_audit.json, scores.json, audit.json.')
# Counts are derived rather than inferred from the printed list.
changes=scores['decision_changes'];improved=sum(r['improved'] for r in changes);worsened=len(changes)-improved
obtained=obtained.replace('четыре исправления 832 во временном представлении и два ухудшения 232 в пространственном представлении?',f'{improved} исправления 832 во временном представлении и {worsened} ухудшения 232 в пространственном представлении.')
interpret=('Ошибка перестановки пространственных индексов не обнаружена. Неверное разделение своих и чужих пар сохраняется после фиксированного удержания конечных изображений, когда оценки двух предысторий сближаются. '
 'Это отклоняет конкретную правку — использовать ответ после одной секунды удержания с прежним считывателем — и не поддерживает объяснение отказов только последним переходом. '
 'Полное установившееся состояние не доказано: остаточная динамика и различия предысторий ненулевые. Конечное представление, физические соответствия деталей и решающее правило ещё требуют разделения.')
decision=('Не внедрять удержание конечного изображения и не подбирать его длительность по этим пяти случаям. Следующая диагностика должна проверить физические соответствия и вклад локальных областей в ошибочный балл на сохранённых картах, прежде чем менять ядро или обучать новый считыватель.')
cannot=('Не установлена точная причина всех отказов и не доказан дефект конкретного нейрона. Проверка ограничена пятью повторно использованными случаями и фиксированными головами решения; удержание синтетическое и не добавляет наблюдений. '
 'Координатная исправность не доказывает соответствие одной физической детали между рамками. Нет обучения, изменения порогов, полного трекера, выигрыша HOTA или независимой оценки обобщения.')
lines=['# Конечное изображение и переходная реакция FlyVis','','1. Результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Проверенная цепочка','',
 'Порядок координат проверен у реально загруженного коннектома и индексов подачи входа/выгрузки карты. До изменённых последовательностей воспроизведены все исходные карты; клонированный контроль совпал точно. Конечные изображения и веса неизменны. Независимо пересчитаны расписания синтетических входов и оценки. Текущее изображение удерживалось 30 наблюдений по два шага 1/60 секунды; длительность задана до опыта. Одинаковое вмешательство применено к обычным признакам.','',
 'Сохранены полные состояния 45 669 нейронов и изменения за последний шаг наблюдения. Повторная проверка state_audit.json читает эти массивы и независимо пересчитывает нормы. Наибольшее различие двух предысторий после удержания: '+str(max(x['max_abs'] for x in contrasts))+'. Ненулевая остаточная динамика не позволяет заявить точную сходимость к равновесию.','',
 'Предварительная попытка прочитать HDF5 из основного окружения завершилась ModuleNotFoundError: h5py. Причина — изолированные зависимости FlyVis находятся в tools/flyvis-runtime. Чтение с этим путём выполнено успешно; установки пакетов и изменения среды не было. Выводы опираются на последующий аудит реально загруженной модели.','',
 '3. Оценки всех сочетаний','']
for r in scores['records']:
    if r['representation']=='C34':lines += [f"Номер {r['old_id']}, {r['mode']}: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines += ['4. Изменившиеся обычные контроли','']
for r in changes:lines += [f"{r['representation']}, номер {r['old_id']}, {r['mode']}: {r['before']:.9f} → {r['after']:.9f}; {'исправление' if r['improved'] else 'ухудшение'}.",'']
lines += ['5. Основание и границы','',
 'Lappalainen и соавторы (2024), Connectome-constrained networks predict neural activity across the fly visual system, https://www.nature.com/articles/s41586-024-07939-3. Первичная работа описывает рекуррентную обработку и обучение оценке движения. Это основание разделять переходные и длительные ответы, но не доказательство причины нашего отказа.','',
 'Изменение числа правильных решений FlyVis относительно исходного условия: 0 %. Исправлено 0 из 3 ключевых ошибок, 0 %. Доля общего достижимого качества не измерена.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID)
c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,False]):item.update(met=value,evidence=rel+'/audit.json')
c['evidence'] += [dict(kind='file',ref=rel+'/'+n) for n in ['scores.json','audit.json','coordinate_audit.json','state_audit.json','replay.json','baseline_gate.json','execution_receipt.json','РЕЗУЛЬТАТ.md']]
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1')
problem['cases'].append(dict(date='2026-11-12',component='Пространственная индексация и переходная реакция полного модуля FlyVis',fact=obtained,hypothesis='Перестановка индексов не найдена. Удержание конечного изображения не устраняет ошибки, первопричина ещё не локализована до физических соответствий, представления либо считывателя.',discriminating_check=decision,limitation=cannot,evidence=[rel+'/audit.json',rel+'/scores.json',rel+'/coordinate_audit.json',rel+'/state_audit.json'],source_sha256=sha(OUT/'audit.json')));problem['cards'].append(ID)
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();d=json.loads(raw);s=next(x for x in d['sources'] if x['id']=='lappalainen2024_recurrent_history_control');field='application'
s.setdefault('revisions',[]).append(dict(date=now(),field=field,old=s[field],reason='Добавлена проверка фиксированного удержания конечного изображения; прежний текст сохранён.'))
s[field]+='\n\n12.11.2026. Фиксированное удержание конечных изображений после исходной либо постоянной предыстории не исправило решений FlyVis. Переходные и длительные ответы различались, но неверное разделение пар сохранилось. Статья служит основанием постановки, а не доказательством нашей причины.'
s['evidence'] += [rel+'/protocol.json',rel+'/audit.json'];s['related'].append('exp:'+ID)
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Порог и веса неизменны, полный трекер не запускался.')
print('RECORDED_NEGATIVE WORKER_ENDED',improved,worsened)
