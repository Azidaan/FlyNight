from common import *
import shutil,psutil
a=read(OUT/'audit.json');scores=read(OUT/'scores.json');base=read(OUT/'baseline_gate.json');assert a['passed']
model=read(OUT/'loaded_model.json');guardpath=Path(model['supervisor']['supervisor_report']);guard=read(guardpath);assert guard['returncode']==0 and guard['reason'] is None and not guard['stop_errors']
owned=[]
for pid,birth in guard['owned'].items():
    try:proc=psutil.Process(int(pid));same=proc.create_time()==birth
    except psutil.NoSuchProcess:same=False
    assert not same,(pid,birth);owned.append(dict(pid=int(pid),create_time=birth,ended_or_PID_reused=True))
save('execution_receipt.json',dict(passed=True,guard=guard,all_owned_ended=True,process_checks=owned))
shutil.copy2(guardpath.with_name('memory_guard.worker.log'),OUT/'worker.log')
maxmap=max(v for r in base['records'] for v in r['parity_errors'].values())
obtained=('На пяти случаях проверены семь условий и три представления:105оценок. Все условия сохранили решения: FlyVis принимает2свои и1чужую, оба обычных контроля3свои и1чужую. '
 'У232 исходная оценка−1,452065; полное согласование времени даёт−1,393503 при линейном заполнении и−1,457348 при удержании. '
 'У832:−1,396481 → −2,096251/−1,319100; у574:1,843682 → 1,893772/1,828521. '
 'Все критерии исправления трёх ошибок не пройдены. Источник research_time_semantics_probe_20261112/scores.json.')
interpret=('Временная семантика причинно меняет численные оценки при фиксированных остальных условиях, но проверенные изменения не исправляют решения. '
 'Нельзя объявлять найденное расхождение времени достаточной причиной отказов232/832/574. Удаление повторов и расширение пропусков проверены отдельно; готовые сдвиги, порог и обученные слои сохранены. '
 'Физические промежуточные изображения неизвестны: линейное заполнение и удержание являются двумя явными предположениями. Вывод не исключает всех возможных вариантов непрерывной истории.')
decision='Не переносить эти изменения времени в рабочую систему и не подбирать длительности по пяти случаям. Сохранить несоответствие времени как установленное свойство реализации с недоказанным вкладом в ошибки. Следующий уровень — отделить влияние содержимого прежней истории от реакции на текущий вход; это не проверено данным опытом.'
cannot='Не установлена точная причина всех отказов, нет улучшения HOTA или подтверждённого дефекта весов ядра. Изменённая история оценивается прежними слоями; возможен сдвиг распределения признаков. Нет независимого теста или полного трекера.'
lines=['# Время наблюдений и ошибки личности','', '1. Результат','',obtained,'',interpret,'',decision,'',cannot,'','2. Контроли и исполнение','',
 f'Точные входы восстановлены из архивов. Десять обычных карт совпали побайтно; максимальная разность десяти исходных карт FlyVis {maxmap:.9g} при заранее установленном допуске0,0001. Максимальное расхождение исходных оценок {max(x["error"] for x in a["original_score_parity"]):.9g}. Все исходные случаи проверены до вмешательств. Неизменяющий контроль совпал точно. Веса и входные файлы сохранились. Источники baseline_gate.json, audit.json, execution_receipt.json.','',
 'Семь условий: исходное время по номеру наблюдения; неизменяющая копия; расширение только пропусков с линейным заполнением; расширение только пропусков с удержанием предыдущего входа; удаление только повторов нулевого времени; оба изменения с линейным заполнением; оба изменения с удержанием. Временные промежутки рассчитывались по номерам кадров, параметры не подбирались.','',
 'Независимая проверка прочитала сохранённые карты, заново построила временные последовательности, использовала другой интерполятор, формулу согласия через средний квадрат различий и скалярное суммирование. Общие исходные карты, геометрия, метки и слои указаны; динамика ядра второй реализацией не проверялась.','',
 '3. Все оценки FlyVis','']
for r in scores['records']:
    if r['representation']=='C34':lines += [f"{r['mode']}, номер {r['old_id']}: {r['score']:.9f}; {'принят' if r['accepted'] else 'отвергнут'}.",'']
lines+=['4. Литературное основание','',
 'Rubanova, Chen, Duvenaud, 2019, Latent ODEs for Irregularly-Sampled Time Series, https://arxiv.org/abs/1907.03907. Работа рассматривает учёт неравномерных интервалов наблюдений в моделях с непрерывным временем. Она помогает отделить реальное время от индекса наблюдения и явно назвать допущения заполнения пропусков. Её архитектура в проект не переносилась; статья не доказывает причину ошибок FlyVis.','',
 'Изменение числа правильных решений относительно исходного условия:0 %. Устранённых из трёх ключевых ошибок:0 %. Доля общего достижимого качества не оценивалась. Отрицательный результат относится к перечисленным условиям.']
with (OUT/'РЕЗУЛЬТАТ.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);c=next(c for c in d['experiments'] if c['id']==ID);c.update(state='completed_negative',done='2026-11-12',obtained=obtained,interpretation=interpret,decision=decision,cannot_claim=cannot)
for item,value in zip(c['acceptance'],[True,True,a['primary_any_elapsed_gate']]):item.update(met=value,evidence='research_time_semantics_probe_20261112/audit.json')
c['evidence'] += [dict(kind='file',ref='research_time_semantics_probe_20261112/'+n) for n in ['scores.json','audit.json','replay.json','baseline_gate.json','execution_receipt.json','РЕЗУЛЬТАТ.md']]
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/system_problems.json';raw=path.read_bytes();d=json.loads(raw);problem=next(x for x in d['problems'] if x['id']=='С1')
problem['cases'].append(dict(date='2026-11-12',component='Временная история полного модуля FlyVis',fact=obtained,hypothesis='Найденное несоответствие времени не стало достаточным объяснением трёх ключевых ошибок; более глубокая причина не установлена.',discriminating_check=decision,limitation=interpret,evidence=['research_time_semantics_probe_20261112/audit.json','research_time_semantics_probe_20261112/scores.json'],source_sha256=sha(OUT/'audit.json')))
problem['cards'].append(ID);assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/useful_literature.json';raw=path.read_bytes();(OUT/'before/useful_literature.json').write_bytes(raw);d=json.loads(raw);ident='rubanova2019_irregular_time';assert not any(s['id']==ident for s in d['sources'])
d['sources'].append(dict(id=ident,title='Неравномерные интервалы наблюдений — Rubanova и соавторы',problem_group='Сохранение личности',method_group='Временная история',publication='Yulia Rubanova, Ricky T. Q. Chen, David Duvenaud. Latent ODEs for Irregularly-Sampled Time Series. NeurIPS, 2019.',doi='10.48550/arXiv.1907.03907',url='https://arxiv.org/abs/1907.03907',sections='Проверены аннотация и положение о передаче интервала между наблюдениями в рекуррентную модель; полный алгоритм не воспроизводился.',established='Авторы рассматривают модели непрерывной динамики для наблюдений с произвольными временными промежутками.',problem='С1 — нужно отделить свойства реализации временной истории от причин неудачного различения объектов.',application='Источник помог различить время по индексу наблюдения и реальные интервалы. В нашем контролируемом опыте отдельно изменены пропуски и повторы при фиксированных весах; оба заполнения пропусков явно обозначены предположениями.',limitation='Архитектура Latent ODE в проект не внедрялась. Наши семь условий не исправили три ключевые ошибки; статья не доказывает причину отказов FlyVis.',evidence=['research_time_semantics_probe_20261112/protocol.json','research_time_semantics_probe_20261112/audit.json'],related=['exp:'+ID,'sys:С1']))
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Получено (`'+ID+'`): '+obtained+' Вывод. '+interpret+' Решение. '+decision+' Веса и порог неизменны; полный трекер не запускался. Литературное основание: Rubanova и соавторы,2019, arXiv1907.03907; оно не заменяет причинную проверку.')
print('RECORDED_NEGATIVE','OWNED_WORKERS_ENDED')
