from common import *
inputs={};cat=read(ROOT/'experiments/project_history.json');topics=read(ROOT/'experiments/obsidian_experiment_topics.json')
for stem,identity,title in [('source_support','remaining_causes_source832_20261112','Исходная часть объекта и отказ832'),('factorial','remaining_causes_geometry_state832_20261112','Совместное влияние координат и состояния на отказ832')]:
 p=read(OUT/'geometry832'/(stem+'_protocol.json'))
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n;inputs[n]=h
 assert not any(c['id']==identity for c in cat['experiments'])
 cat['experiments'].append(dict(id=identity,title=title,area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить достаточность указанной правки от остальных факторов; сохранить все варианты и ошибки остальных четырёх случаев.',acceptance=[dict(criterion=c,met=None) for c in p['criteria']],evidence=[dict(kind='file',ref=(OUT/'geometry832'/(stem+'_protocol.json')).relative_to(ROOT).as_posix())]))
 topics['topics'][identity]=['Сохранение личности','Причины отказов','Геометрия и смешение объектов']
 for name in [stem+'_protocol.json',stem+'_worker.py','audit_addendum.py']:
  f=OUT/'geometry832'/name;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
 with (OUT/'geometry832'/(stem+'_GO.json')).open('x',encoding='utf8') as f:json.dump(dict(approved=True,registered=now(),id=identity,protocol_sha256=sha(OUT/'geometry832'/(stem+'_protocol.json'))),f,indent=2)
a=read(OUT/'overlap574/protocol_addendum_v2.json');assert sha(OUT/'overlap574/protocol_addendum_v2.json')=='7521382d0160569e7cf327100a7ab2207a0ea6eed4a4175bae521a07457e7512'
for n,h in a['additional_inputs'].items():assert sha(ROOT/n)==h,n;inputs[n]=h
for n,h in a['new_source_hashes'].items():assert sha(OUT/'overlap574'/n)==h;inputs[(OUT/'overlap574'/n).relative_to(ROOT).as_posix()]=h
for name in ['proposed_protocol.json','protocol_addendum_v2.json','audit.py','baseline_diagnostic.json']:
 f=OUT/'overlap574'/name;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
card=next(c for c in cat['experiments'] if c['id']=='remaining_causes_overlap574_20261112');card['evidence'].append(dict(kind='file',ref='research_remaining_causes_20261112/overlap574/protocol_addendum_v2.json'))
(ROOT/'experiments/project_history.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (OUT/'overlap574/GO_v2.json').open('x',encoding='utf8') as f:json.dump(dict(approved=True,registered=now(),addendum_sha256=sha(OUT/'overlap574/protocol_addendum_v2.json')),f,indent=2)
for n in ['run_queue.py','register_followups.py']:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
jobs=[]
for entry,args,timeout in [('overlap574/worker_v2.py',[],600),('overlap574/audit.py',[],600),('geometry832/source_support_worker.py',[],180),('geometry832/audit_addendum.py',['source_support'],300),('geometry832/factorial_worker.py',[],180),('geometry832/audit_addendum.py',['factorial'],600)]:jobs.append(dict(entry=(OUT/entry).relative_to(ROOT).as_posix(),args=args,timeout=timeout))
save('queue2.json',dict(registered=now(),runner_sha256=sha(OUT/'run_queue.py'),inputs=inputs,jobs=jobs,deadline='2026-11-13T07:35:00+03:00',execution='Sequential after queue1 completion; memory reserve, original strict tolerances, no new neural forward. Original574 failure retained. All5controls, oracle geometry only diagnostics.'))
journal('Постановка продолжения832: отдельно исходная опора объекта (`remaining_causes_source832_20261112`) и сочетание координат с уже сохранёнными состояниями (`remaining_causes_geometry_state832_20261112`). Условия зарегистрированы до исходов, все пять случаев, неизменные головы и нулевой порог. Первая проверка координат832 и независимый пересчёт пройдены: временное представление меняет знак, C34не исправлено; оба вывода сохраняются раздельно. Источники geometry832/result.json,independent_audit.json,coordinate_audit.json. Исправление574v2 использует точные архивные карты, строгая проверка1e-9 не ослаблена; прежний сбой сохранён. Очередь research_remaining_causes_20261112/queue2.json запускается после подтверждённого завершения queue1.')
print('REGISTERED_QUEUE2',len(jobs),len(inputs))
