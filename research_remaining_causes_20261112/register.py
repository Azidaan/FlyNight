from common import *
import shutil,subprocess
jobs=[('geometry832','remaining_causes_geometry832_20261112','Геометрия рамок и отказ своей пары832'),('overlap574','remaining_causes_overlap574_20261112','Перекрывающиеся объекты и ложное разрешение574')]
assert not (OUT/'before').exists();(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
catalog=read(ROOT/'experiments/project_history.json');topics=read(ROOT/'experiments/obsidian_experiment_topics.json');bound={};meta=[]
for folder,identity,title in jobs:
 p=read(OUT/folder/'proposed_protocol.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n;bound[n]=h
 for n,h in p.get('source_hashes',{}).items():assert sha(OUT/folder/n)==h;bound[(OUT/folder/n).relative_to(ROOT).as_posix()]=h
 for n in ['proposed_protocol.json','worker.py','audit.py']:
  q=OUT/folder/n;bound[q.relative_to(ROOT).as_posix()]=sha(q)
 criteria=p.get('criteria',p.get('acceptance'));assert criteria
 assert not any(c['id']==identity for c in catalog['experiments'])
 catalog['experiments'].append(dict(id=identity,title=title,area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p.get('hypothesis',str(p.get('hypotheses'))),expected='Установить достаточность конкретного вмешательства, сохранив отрицательные контроли и границу физического вывода.',acceptance=[dict(criterion=x,met=None) for x in criteria],evidence=[dict(kind='file',ref=(OUT/folder/'proposed_protocol.json').relative_to(ROOT).as_posix())]))
 topics['topics'][identity]=['Сохранение личности','Причины отказов','Геометрия и смешение объектов'];meta.append(dict(folder=folder,id=identity,protocol_sha256=sha(OUT/folder/'proposed_protocol.json')))
for n in ['register.py','supervise_jobs.py']:
 q=OUT/n;bound[q.relative_to(ROOT).as_posix()]=sha(q)
if (OUT/'geometry832/coordinate_audit.py').exists():bound[(OUT/'geometry832/coordinate_audit.py').relative_to(ROOT).as_posix()]=sha(OUT/'geometry832/coordinate_audit.py')
save('batch_protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),jobs=meta,inputs=bound,authorization='New user request after18:20 explicitly authorizes continued autonomous832/574 investigation and all acceleration modes, including parallelAIagents/programs/ReID. No production thresholds/weights/fulltracker changes.',execution='3AIresearch agents with isolated folders, at most2CPUworkers with2threads each, shared systemmemoryreserve. Independent supervisor per ownedPID and creationtime, no arbitrary allocationcap. No new GPU call in this batch.',deadline='2026-11-13T07:35:00+03:00',hard_cutoff='2026-11-13T07:40:00+03:00',limits='Oracle geometry and bbox-exclusive support do not prove physical pixel ownership. ReID proposal separately registered before any forward.'))
(ROOT/'experiments/project_history.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf8');(ROOT/'experiments/obsidian_experiment_topics.json').write_text(json.dumps(topics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка двух проверок (`remaining_causes_geometry832_20261112`, `remaining_causes_overlap574_20261112`): после нового прямого поручения автора продолжить разбор832/574 и ускорить независимые направления. Зарегистрированы отдельные гипотезы, исходные суммы, вмешательства и контроли агентов; максимум2CPUзадания,2потокакаждое, общий резерв памяти и независимые контроллеры. Эталонная геометрия — только диагностика, не вход рабочего трекера. ТриИИагента изолированы, общие документы ведёт координатор. Веса, пороги, окружение и полный трекер не меняются. Остановка новых длинных расчётов до13.11 07:35МСК, всех до07:40. Источник research_remaining_causes_20261112/batch_protocol.json.')
for x in meta:
 with (OUT/x['folder']/'GO.json').open('x',encoding='utf8') as f:json.dump(dict(approved=True,registered=now(),experiment_id=x['id'],batch_protocol_sha256=sha(OUT/'batch_protocol.json'),protocol_sha256=x['protocol_sha256']),f,indent=2)
print('REGISTERED',len(jobs),len(bound))
