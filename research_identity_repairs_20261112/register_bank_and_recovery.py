from common import *
bank=read(OUT/'review/bank_protocol.json');policy=read(OUT/'recovery574/proposal_protocol.json')
assert sha(OUT/'review/bank_protocol.json')=='c6d53335f1d6238cb38514a5075dd572d7e0c2f32a495086d2ea77fbe4fb0fb3'
assert sha(OUT/'recovery574/proposal_protocol.json')=='4aa446d06cfa64f6f436de15523d3999a80d32b7704837f444da285dcf5655e9'
inputs={}
for p in [bank,policy]:
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n;inputs[n]=h
for n,h in policy['sources'].items():assert sha(OUT/'recovery574'/n)==h,n;inputs[(OUT/'recovery574'/n).relative_to(ROOT).as_posix()]=h
for n in ['review/bank_protocol.json','review/extract_bank.py','review/audit_bank.py','recovery574/proposal_protocol.json','bind_bank.py','run_queue.py','register_bank_and_recovery.py']:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
cat=read(ROOT/'experiments/project_history.json');tp=ROOT/'experiments/obsidian_experiment_topics.json';topics=read(tp)
for identity,title,p,hypothesis in [('identity_repair_reference_bank_20261112','Память всех конкурирующих объектов для проверки восстановления',bank,'Полный банк прошлых реальных наблюдений позволяет проверить взаимный выбор без подмены конкуренции одной известной историей.'),('identity_repair_excluded_low_recovery_20261112','Подтверждение исключённых детекций без изменения рабочих порогов',policy,'Отдельная ветвь с фиксированными проверками внешнего вида, близости, взаимности и владения может восстановить574, сохранив прежние назначения.')]:
 assert not any(c['id']==identity for c in cat['experiments'])
 cat['experiments'].append(dict(id=identity,title=title,area='flyvis',state='running',started='2026-11-12',question=p.get('question',p.get('method')),hypothesis=hypothesis,expected='Проверка на прежних пяти случаях без эталонных меток в принимаемом решении; все отрицательные контроли сохраняются.',acceptance=[dict(criterion=c,met=None) for c in p['acceptance']],evidence=[dict(kind='file',ref=(OUT/('review/bank_protocol.json' if p is bank else 'recovery574/proposal_protocol.json')).relative_to(ROOT).as_posix())]))
 topics['topics'][identity]=['Сохранение личности','Подтверждение восстановления','Внешний вид и конкуренция']
(ROOT/'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode());tp.write_bytes((json.dumps(topics,ensure_ascii=False,indent=2)+'\n').encode())
jobs=[dict(entry=(OUT/'review/extract_bank.py').relative_to(ROOT).as_posix(),args=['--go-protocol-sha256',sha(OUT/'review/bank_protocol.json')],timeout=1200),dict(entry=(OUT/'review/audit_bank.py').relative_to(ROOT).as_posix(),timeout=600),dict(entry=(OUT/'bind_bank.py').relative_to(ROOT).as_posix(),timeout=60),dict(entry=(OUT/'recovery574/worker.py').relative_to(ROOT).as_posix(),timeout=600),dict(entry=(OUT/'recovery574/audit.py').relative_to(ROOT).as_posix(),timeout=600)]
save('queue2.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=jobs,deadline='2026-11-13T07:35:00+03:00'))
journal('Постановка (`identity_repair_reference_bank_20261112`, `identity_repair_excluded_low_recovery_20261112`): общая память107прошлых наблюдений на37кадрах; исходные5целевых образцов не пересчитываются,222текущих признака переиспользуются. Восстановление допускает только исключённые низкоуверенные рамки: строгий взаимный выбор на полном наборе конкурентов, заимствованные неизменные BoT-SORT ограничения внешнего вида и близости, группы дублей по прежнему порогу0,7. Перенос этих ограничений наMASA — проверяемое предположение, не калиброванная вероятность. Все уже сделанные назначения защищены. Предусмотрены отключение ветви, перестановка признаков, отсутствие конкурентов/признаков и удаление всех своих кандидатов только для отрицательного контроля. Источники research_identity_repairs_20261112/review/bank_protocol.json и recovery574/proposal_protocol.json. Никаких глобальных изменений high/low порогов, обучения или полного трекера.')
print('REGISTERED_BANK_AND_RECOVERY')
