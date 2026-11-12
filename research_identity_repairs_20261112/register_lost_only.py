from common import *
p=read(OUT/'recovery574/scope_addendum_v3.json');assert sha(OUT/'recovery574/scope_addendum_v3.json')=='ce5597c664805d70cb60083847ac245bd9ad5f94914c37562bdd9ddfa1582cc2';inputs={}
for n,h in p['new_sources'].items():assert sha(OUT/'recovery574'/n)==h,n;inputs[(OUT/'recovery574'/n).relative_to(ROOT).as_posix()]=h
for n in ['recovery574/scope_addendum_v3.json','recovery574/proposal_protocol.json','recovery574/descriptor_binding.json','run_queue.py','register_lost_only.py']:
 f=OUT/n;inputs[f.relative_to(ROOT).as_posix()]=sha(f)
inputs.update(read(OUT/'recovery574/descriptor_binding.json')['inputs'])
cat=read(ROOT/'experiments/project_history.json');identity='identity_repair_lost_only_20261112';assert not any(c['id']==identity for c in cat['experiments'])
original=read(OUT/'recovery574/proposal_protocol.json')
cat['experiments'].append(dict(id=identity,title='Подтверждение исключённых детекций только для потерянных объектов',area='flyvis',state='running',started='2026-11-12',question='Исправляет ли та же фиксированная ветвь восстановление574 при ограничении области только потерянными объектами?',hypothesis='Узкая ветвь восстановления не должна самостоятельно обновлять неподтверждённые текущие назначения активных строк; потерянные строки проверяются теми же правилами.',expected='Сохранить все пять целевых проверок и отрицательные контроли; первичный вариант с двумя состояниями не удаляется.',acceptance=[dict(criterion=c,met=None) for c in original['acceptance']],evidence=[dict(kind='file',ref=(OUT/'recovery574/scope_addendum_v3.json').relative_to(ROOT).as_posix())]))
(ROOT/'experiments/project_history.json').write_bytes((json.dumps(cat,ensure_ascii=False,indent=2)+'\n').encode())
tp=ROOT/'experiments/obsidian_experiment_topics.json';t=read(tp);t['topics'][identity]=['Сохранение личности','Подтверждение восстановления','Внешний вид и конкуренция'];tp.write_bytes((json.dumps(t,ensure_ascii=False,indent=2)+'\n').encode())
(OUT/'recovery574/GO_lost_only.json').write_bytes((json.dumps(dict(approved=True,registered=now(),addendum_sha256=sha(OUT/'recovery574/scope_addendum_v3.json')),indent=2)+'\n').encode())
save('queue4.json',dict(registered=now(),inputs=inputs,runner_sha256=sha(OUT/'run_queue.py'),jobs=[dict(entry=(OUT/'recovery574'/n).relative_to(ROOT).as_posix(),timeout=600) for n in ['worker_v3.py','audit_v3.py']],deadline='2026-11-13T07:35:00+03:00'))
journal('Постановка вторичного уточнения (`identity_repair_lost_only_20261112`): те же фиксированные правила подтверждения применяются только к потерянным строкамstate2. Первичный вариантstate1/2полностью сохраняется; выбор области для этого варианта сделан после частичного расчёта232 и до интерпретации остальных полных исходов. Порогов и весов не меняли. Источник research_identity_repairs_20261112/recovery574/scope_addendum_v3.json. Это исследовательская проверка уже выбранных случаев, не отложенная независимая выборка.')
print('REGISTERED_LOST_ONLY')
