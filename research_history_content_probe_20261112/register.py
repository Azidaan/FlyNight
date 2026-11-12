from common import *
import shutil,subprocess
import numpy as np
old=read(ROOT/'research_time_semantics_probe_20261112/protocol.json');inputs=dict(old['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
for n in ['common.py','register.py','worker.py','audit.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['research_time_semantics_probe_20261112/protocol.json','research_time_semantics_probe_20261112/audit.json','runs/z13_anchor_state_20261025/protocol.json','runs/common_state_memory_20261109/protocol.json']:
    inputs[n]=sha(ROOT/n)
for c in old['cases']:
    for ep in c['endpoints']:assert ep['length']>=4
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=old['cases'],
 question='Does earlier retinal content cause the incorrect own/foreign decision while the final two inputs, initial state and update count remain fixed?',
 hypothesis='History before the final observed transition can alter the recurrent maps and their agreement; separating reference and candidate histories localizes any decision effect.',
 action='Consider a standardized finite prehistory for identity-map extraction only if its fixed version corrects errors without losing the correct controls. This diagnostic cannot authorize tracker deployment.',
 component='Input-history intervention in the full FlyVis module. Frozen weights and original observation-index timing, no recurrent-tensor replacement or readout fitting.',
 modes=['index','identity','hold_first','hold_penultimate'],
 pairs=[['index','index'],['identity','identity'],['hold_first','index'],['index','hold_first'],['hold_first','hold_first'],['hold_penultimate','index'],['index','hold_penultimate'],['hold_penultimate','hold_penultimate']],
 intervention='For each endpoint separately, clone its original prefix of length L. index/identity preserve it. hold_first replaces positions1 through L-3 by x0. hold_penultimate replaces these positions by x[L-2]. In all modes x0, the last two inputs, prefix length, fade-in, update count and dt are unchanged. Both replacements use only inputs available by that endpoint, never labels. Endpoint-specific replacement is intentionally no longer one continuous factual history; it tests a candidate standardized extraction rule.',
 controls='Exact cloned identity; two predetermined constant-content alternatives; independent reference-only/current-only/both interventions. All five cases and both image controls, same queries, saved shifts and fitted heads. No new alignment search.',
 acceptance=['All ten untouched core maps match frozen caches within1e-4 and all ten ordinary maps exactly; gate all original cases before interventions.','Identity maps and decisions exact; first and last two retinal inputs unchanged; same lengths and update counts; independent interpolation/agreement/scalar evaluator error below1e-9; immutable weights and inputs.','At least one predeclared BOTH-endpoint condition preserves own70/1102, accepts at least one own232/832 and rejects foreign574 at threshold0. All unilateral outcomes retained; no selecting a different mode per case.'],
 prior_tests=[dict(id='z13_anchor_state_v1',difference='Removed intervening search recurrence from exemplar to query in different data and decoder; negative result retained.'),dict(id='common_state_memory_20261109',difference='Changed template age on a common state chain; this probe keeps endpoints and final transitions fixed.'),dict(id='time_semantics_probe_20261112',difference='Changed timing, not this earlier-content-only factor; original timing preserved here.')],
 literature=dict(title='Connectome-constrained networks predict neural activity across the fly visual system',authors='Lappalainen et al.',year=2024,url='https://www.nature.com/articles/s41586-024-07939-3',role='Recurrent visual dynamics justify controlling prior content independently from the final transition; paper does not establish the cause of our errors.'),
 fixed=dict(threshold=0,new_fits=0,tracker_calls=0,dt=1/60,steps_per_observation=2),deadline='2026-11-12T07:40:00+03:00',
 limitations='Five reused engineering cases, no generalization claim. Counterfactual constant prefixes are not the true missing scene; frozen heads may be out of distribution. Loss of useful motion is possible. Effects identify this content intervention, not a specific neuron, proof of harmful natural memory, or complete ID/HOTA improvement. Initial fade-in depends on unchanged first input, which this experiment does not test.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json')
d['experiments'].append(dict(id=ID,title='Предыстория FlyVis — неизменный последний переход и раздельная история образца и кандидата',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить действие раннего содержимого истории от последнего перехода и времени; не приписывать эффект отдельному нейрону.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref=OUT.relative_to(ROOT).as_posix()+'/protocol.json')]))
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): заменить только содержимое более ранних входов, сохранив первый и последние два, время и веса; две фиксированные замены. Раздельно история образца, кандидата и обе. Пять известных случаев, исходный и неизменяющий контроли; прежние отрицательные проверки памяти учтены. Трекер и обучение не запускаются. Протокол: '+OUT.relative_to(ROOT).as_posix()+'/protocol.json.')
print('REGISTERED',len(inputs))
