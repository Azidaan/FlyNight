from common import *
import shutil,subprocess
old=read(ROOT/'research_time_semantics_probe_20261112/protocol.json');inputs=dict(old['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
for n in ['common.py','setup_probe.py','register.py','worker.py','audit.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['tools/flyvis-runtime/flyvis/network/dynamics.py','tools/flyvis-runtime/flyvis/network/stimulus.py','tools/flyvis-runtime/flyvis/datasets/rendering/eye.py']:
    inputs[n]=sha(ROOT/n)
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=old['cases'],
 question='Does the extra rejection of own232 disappear after a fixed one-second synthetic hold of each unchanged final retinal image?',
 hypothesis='A transient response to the recent image transition may hurt identity comparison; a held terminal image separates this from a response that persists under static stimulation.',
 action='Consider a separate terminal-image response for identity comparison, keeping motion processing separate. Reject this correction if the predeclared local criterion fails. No tracker deployment.',
 component='Response timing and input history of the complete FlyVis module; frozen neural core and frozen decision head.',
 modes=['index','identity','history_hold','terminal_hold'],
 pairs=[['index','index'],['identity','identity'],['history_hold','index'],['index','history_hold'],['history_hold','history_hold'],['terminal_hold','index'],['index','terminal_hold'],['terminal_hold','terminal_hold']],
 intervention='history_hold appends30 copies of the terminal input to the original prefix. terminal_hold replaces every prefix input by that same terminal input and appends30 copies. Both run identical total update counts, steps2 and dt1/60. Duration1s matches existing fade-in, fixed before results; no duration sweep. Original and clone identity first. Same sequences applied to ordinary temporal/spatial controls. No future image or new shift.',
 controls='Check actual loaded connectome coordinates for every input and output type and cached geometry before core replay. Reproduce all10 original core maps within1e-4 and ordinary maps exactly before interventions. Exact clone identity; immutable weights/inputs; independent scores. Save final whole-network states and last-step residuals: a finite hold is not presumed a steady state.',
 acceptance=['Loaded input/output coordinate order and saved geometry match exactly; original maps and clone identity pass.','Weights and terminal images immutable; independent schedule and score checks pass.','Primary: either BOTH-endpoint held condition accepts232 while preserving70/1102. Secondary: also accepts832 and rejects574. No case-specific mode selection.'],
 fixed=dict(threshold=0,new_fits=0,tracker_calls=0,dt=1/60,steps_per_observation=2,hold_observations=30,hold_seconds=1.0),
 literature=dict(title='Connectome-constrained networks predict neural activity across the fly visual system',authors='Lappalainen et al.',year=2024,url='https://www.nature.com/articles/s41586-024-07939-3',role='Recurrent visual dynamics and optical-flow training motivate distinguishing transient from held-input responses; the paper does not establish our failure cause.'),
 prior_tests=['History-content check preserved last transition; this probe removes it in terminal_hold.','Time-semantics check preserved terminal transition and did not hold terminal input for1s.','Fixed-crop check preserved actual changes in final images.'],
 deadline='2026-11-12T07:40:00+03:00',limitations='Five reused engineering cases, not independent generalization. Synthetic hold is not real elapsed video or future evidence. Frozen decision heads may be distribution-shifted. Changed scores alone do not identify a defective neuron. No claim of convergence without small residual and agreement between histories. Whole-network states checked, but no independent dynamics implementation.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json');rel=OUT.relative_to(ROOT).as_posix()
d['experiments'].append(dict(id=ID,title='Переходная реакция FlyVis — удержание конечного изображения при неизменном решающем правиле',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить переходную реакцию на последнее изменение изображения от сохраняющейся оценки его представления.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref=rel+'/protocol.json')]))
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): проверить влияние переходной реакции, удерживая конечное изображение одну модельную секунду; сравнить исходную и постоянную предысторию. До вмешательства сверить координаты реально загруженной модели и воспроизвести исходные карты. Пять известных случаев, неизменные порог, веса и сдвиги; полный трекер не запускается. Протокол: '+rel+'/protocol.json.')
print('REGISTERED',len(inputs))
