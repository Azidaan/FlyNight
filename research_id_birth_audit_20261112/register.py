from common import *
import shutil,subprocess
native=ROOT/'runs/domain_matched_native_20261109'
execution=read(native/'execution_protocol.json');inputs={};source_checks={}
names=['runs/domain_matched_native_20261109/native_run.py','runs/domain_matched_native_20261109/runtime_adapter.py','runs/full_template_native_20261109/native_run.py','runs/full_template_native_20261109/native_runtime.py','tracking/spatial_identity_assignment.py','tracking/xywh_bytetrack.py','.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py']
for n in names:
    digest=sha(ROOT/n);assert digest==execution['sources'][n],n
    inputs[n]=digest;source_checks[n]=dict(saved=digest,current=digest,match=True)
cases=[dict(video='b23b2649-8349d2a1',ids=[232,474],target='00130234',endpoint=544),dict(video='b1cd1e94-549d0bfe',ids=[832,773],target='00118997',endpoint=540)]
for case in cases:
    v=case['video'];receipt=read(native/'rollouts/C34'/(v+'.receipt.json'))
    for suffix in ['.npz','.events.json.gz']:
        n=f'runs/domain_matched_native_20261109/rollouts/C34/{v}{suffix}';assert sha(ROOT/n)==receipt['artifacts'][n];inputs[n]=sha(ROOT/n)
    for n in [f'data/bdd100k/labels/box_track_20/val/{v}.json',f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{v}.json',read(native/'protocol.json')['videos'][v]['detections']]:inputs[n]=sha(ROOT/n)
for n in ['protocol.json','execution_protocol.json']:
    path=native/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['common.py','register.py','trace.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=cases,source_checks=source_checks,archived_head=execution['head'],question='When did the competing IDs first appear and first overlap the same annotated object, and which association stages are evidenced?',method='Read all saved assignments and public outputs for both videos. Verify literal raw origins. Independently label public boxes and raw detections only at exact annotation anchors with unique noncrowd-car IoU>=0.5. Keep every annotation match, ambiguity and unannotated frame. Inventory saved full matrices at births and ownership changes. No interpolation of annotation or assumption that lower ID is canonical.',acceptance=['Archived file hashes and owning sources verified.','All four ID histories traced without discarding unknowns; public output and raw provenance checked.','State exactly which birth/assignment mechanisms are observed, source-implied, or unresolved due to missing matrix/state.'],fixed=dict(tracker_calls=0,core_calls=0,threshold_changes=0),limitation='Observational archival audit, not a new tracker experiment or HOTA result. Sparse proposal-only matrices may not identify the full causal path at birth.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';d=read(path);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json');rel=OUT.name
d['experiments'].append(dict(id=ID,title='Происхождение конкурирующих ID до отказов 232 и832',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis='Поздний отказ восстановления может происходить после уже возникшего раздвоения истории объекта; это проверяемая версия, а не установленная причина.',expected='Отделить позднюю оценку пары от момента рождения и смены конкурирующих номеров.',acceptance=[dict(criterion=x,met=None) for x in p['acceptance']],evidence=[dict(kind='file',ref=rel+'/protocol.json')]))
path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','История и назначения','Происхождение конкурирующих номеров'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): проследить рождение и аннотированную историю ID232/474 и832/773 до поздних отказов. Сверить происхождение детекций и наличие полных матриц на критических кадрах; неизвестные соответствия не заполнять догадками. Архивный разбор без запуска трекера, ядра и изменения порогов. Источник '+rel+'/protocol.json.')
print('REGISTERED',len(inputs))
