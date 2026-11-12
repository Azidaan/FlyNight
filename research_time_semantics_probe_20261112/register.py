from common import *
import shutil,subprocess
import numpy as np
from scripts.z13_target_model import receptor_features_rich
ag=read(ROOT/'research_agreement_readout_20261112/protocol.json');old=read(ROOT/'runs/domain_matched_native_20261109/execution_protocol.json');inputs={};cases=[]
def bind(n):
    path=Path(n);path=path if path.is_absolute() else ROOT/path;n=path.relative_to(ROOT).as_posix();inputs[n]=sha(path);return n
for n in ['common.py','register.py','worker.py','audit.py']:bind(OUT/n)
for n in ['research_agreement_readout_20261112/protocol.json','research_agreement_readout_20261112/predictions.json','research_saved_shift_probe_20261112/time_path_observation.json','runs/domain_matched_native_20261109/execution_protocol.json','runs/flyvis/results/flow/0000/000/_meta.yaml','tracking/flyvis_adapter.py','tracking/flyvis_checkpoint.py','tracking/flyvis_activity.py','tracking/flyvis_motion.py','tracking/gpu_lock.py','tracking/channel_agreement_correspondence.py','runs/z13_local_peak_20261021/geometry.json','runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py']:bind(n)
for n,h in old['sources'].items():
    if n.startswith(('tracking/','scripts/','tools/flyvis-runtime/flyvis/network/')) or n in ['runs/flyvis/results/flow/0000/000/best_chkpt','runs/z13_target_center_dev_20261018/C/normalization.json','runs/z13_target_center_dev_20261018/R1/normalization.json']:
        bind(n);assert inputs[n]==h,n
norm=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json');mu,sd=[np.array(norm[k])[:34] for k in ['mean','sd']]
for r in ag['screen']:
    ref,cur=[ag['snapshots'][k] for k in r['keys']];views=[]
    cm=Path(next(v['path'] for v in cur['files'] if v['slot']=='C34')).with_suffix('.json');meta=read(ROOT/cm);bind(cm)
    for frame,crop in cur['sequence']:
        key=hashlib.sha256(json.dumps([int(frame),crop],separators=(',',':')).encode()).hexdigest()[:32]
        names=[n for n in meta['retinal_sources'] if Path(n).stem==key];assert names,(r['old_id'],frame)
        for n in names:bind(n);assert inputs[n]==meta['retinal_sources'][n]
        arrays=[np.load(ROOT/n,allow_pickle=False) for n in names];assert all(np.array_equal(arrays[0],a) for a in arrays)
        assert arrays[0].shape==(721,) and arrays[0].dtype==np.float16 and np.isfinite(arrays[0]).all()
        views.append(names[0])
    hx=np.stack([np.load(ROOT/n,allow_pickle=False) for n in views]).astype(np.float32)
    assert cur['sequence'][:len(ref['sequence'])]==ref['sequence']
    endpoints=[]
    for snapshot,length in [(ref,len(ref['sequence'])),(cur,len(cur['sequence']))]:
        paths={v['slot']:v['path'] for v in snapshot['files']};assert set(paths)=={'C34','R1'}
        for n in paths.values():bind(n);assert inputs[n]==ag['inputs'][n]
        actual=((receptor_features_rich(hx[None,:length].astype(float))[-1][0].T-mu)/sd).astype(np.float32)
        assert np.array_equal(actual,np.load(ROOT/paths['R1'],allow_pickle=False)),(r['old_id'],length)
        endpoints.append(dict(length=length,paths=paths))
    cases.append(dict(case=r,sequence=cur['sequence'],views=views,endpoints=endpoints,ordinary_input_witness_exact=True))
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=cases,
 question='Does changing observation-index time, while keeping recorded observations and all scoring rules fixed, resolve the known identity failures?',
 hypothesis='Fixed-step replay compresses missing frame intervals and advances padded duplicates. Separating these changes may alter the incorrect spatial evidence.',
 component='Temporal input replay of the full FlyVis module; frozen recurrent weights. New diagnostic activations only, no training or tracker.',
 modes=['index','identity','gaps_linear','gaps_hold','dedup_only','elapsed_linear','elapsed_hold'],
 intervention='index: original one step per adjacent observation. identity: cloned original input. gaps_*: insert delta-1 nominal frames only for positive gaps, preserve duplicate padding. dedup_only: remove zero-time duplicates, leave positive gaps compressed. elapsed_*: combine both. Linear fills intermediate retinal arrays by interpolation of endpoints; hold repeats the preceding retinal array. Neither is claimed to reconstruct the missing image. All endpoints are at or before the decision frame.',
 controls='All five cases, original/reference map parity before any interventions; exact identity control. Same temporal inputs also drive both ordinary-image controls. Saved query coordinates and fitted agreement heads fixed per representation; no shift search.',
 acceptance=['Original10C34maps match cached maps within1e-4 and all10ordinary maps exactly; stop before interventions on failure.','Cloned-input identity maps/scores exact; independent CPU reader and alternative interpolation reproduce all105scores within1e-9; all inputs and weights unchanged.','For either elapsed mode individually: retain correct own70/1102, accept at least one of own232/832 and reject foreign574 at fixed threshold0. Report every mode and case; never choose a mode for each case.'],
 fixed=dict(new_fits=0,threshold=0,dt=1/60,steps_per_nominal_frame=2,tracker_calls=0),
 limits='Five previously inspected development cases. Interpolation/hold are explicit counterfactual assumptions; missing physical images not recovered. Frozen heads trained on index-time data may have distribution mismatch. New-map effects do not prove a specific neural mechanism or trajectory improvement.',
 computational_budget='One guarded GPU worker, expected minutes; stop safely at registered deadline. No simultaneous training.',deadline='2026-11-12T07:40:00+03:00'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json')
d['experiments'].append(dict(id=ID,title='Время наблюдений FlyVis — пропуски и повторы кадров при неизменных весах',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить влияние временного порядка от остальных компонентов; не выдавать заполнение пропусков за реальные изображения.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref='research_time_semantics_probe_20261112/protocol.json')]))
assert raw==path.read_bytes();path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): на пяти известных случаях проверить отдельно интервалы пропусков и повторы кадров; два заполнения пропусков — линейное и удержание предыдущего входа. Точные сохранённые входы восстановлены, обычные карты совпали побайтно. Перед вмешательствами обязательно совпадение штатных карт ядра; веса, пороги и сдвиги неизменны, трекер не запускается. Протокол: research_time_semantics_probe_20261112/protocol.json.')
print('REGISTERED',len(inputs),'EXACT_INPUT_WITNESSES',10)
