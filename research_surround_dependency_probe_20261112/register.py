from common import *
import shutil,subprocess
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.z13_readout import HexGeometry
old=read(ROOT/'research_time_semantics_probe_20261112/protocol.json');inputs=dict(old['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
for n in ['common.py','register.py','worker.py','audit.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['research_history_content_probe_20261112/protocol.json','research_history_content_probe_20261112/audit.json','research_time_semantics_probe_20261112/protocol.json']:
    inputs[n]=sha(ROOT/n)
geom=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(geom['points'],geom['eye_size'],[1280,720]);nb=HexGeometry(np.array(geom['points']),geom['eye_size']).neighbors
cases=[]
for c in old['cases']:
    r=c['case'];plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets'])
    base_ref=plan['cells'];base_cur=[]
    for arm in ARMS:
        query=plan['query'][:,r['old_variants'][arm]['winner']];simplex=reader.mesh.find_simplex(query);assert (simplex>=0).all();base_cur.extend(reader.mesh.simplices[simplex].ravel().tolist())
    needed=[np.unique(base_ref),np.unique(base_cur)];protected=[np.unique(np.r_[v,nb[v,1:].ravel()]).tolist() for v in needed]
    assert all(0<len(v)<721 and min(v)>=0 and max(v)<721 for v in protected)
    cases.append(dict(c,protected=protected,direct_readout_cells=[v.tolist() for v in needed]))
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=cases,
 question='Does retinal input outside the exact dependencies of the ordinary local reader cause the additional rejection of own pair232 by the FlyVis path?',
 hypothesis='Lateral and recurrent processing can carry outside-input changes into the compared C34 maps while both ordinary readers remain exactly invariant.',
 action='Test whether suppressing outside variation is a viable specific correction for the extra FlyVis-path error; no tracker deployment or training follows automatically.',
 component='Input intervention before the frozen full FlyVis module. Geometry, normalization, selected shifts and heads remain fixed.',
 modes=['index','identity','outside_gray','outside_frozen'],
 pairs=[['index','index'],['identity','identity'],['outside_gray','index'],['index','outside_gray'],['outside_gray','outside_gray'],['outside_frozen','index'],['index','outside_frozen'],['outside_frozen','outside_frozen']],
 intervention='Protect the union of every retinal cell directly read at the reference endpoint and every triangle vertex sampled at the current endpoint for all three saved shifts, plus one ring of neighbors required by the ordinary spatial reader. Protect these fixed retinal indices at every prefix time. Outside_gray sets all remaining input values to0.5; outside_frozen repeats the first original value of each outside cell. Original prefix lengths and dt remain. Apply separately to reference/current/both. The protected set is a computational dependency region, NOT a physical foreground mask. Endpoint-specific sets use causal box/shift metadata without labels; they need not define one continuous factual history.',
 controls='Untouched cached-map parity for all cases before interventions; cloned identity; two fixed replacements; both ordinary reader scores must stay EXACTLY identical for all conditions. Stop interpretation if any direct control changes. No search for a favorable mask, radius, constant or shift.',
 acceptance=['All untouched10C34 maps match cached maps within1e-4 and10ordinary maps exactly; stop before interventions if not.','Identity exact, protected inputs exact, ordinary control scores exact, independent support reconstruction and score error below1e-9; weights/inputs unchanged.','Primary local diagnostic: at least one BOTH-endpoint replacement accepts own232 while retaining own70/1102. Report other cases and unilateral effects; do not call this resolution of all three errors. Secondary: also accept own832 and reject foreign574, reported separately.'],
 prior_tests=[dict(id='background_patch_gate_20261108',difference='Changed readout using saved target/background examples, did not intervene outside protected retinal dependencies before the core.'),dict(id='flyvis_exclusive_t4t5_v1',difference='Removed shared output cells; this probe preserves output sampling and changes only other input cells.'),dict(id='history_content_probe_20261112',difference='Changed earlier full-field content with final transition fixed; this probe fixes all directly used cells across all times.')],
 literature=dict(title='Connectome-constrained networks predict neural activity across the fly visual system',authors='Lappalainen et al.',year=2024,url='https://doi.org/10.1038/s41586-024-07939-3',section='Spatio-temporal, spatial and temporal receptive fields; Representing the model as a hexagonal convolutional neural network',role='Spatial connections and receptive fields justify testing dependence beyond sampled columns. They do not prove that such dependence caused this identity error.'),
 fixed=dict(threshold=0,new_fits=0,tracker_calls=0,dt=1/60,steps_per_observation=2),deadline='2026-11-12T07:40:00+03:00',
 limitations='Five previously inspected engineering cases. Preserved inputs are computational dependencies, not guaranteed physical object pixels. Intervention can remove useful scene cues and create an artificial boundary; gray changes initial adaptation too. Frozen weights/readouts may face distribution change. Positive result identifies a sufficient input intervention in this architecture, not a neuron, harmful background generally, or ID/HOTA improvement. Negative result does not exclude context within the protected region.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json');rel=OUT.relative_to(ROOT).as_posix()
d['experiments'].append(dict(id=ID,title='Окружение карты FlyVis — неизменные входы обычных признаков и вмешательство вне их области',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить влияние входного окружения через ядро от прямого изменения сравниваемых обычных признаков.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref=rel+'/protocol.json')]))
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): вмешательство вне точной области входных зависимостей обычных признаков. Все нужные ячейки, вершины интерполяции и соседи сохранены на всей истории; две фиксированные замены остальных входов. Главная локальная мишень — дополнительный отказ232 в пути FlyVis; все пять случаев и вред проверяются. Это не маска физического объекта. Веса, пороги и сдвиги неизменны, трекер не запускается. Протокол: '+rel+'/protocol.json.')
print('REGISTERED',[(c['case']['old_id'],list(map(len,c['protected']))) for c in cases])
