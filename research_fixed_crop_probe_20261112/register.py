from common import *
import shutil,subprocess
old=read(ROOT/'research_time_semantics_probe_20261112/protocol.json');inputs=dict(old['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
for n in ['common.py','register.py','render.py','worker.py','audit.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['research_time_semantics_probe_20261112/protocol.json','research_surround_dependency_probe_20261112/crop_observation.json','runs/full_template_native_20261109/sparse_views.py','runs/full_template_native_20261109/sequential_views.py','tracking/retina.py']:
    inputs[n]=sha(ROOT/n)
videos=read(LEGACY/'protocol.json')['videos'];inputs[(LEGACY/'protocol.json').relative_to(ROOT).as_posix()]=sha(LEGACY/'protocol.json')
for c in old['cases']:
    video=c['case']['video'];entry=videos[video];name=entry['video_path'];inputs[name]=sha(ROOT/name);assert inputs[name]==entry['sources'][name]
    marker=LEGACY/'processed'/(video+'.json');inputs[marker.relative_to(ROOT).as_posix()]=sha(marker)
    for f in (LEGACY/'processed'/(video+'.checkpoints')).glob('*.json'):
        m=read(f);inputs[f.relative_to(ROOT).as_posix()]=sha(f);state=f.parent/m['state_file'];inputs[state.relative_to(ROOT).as_posix()]=sha(state);assert inputs[state.relative_to(ROOT).as_posix()]==m['state_sha256']
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=old['cases'],
 question='Does changing crop coordinates before recurrent processing cause the extra rejection of own232, compared with processing the same terminal images with fixed endpoint crops?',
 hypothesis='Changes in crop origin and scale induce retinal changes in addition to scene motion. Re-rendering prior observations in fixed endpoint coordinates may remove that component before the core.',
 action='Test fixed-coordinate history extraction as one possible correction. Keep terminal images, comparison coordinates and frozen heads unchanged; no tracker deployment.',
 component='Input rendering of the full FlyVis module before recurrence; same global retina preprocessing and frozen core.',
 modes=['index','identity','last_pair_fixed','all_fixed'],
 pairs=[['index','index'],['identity','identity'],['last_pair_fixed','index'],['index','last_pair_fixed'],['last_pair_fixed','last_pair_fixed'],['all_fixed','index'],['index','all_fixed'],['all_fixed','all_fixed']],
 intervention='For each endpoint separately, last_pair_fixed re-renders only the penultimate observed frame in the endpoint crop; all_fixed re-renders every prefix frame in that crop. Terminal frame/crop exact; original frame list, observation-index dt, preprocessing and count unchanged. Reference and current endpoints may use different fixed crops. No estimated flow, landmark matching, new shift or future frame. Endpoint crop is available when recomputing its history; this is retrospective prefix extraction, not a zero-cost streaming correction.',
 controls='Re-render every original retinal view from sequential native decoding and require exact float16 parity BEFORE changed inputs. Alternative patch averaging checks every rendered view against NumPy sums within0.001. All original core maps gated before changed core passes. Cloned identity. Same changed input histories applied to both ordinary controls. Reference/current/both interventions reported.',
 acceptance=['Exact original retinal restoration and terminal input parity; independently aggregated rendered views error<=0.001; all ten original core maps parity<=1e-4 and ordinary maps exact before changed core passes.','Identity exact, immutable inputs/weights, independent reconstruction of all frame/crop schedules and independent scores within1e-9.','Primary local criterion: either BOTH-endpoint condition accepts232 and preserves70/1102. Secondary: additionally accept832 and reject574. Every case, unilateral effect and ordinary control reported; no selection of mode per case.'],
 prior_tests=[dict(id='coordinate_frozen_reader_screen_20261109',difference='Aligned output maps after their formation; current experiment changes input coordinates before recurrence.'),dict(id='saved_shift_cross_probe_20261112',difference='Crossed saved output comparison shifts; current shifts remain frozen.'),dict(id='history_content_probe_20261112',difference='Artificial constant early inputs; this probe uses actual video frames in a declared physical crop.')],
 literature=dict(title='Connectome-constrained networks predict neural activity across the fly visual system',authors='Lappalainen et al.',year=2024,url='https://www.nature.com/articles/s41586-024-07939-3',role='Training uses visual sequences and optical flow. Coordinate changes in rendered sequences must be distinguished from downstream map alignment; paper does not prove our failure cause.'),
 fixed=dict(threshold=0,new_fits=0,tracker_calls=0,dt=1/60,steps_per_observation=2),deadline='2026-11-12T07:40:00+03:00',
 limitations='Five reused engineering cases. Fixed crop can lose an object previously followed by the moving crop; preserving terminal images does not preserve all earlier content. Global camera motion remains. New histories change initial adaptation and ordinary temporal features, and may shift frozen-head distributions. A positive local result does not identify a neuron or establish ID/HOTA benefit. No claim that a fixed crop reconstructs physical correspondences.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json');rel=OUT.relative_to(ROOT).as_posix()
d['experiments'].append(dict(id=ID,title='Входное окно FlyVis — неподвижные координаты истории при сохранении конечных наблюдений',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Отделить смену координат входа до ядра от последующего совмещения карт.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref=rel+'/protocol.json')]))
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): повторно получить реальные прошлые входы в координатах конечного окна; отдельно только последний переход и вся история. Сначала точное восстановление всех исходных входов из видео. Конечные наблюдения, веса, пороги и сдвиги сохранены; пять известных случаев, главная локальная мишень232 и обязательные остальные контроли. Полный трекер не запускается. Протокол: '+rel+'/protocol.json.')
print('REGISTERED',len(inputs))
