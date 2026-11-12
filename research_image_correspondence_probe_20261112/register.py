from common import *
import shutil,subprocess
old=read(ROOT/'research_time_semantics_probe_20261112/protocol.json');inputs=dict(old['inputs'])
for n,h in inputs.items():assert sha(ROOT/n)==h,n
manifest=read(ROOT/'research_failure_localization_20261112/frame_manifest.json')
for r in manifest['records']:
    path=ROOT/'research_failure_localization_20261112'/r['frame_png'];assert sha(path)==r['sha256'];inputs[path.relative_to(ROOT).as_posix()]=r['sha256']
for n in ['common.py','setup.py','register.py','worker.py','audit.py']:
    path=OUT/n;inputs[path.relative_to(ROOT).as_posix()]=sha(path)
for n in ['research_failure_localization_20261112/frame_manifest.json','research_failure_localization_20261112/stage3.json']:
    inputs[n]=sha(ROOT/n)
save('protocol.json',dict(registered=now(),head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),inputs=inputs,cases=old['cases'],
 question='Does independently estimated image displacement, replacing box-relative correspondence on the same saved maps, remove wrong identity decisions?',
 hypothesis='Scaling each reference box into a changed candidate box can compare different visible parts. Independent local image motion may restore correspondence on a verified common subset.',
 action='Test image-based correspondence as a replacement for box-relative coordinate mapping; reject promotion if controls or local criterion fail. No live tracker changes.',
 component='Coordinate mapping of the external readout of the full FlyVis module; no neural-core replay or weight changes.',
 flow=dict(lk_window=[11,11],lk_max_level=2,lk_iterations=30,lk_epsilon=.01,lk_min_eigenvalue=1e-4,dis_preset='MEDIUM',consistency_pixels=1.,minimum_cells=10,minimum_rank=2,threads=2,seed=0),
 modes=['original_full','original_subset','identity_subset','lk','dis','opposite_lk'],
 intervention='Compute raw-grayscale forward/backward LK at original template cells, with zero initial displacement; compute full-frame DIS forward/backward separately. Common subset requires both LK statuses, LK forward-back <=1px, DIS forward-back <=1px, method agreement <=1px, LK/DIS endpoints inside candidate box, and eye-hull support for every compared query. Opposite_LK reverses displacement with same magnitude and no candidate-inside requirement, as deliberately wrong correspondence control. At least10 cells and rank2 required. Conditions fixed before results. GT labels not used to select correspondence.',
 controls='Reproduce all original15 scores before changed comparisons. Original_subset and identity_subset use the identical common mask. Each arm retains its saved original shift for original modes; image-based modes use the same world points across arms. Geometry covariates, weights, normalization, maps and threshold remain fixed. Independently interpolate DIS fields, reconstruct subset and map sampling/scores. Report all cases including insufficient support. No fit or score-based correspondence search.',
 acceptance=['All original scores reproduce within1e-9; unchanged maps and identity subset exact; independent score/interpolation checks within1e-9.','Report readiness and source-to-query discrepancies for all five cases; approximate correspondences never labelled physical ground truth.','Primary local action:232 original_subset rejects, LK and DIS accept, opposite_LK rejects; both own controls70/1102 ready and accepted by LK/DIS. Secondary:832 corrected and574 rejected with all five ready.'],
 fixed=dict(threshold=0,new_fits=0,tracker_calls=0,core_calls=0,new_neural_maps=0),deadline='2026-11-12T07:40:00+03:00',
 literature=[dict(title='An Iterative Image Registration Technique with an Application to Stereo Vision',authors='Lucas and Kanade',year=1981,url='https://courses.cs.duke.edu/cps274/fall11/papers/LucasKanade81.pdf'),dict(title='Fast Optical Flow using Dense Inverse Search',authors='Kroeger et al.',year=2016,url='https://arxiv.org/abs/1603.03590')],
 limitations='Five reused engineering cases; two algorithms can share errors on background, aperture ambiguity, reflections and occlusion. Consistency checks are new diagnostic eligibility rules, not changes to tracker thresholds. Restricting common support can change scores: compare original_subset explicitly and never attribute mask-only gains to alignment. Frozen heads may be distribution-shifted. Missing support is inconclusive, not wrong identity. Candidate box restriction avoids following old object outside the candidate and calling that candidate accepted. No full tracking or HOTA inference.'))
(OUT/'before').mkdir()
for n in ['experiments/project_history.json','experiments/obsidian_experiment_topics.json','experiments/system_problems.json','experiments/useful_literature.json','dissertation/ЕДИНЫЙ_ПЛАН.md']:shutil.copy2(ROOT/n,OUT/'before'/Path(n).name)
path=ROOT/'experiments/project_history.json';raw=path.read_bytes();d=json.loads(raw);assert not any(c['id']==ID for c in d['experiments']);p=read(OUT/'protocol.json');rel=OUT.relative_to(ROOT).as_posix()
d['experiments'].append(dict(id=ID,title='Соответствия деталей — независимое движение изображения при неизменных картах FlyVis',area='flyvis',state='running',started='2026-11-12',question=p['question'],hypothesis=p['hypothesis'],expected='Измерить вклад координат сравнения отдельно от выбора ячеек и изменения нейронных карт.',acceptance=[dict(criterion=s,met=None) for s in p['acceptance']],evidence=[dict(kind='file',ref=rel+'/protocol.json')]))
assert path.read_bytes()==raw;path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
path=ROOT/'experiments/obsidian_experiment_topics.json';d=read(path);d['topics'][ID]=['Сохранение личности','Выбор продолжения среди рамок','Причины неверного сравнения карт'];path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
journal('Постановка (`'+ID+'`): получить соответствия из исходных изображений двумя методами, проверить согласие и возврат точки, затем сравнить карты на одинаковом подмножестве ячеек. Параметры методов и критерии пригодности фиксированы заранее; порог трекера, веса и карты неизменны. Согласие алгоритмов не объявляется точной физической разметкой. Источник: '+rel+'/protocol.json.')
print('REGISTERED',len(inputs))
