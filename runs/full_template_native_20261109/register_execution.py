"""Pin the runtime domain before fitting final heads or native outcomes."""
from common import *
start('cpu',timeout=600)
import collections,subprocess
assert not (OUT/'execution_protocol.json').exists()
p=read(OUT/'protocol.json');verify(p)
assert read(OUT/'input_witness_recovery.json')['all_arrays_exact']
rows=read(TRAIN/'features.json')['records']
gaps=dict(collections.Counter(r['geometry'][1] for r in rows));assert gaps=={6.:89,5.:2}
files=[OUT/n for n in ['register_execution.py','native_runtime.py','native_run.py','deploy.py',
    'common.py','sparse_views.py','sequential_views.py','protocol.json','input_recovery_protocol.json','input_witness_recovery.json']]
files += [ROOT/n for n in p['sources'] if n.endswith('.py')]
files += [TRAIN/'features.json',ROOT/'runs/historical_direct_query_inventory_20261109/inventory.py',
    ROOT/'runs/identity_channel_head_20261109/protocol.json',ROOT/'runs/z13_local_peak_20261021/geometry.json',
    ROOT/'tracking/observed_spatial_reader.py',ROOT/'tracking/hex_linear_interpolation.py',
    ROOT/'tracking/channel_balanced_correspondence.py',ROOT/'tracking/z13_region_assignment.py',
    ROOT/'tracking/z13_readout.py',ROOT/'tracking/foveated_eyes.py',ROOT/'tracking/offline_association.py',
    ROOT/'datasets/bdd100k_loader.py',ROOT/'scripts/z13_target_model.py',ROOT/'scripts/run_bdd_video_rate_experiment.py',
    ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',
    ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json',ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json']
files += list((ROOT/'tools/flyvis-runtime/flyvis/network').glob('*.py'))
files += [LEGACY/'runtime_inputs'/(v+'.json') for v in p['videos']]
training_videos={r['video'] for r in rows};assert len(training_videos)==32 and not training_videos.intersection(p['videos'])
write('execution_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    amendment_before_any_native_outcome='Respect the OBSERVABLE historical training domain, identically for all4 new arms: car, original native FIRST row unmatched, at least2distinct history times, last-observation gap5or6, old19-shift support>=3. Shadow native solver chooses eligible ROWS only; candidate COLUMNS and all rival rows remain in global actual assignment. Other cases retain native costs. No outcome-based threshold tuning.',
    justification='The source inventory excluded already-matched native-first rows and last-observation gaps>6. Actual91 training gaps are89at6 and2at5. Applying a learned verifier up to track_buffer30 would extrapolate outside observed support. Domain guard fixed before any new model/native result.',
    unobservable_selection_limitation='Historical training also excluded another public native detection with the same GT at reference. That GT-based exclusion cannot be reproduced causally and is NOT used online; report this population limitation, no invented proxy.',
    training_gaps=gaps,training_videos=sorted(training_videos),validation_videos=list(p['videos']),
    models='One all91 final fit eachG7,C34,R134spatial,R134temporal, samelambda1/freeintercept/classbalancedloss/zero threshold; original7 geometry normalized on all91 training rows. Undefined signal uses G7. No validation labels in fit or runtime.',
    cache='Only closed matching legacy receipt hashes authorize R1/C8 reuse. New C34 maps have causal retinal-source and execution passports; native checkpoints restore retinal processing, never core state. Core rebuilt from same four warm views+causal search sequence. Compare available old C8 tolerance1e-4. No new full-video processed arrays.',
    parity='Two FULL native equivalence passes on all20: alpha0 disabled and alpha.2 active with no accepted proposals. Both raw arrays and annotatedJSON must be exactly equal. Same runtime and native source passports used by active arms.',
    zero_protocol=p['zero_gate'],statistics=p['statistics'],acceptance=p['acceptance'],save_stop=STOP.isoformat(),
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files}))
journal('Исполнитель проверки реальныхID: зафиксирована область обученных связей',
    'Точность восстановленных входов подтверждена на всех20видео:21вид совпал с готовым массивом точно, `runs/full_template_native_20261109/input_witness_recovery.json`. До любого нового исхода назначения зарегистрирован исполнитель и единое ограничение области дляG7, FlyVis34 и двух входных контролей: толькоcar, строка без исходного первого назначения,≥2времени истории, пропуск5или6кадров и прежняя опора≥3ячейки. В91обучающейсвязи89пропусков6кадров и2пропуска5кадров, `runs/full_template_identity_20261109/features.json`. Это защита от переноса на необученные30кадров; пороги не менялись, чужие столбцы не замораживаются. Историческое исключение поGTне используется онлайн. Паспорт `runs/full_template_native_20261109/execution_protocol.json`. Далее четыре окончательные модели и два полных нулевых контроля; новых карт/назначений пока нет.')
status('execution_registered',training_pairs=91,zero_runs_pending=40)
print('EXECUTION REGISTERED:4 fixed models,40 zero runs,training domain pinned',flush=True)
