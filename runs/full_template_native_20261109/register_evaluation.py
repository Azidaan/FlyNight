"""Pin unchanged official metrics and direct-ID fate definitions."""
from common import *
start('cpu',timeout=600)
assert not (OUT/'evaluation_protocol.json').exists()
p=read(OUT/'protocol.json')
files=[OUT/n for n in ['register_evaluation.py','evaluate_native.py','native_statistics.py','id_fate.py','deployment_audit.py',
    'execution_protocol.json','deployment.json','zero_complete.json']]
files += [ROOT/'evaluation/hota_eval.py',ROOT/'runs/frozen_reader_transfer_20261109/external_inventory.py']
files += list((ROOT/'.venv/Lib/site-packages/trackeval').rglob('*.py'))
for video,entry in p['videos'].items():
    files += [ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json',LEGACY/'runtime_inputs'/(video+'.json')]
    for arm in ['B0','G']:
        files += [LEGACY/'metrics'/arm/(video+'.json'),LEGACY/'metrics'/arm/(video+'.details.json.gz')]
write('evaluation_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    execution_sha256=sha(OUT/'execution_protocol.json'),videos=list(p['videos']),
    primary_comparisons=p['primary_comparisons'],statistics=p['statistics'],acceptance=p['acceptance'],
    official='Unchanged evaluation/hota_eval.py, TrackEvalHOTA/CLEAR/Identity, splitval, all8validclasses. New4arms80per-video results; both oldB0/G component files and details reused, not reevaluated.',
    ID_fate='All logged proposals, including domain refusals, keep their original key. Direct own/foreign only if the actual published reference observation and current raw candidate have UNIQUE noncrowdcarGT IoU>=.5 at EXACT clock annotation frames. No interpolated labels. Use recorded raw assignment provenance, never nearest-box substitution. Head refusal, global match, emitted oldID and future annotated old-ID observations reported separately. Compare native IDs via reference raw observation, not equal numeric IDs across diverged runs. Unknowns explicit; repeated proposals are not independent objects.',
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files}))
print('Metric and ID-fate execution registered',flush=True)
