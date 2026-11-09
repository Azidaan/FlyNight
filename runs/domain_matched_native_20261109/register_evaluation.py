"""Freeze metric details before evaluation or inspection of new outcomes."""
from evaluation_common import *
os.environ['CUDA_VISIBLE_DEVICES']=''
import psutil
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
import subprocess
assert not (OUT/'evaluation_protocol.json').exists()
p=read(OUT/'protocol.json');old=read(OLD/'evaluation_protocol.json')
for name,digest in old['sources'].items():
    if name.startswith('.venv/Lib/site-packages/trackeval/') or name=='evaluation/hota_eval.py':assert sha(ROOT/name)==digest,name
reused=dict(native_B0='runs/identity_assignment_20261108/full_precision/metrics/B0',legacy_unfiltered_G='runs/identity_assignment_20261108/full_precision/metrics/G',old_G7='runs/full_template_native_20261109/metrics/G7',old_C34='runs/full_template_native_20261109/metrics/C34')
files=[Path(__file__),OUT/'evaluation_common.py',OUT/'evaluate_native.py',OUT/'native_statistics.py',OUT/'id_fate.py',OUT/'evaluation_queue.py',OUT/'execution_protocol.json',OUT/'protocol.json',OUT/'deployment.json',OLD/'comparison_report.json',OLD/'id_fate.json',ROOT/'evaluation/hota_eval.py']
files += [ROOT/name for name in old['sources'] if name.startswith('.venv/Lib/site-packages/trackeval/')]
for video,entry in p['videos'].items():
    files += [ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json',LEGACY/'runtime_inputs'/(video+'.json')]
    for folder in reused.values():
        folder=ROOT/folder;record=read(folder/(video+'.json'));assert record['passed'] and sha(folder/(video+'.details.json.gz'))==record['details_sha256']
        files += [folder/(video+'.json'),folder/(video+'.details.json.gz')]
    files += [ROOT/entry['detections'],ROOT/entry['raw_array']]
sources={f.relative_to(ROOT).as_posix():sha(f) for f in files}
primary=[['C34',a] for a in ['native_B0','G7','R134_spatial','R134_temporal','G_DOMAIN']]
secondary=[['C34','old_C34'],['G7','old_G7'],['C34','legacy_unfiltered_G'],*[[a,'native_B0'] for a in ['G7','G_DOMAIN','R134_spatial','R134_temporal']]]
write('evaluation_protocol.json',dict(registered=datetime.now(MSK).isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
    native_execution_sha256=sha(OUT/'execution_protocol.json'),sources=sources,videos=list(p['videos']),reused_metrics=reused,
    primary_comparisons=primary,secondary_comparisons=secondary,
    statistics='20000 paired video bootstrap draws,seed42,all20fixedvideos. Five primary two-sided99%percentile intervals; secondary95%descriptive. Official pooled HOTA; verify point and first20bootstrap resamples against TrackEval combine_sequences/combine_classes_det_averaged.',
    harm_clarification_before_outcomes='Require nonnegative pooled AND mean-video AssA/IDF1 point differences, and no total IDSw/FP/FN increase against allfiveprimarycontrols. Both aggregations disclosed. This preserves the old mean-video guards and also makes the new protocol point-metric language explicit before outputs.',
    ID_fate='Every logged proposal retained. Own/foreign require unique actual noncrowdcarGT IoU>=.5 at exact annotated current and public reference frames with a verified literal raw origin. Negative raw IDs stay unknown. No nearest-box repair. Record head,correctedcost,columnrival,globalassignment,dedup,publicoldID and future actual annotated observations. Link cross-runIDs by reference raw source. Unknown native public origins retained.',
    no_rerun='Evaluate100newarmvideooutputs only. Reuse80oldmetrics/details across4referencearms; no oldtracking/core/fits repeated. No independent final test at this stage.',
    text_check='Methods and objects separated; current and planned results distinguished; no diagnostic ID-success claim.'))
print('EVALUATION_REGISTERED',len(sources),'sources',flush=True)
