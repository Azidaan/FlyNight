"""Preserve failed provenance assertion and register an output-only correction."""
from common import *
start('cpu',timeout=600)
import psutil
old=ROOT/'runs/flyvis_repairs_20261106/unification_20261107/uncapped_checks/83fe466a7d3c469b900a6d043b7f184a'
guard=read(old/'memory_guard.json');assert guard['returncode']==1 and guard['reason'] is None
for pid,birth in guard['owned'].items():
    try:assert psutil.Process(int(pid)).create_time()!=birth
    except psutil.NoSuchProcess:pass
assert read(OUT/'comparison_report.json')['passed'] and read(OUT/'deployment_audit.json')['passed']
p=read(OUT/'protocol.json')
files=[OUT/n for n in ['register_fate_recovery.py','repair_public_provenance.py','id_fate_v2.py','provenance_math_test.py',
    'recovery3_queue.py','common.py','status_write_recovery.py','id_fate.py','comparison_report.json','deployment_audit.json',
    'execution_protocol.json','evaluation_protocol.json','register_boundary.py','boundary_audit.py']]
files += [old/'memory_guard.json',old/'memory_guard.worker.log',
    ROOT/'.venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py',OUT/'native_runtime.py']
for video in p['videos']:
    for arm in p['arms']:
        files += [OUT/'rollouts'/arm/(video+'.receipt.json'),OUT/'rollouts'/arm/(video+'.npz'),
            OUT/'rollouts'/arm/(video+'.events.json.gz')]
write('fate_recovery_protocol.json',dict(registered=datetime.now(MSK).isoformat(),
    failure='Recorded export_raw_id can remain at the birth detection when native _unconfirmed_association directly calls track.update and bypasses _apply_match. The low-score _second_association actually DOES use _apply_matches; earlier commentary attributing it to low-score matching was incorrect. Exact frame assertion correctly stopped the first fate analysis.',
    scope='Logging provenance only. Native predicted boxes/IDs, classifier inputs/histories and all80official metric results are unchanged. Do not rerun tracker, maps, training, orTrackEval.',
    correction='For every exported assignment keep a logged pointer only when it belongs to the actual current frame and, if public, agrees with actual output score/class. Otherwise recover only from UNIQUE exact public frame+float32 score+class. Preserve the original pointer. No nearest boxes or GT used in reconstruction; ambiguous/nonpublic unresolved pointers becomeNone, not invented. Direct reference labels require valid public provenance. Candidate proposals remain their exact current high raw IDs.',
    source='Installed byte_tracker.py _unconfirmed_association calls update directly; _second_association calls _apply_matches. New raw_provenance field makes the correction inspectable. All fate conditions/GTIoU.5/unknown treatment otherwise unchanged.',
    sources={x.relative_to(ROOT).as_posix():sha(x) for x in files},save_stop=STOP.isoformat()))
journal('Исправляется происхождение публичных рамок в разборе судьбыID',
    'Полныйпилотзавершил80новыхоценок: критерийнепройден, `runs/full_template_native_20261109/comparison_report.json`. ПриразбореIDнепрошлапроверкакадраисходнойрамки: export_raw_id сохранялрамкурожденияпослепервогоподтверждения. Прямойкод `_unconfirmed_association` обходит `_apply_match`; `_second_association` егоиспользует, поэтому мояпредыдущаяформулировкапронизкуюуверенностьбыланеточной. Ошибкаотноситсякжурналу, а не к траекториям/входамсчитывателя. Паспорт `runs/full_template_native_20261109/fate_recovery_protocol.json` фиксируетвосстановлениетолькопоуникальнымточнымкадру/оценке/классуизготовогопубличноговыхода, безближайшихрамок/GT; неразрешённыеостаютсянеизвестными. Никакиеготовыеинференсы,траектории и метрики не повторяются; старыйсбойсохранён.')
print('Direct fate provenance recovery registered',flush=True)
