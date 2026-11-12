"""Technical geometry-input correction; original failed proof remains intact."""
from common import ROOT, OUT, read, sha, write, now
import ast
old = (OUT / 'assignment_loss_probe.py').read_text(encoding='utf8')
source = old.replace("'GO_assignment_loss.json'", "'GO_assignment_loss_v2.json'").replace("'assignment_loss_protocol.json'", "'assignment_loss_protocol_v2.json'")
insertion = '''
def native_detection_box(raw_id):
    # _DetSlice.xywh computesfloat32 centers/sizes; parse_bboxes concatenates
    # anint64 index (promotingtofloat64); STrack.xywh2ltwh thenstoresfloat32;
    # STrack.xyxy reconstructsright/bottom fromfloat32 tlwh. No trackercreated.
    value = raw['xyxy'][raw_id].astype(np.float32)
    centers = (value[:2] + value[2:]) / np.float32(2)
    sizes = value[2:] - value[:2]
    xywh = np.r_[centers, sizes].astype(np.float64)
    xywh[:2] -= xywh[2:] / 2
    tlwh = xywh.astype(np.float32)
    result = tlwh.copy()
    result[2:] += result[:2]
    return result

native_boxes = {raw_id: native_detection_box(raw_id) for raw_id in columns}
'''
source = source.replace('double_cost = np.array(', insertion + '\ndouble_cost = np.array(', 1)
source = source.replace("raw['xyxy'][raw_id].astype(float)", "native_boxes[raw_id].astype(float)")
source = source.replace("typed_fused(box(row), raw['xyxy'][raw_id],", "typed_fused(box(row), native_boxes[raw_id],")
source = source.replace("candidate = raw['xyxy'][109699].astype(float).tolist()", "candidate = native_boxes[109699].astype(float).tolist()")
source = source.replace("candidate_box=candidate,", "candidate_box=candidate, candidate_raw_box=raw['xyxy'][109699].astype(float).tolist(),")
source = source.replace("write('assignment_loss_result.json'", "write('assignment_loss_result_v2.json'")
source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
ast.parse(source)
with (OUT / 'assignment_loss_probe_v2.py').open('xb') as stream:
    stream.write(source.encode('utf8'))
protocol = read(OUT / 'assignment_loss_protocol.json')
protocol['technical_recovery_registered'] = now()
protocol['original_protocol_sha256'] = sha(OUT / 'assignment_loss_protocol.json')
protocol['original_failed_execution'] = 'research_identity_holdout_20261112/queue5_execution.json'
protocol['reason'] = 'Original fullmatrix formula witness failed2.1457672119140625e-6 because rawxyxy boxes were used before the actual native xyxy→xywh→tlwh→xyxy arithmetic. Add exact documented initialization geometry; allscientificsettings, rowremoval, sourceframes, costlimit and1e-6 acceptance unchanged. Counterfactual didnotrun inoriginalfailedproof.'
protocol['worker_sha256'] = sha(OUT / 'assignment_loss_probe_v2.py')
for name in ['tracking/offline_association.py', '.venv/Lib/site-packages/ultralytics/trackers/bot_sort.py', '.venv/Lib/site-packages/ultralytics/trackers/utils/stracks.py', '.venv/Lib/site-packages/ultralytics/utils/ops.py']:
    protocol['inputs'][name] = sha(ROOT / name)
write('assignment_loss_protocol_v2.json', protocol)
write('GO_assignment_loss_v2.json', dict(approved=True, registered=now(), protocol_sha256=sha(OUT / 'assignment_loss_protocol_v2.json'),
                                        authorization='Parent explicit instruction permits ownlaunch afterpreregistration oftechnicalinputbugfix; no scientificgate, threshold, weights orintervention change. Originalfailure retained.'))
queue_inputs = dict(protocol['inputs'])
for name in ['assignment_loss_probe_v2.py', 'assignment_loss_protocol_v2.json', 'GO_assignment_loss_v2.json']:
    queue_inputs[str((OUT / name).relative_to(ROOT)).replace('\\', '/')] = sha(OUT / name)
write('queue5b.json', dict(registered=now(), inputs=queue_inputs, runner_sha256=sha(ROOT / 'research_identity_holdout_20261112/run_queue.py'),
                          jobs=[dict(entry='research_identity_holdout_20261112/recovery/assignment_loss_probe_v2.py', timeout=120)], deadline='2026-11-13T07:35:00+03:00'))
print('TECHNICAL_V2_REGISTERED', sha(OUT / 'assignment_loss_protocol_v2.json'), sha(OUT / 'queue5b.json'))
