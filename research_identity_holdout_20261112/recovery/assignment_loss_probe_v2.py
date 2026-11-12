"""One saved1160 matrix: scalar source-formula audit and row-removal probe."""
from common import ROOT, OUT, read, sha, write, now
import gzip, math
import numpy as np
import lap
from scipy.optimize import linear_sum_assignment
assert (OUT / 'GO_assignment_loss_v2.json').exists(), 'Await parent registration of this new diagnostic'
protocol = read(OUT / 'assignment_loss_protocol_v2.json')
for path, digest in protocol['inputs'].items():
    assert sha(ROOT / path) == digest, path
assert sha(__file__) == protocol['worker_sha256']
with gzip.open(ROOT / protocol['archive'], 'rt', encoding='utf8') as stream:
    event = next(e for e in __import__('json').load(stream)['events'] if e['frame'] == 1160)
with np.load(ROOT / protocol['detections'], allow_pickle=False) as stream:
    raw = {key: stream[key].copy() for key in ['xyxy', 'conf', 'frame_idx', 'cls']}
native = read(ROOT / protocol['native_protocol'])
threshold = native['native_config']['match_thresh']
assert threshold == .8 and native['native_config']['fuse_score'] is True
row_ids = [row['tid'] for row in event['rows']]
columns = list(event['high_raw_ids'])
original = np.asarray(event['C'], np.float32)
corrected = np.asarray(event['corrected'], np.float32)

def box(row):
    mean = row['mean']
    left, top = mean[0] - mean[2] / 2, mean[1] - mean[3] / 2
    return [left, top, left + mean[2], top + mean[3]]

def scalar_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / (union + 1e-7)

def typed_iou(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    widths = np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), np.float32(0))
    intersection = np.float32(widths[0] * widths[1])
    union = np.float32(np.float32((a[2] - a[0]) * (a[3] - a[1])) + np.float32((b[2] - b[0]) * (b[3] - b[1])) - intersection)
    return np.float32(intersection / np.float32(union + np.float32(1e-7)))

def typed_fused(a, b, confidence):
    value = typed_iou(a, b)
    geometry_cost = np.float32(1 - value)
    recovered_similarity = np.float32(1 - geometry_cost)
    return np.float32(1 - np.float32(recovered_similarity * np.float32(confidence)))


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

double_cost = np.array([[1 - scalar_iou(box(row), native_boxes[raw_id].astype(float)) * float(raw['conf'][raw_id]) for raw_id in columns] for row in event['rows']])
typed_cost = np.array([[typed_fused(box(row), native_boxes[raw_id], raw['conf'][raw_id]) for raw_id in columns] for row in event['rows']], np.float32)
typed_error = float(abs(typed_cost - original).max())
double_error = float(abs(double_cost - original).max())
assert typed_error <= 1e-6, typed_error

def solve(matrix):
    _, x, _ = lap.lapjv(matrix, extend_cost=True, cost_limit=threshold)
    return sorted((i, int(j)) for i, j in enumerate(x) if j >= 0)

def independent_objective(matrix, pairs):
    n, k = matrix.shape
    augmented = np.zeros((n + k, n + k), float)
    augmented[:n, :k] = matrix
    augmented[:n, k:] = threshold / 2
    augmented[n:, :k] = threshold / 2
    ii, jj = linear_sum_assignment(augmented)
    reference = math.fsum(float(augmented[i, j]) for i, j in zip(ii, jj))
    actual = math.fsum(float(matrix[i, j]) for i, j in pairs) + threshold / 2 * (n + k - 2 * len(pairs))
    assert abs(reference - actual) < 1e-9
    return abs(reference - actual)

baseline = solve(corrected)
assert baseline == sorted(map(tuple, event['actual_first']))
identity = solve(corrected.copy())
assert identity == baseline
retained = [index for index, tid in enumerate(row_ids) if tid != 2445]
assert len(retained) == len(row_ids) - 1
counter_matrix = corrected[retained].copy()
counter_local = solve(counter_matrix)
counter = sorted((retained[i], j) for i, j in counter_local)
assert np.array_equal(counter_matrix, corrected[retained])
objective_errors = [independent_objective(corrected, baseline), independent_objective(counter_matrix, counter_local)]
to_pairs = lambda pairs: [dict(tid=row_ids[i], raw_id=columns[j], cost=float(corrected[i, j])) for i, j in pairs]
target_column = columns.index(109699)
rows = []
for tid in [2179, 2445]:
    index = row_ids.index(tid)
    predicted = box(event['rows'][index])
    candidate = native_boxes[109699].astype(float).tolist()
    rows.append(dict(tid=tid, row=index, predicted_xyxy=predicted, candidate_raw_id=109699, candidate_box=candidate, candidate_raw_box=raw['xyxy'][109699].astype(float).tolist(),
                     detection_confidence=float(raw['conf'][109699]), scalar_float64_iou=scalar_iou(predicted, candidate), typed_float32_iou=float(typed_iou(predicted, candidate)),
                     reconstructed_float32_cost=float(typed_cost[index, target_column]), reconstructed_float64_cost=float(double_cost[index, target_column]),
                     saved_base_cost=float(original[index, target_column]), saved_corrected_cost=float(corrected[index, target_column]),
                     last_history=event['rows'][index]['history'][-1], state=event['rows'][index]['state'], cls=event['rows'][index]['cls']))
archived_sources = native.get('sources', {})
source_matches = {path: dict(current_sha256=digest, archived_sha256=archived_sources.get(path), matches=archived_sources.get(path) == digest if path in archived_sources else None) for path, digest in protocol['inputs'].items() if path.endswith('.py')}
for path, digest in protocol['inputs'].items():
    assert sha(ROOT / path) == digest, path
write('assignment_loss_result_v2.json', dict(created=now(), rows=rows, matrix_shape=list(original.shape), changed_by_neural_readout=int(np.count_nonzero(original != corrected)),
                                         full_matrix_float32_formula_max_error=typed_error, full_matrix_scalar_float64_formula_max_error=double_error,
                                         baseline_pairs=to_pairs(baseline), without2445_pairs=to_pairs(counter),
                                         raw109699_baseline_owner=[row_ids[i] for i, j in baseline if j == target_column],
                                         raw109699_without2445_owner=[row_ids[i] for i, j in counter if j == target_column],
                                         target2179_without2445_assignment=[pair for pair in to_pairs(counter) if pair['tid'] == 2179],
                                         changed_pairs=dict(removed=[p for p in to_pairs(baseline) if (p['tid'], p['raw_id']) not in {(q['tid'], q['raw_id']) for q in to_pairs(counter)}],
                                                            added=[p for p in to_pairs(counter) if (p['tid'], p['raw_id']) not in {(q['tid'], q['raw_id']) for q in to_pairs(baseline)}]),
                                         independent_solver_objective_max_error=max(objective_errors), source_fingerprint_comparison=source_matches,
                                         no_GT_identity_claim_at1160=True, full_tracker_calls=0, model_calls=0, new_weights=0,
                                         scope='Archivedfirst-assignment algebra only; removal of an active competitor is a diagnostic, not a valid deployedrepair or evidence that109699 is the ownobject at unannotated1160.'))
print('ASSIGNMENT_LOSS_DIAGNOSTIC_COMPLETE', typed_error, flush=True)
