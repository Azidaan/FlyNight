"""Last registered matrix diagnostic; no live tracker or GT1160 claim."""
from common import ROOT, OUT, read, sha, write, now
import gzip, math
import numpy as np, lap
from scipy.optimize import linear_sum_assignment
assert (OUT / 'GO_classmask.json').exists(), 'Await root registration of this final diagnostic'
protocol = read(OUT / 'classmask_protocol.json')
assert sha(__file__) == protocol['worker_sha256']
for path, digest in protocol['inputs'].items():
    assert sha(ROOT / path) == digest, path
with gzip.open(ROOT / protocol['archive'], 'rt', encoding='utf8') as stream:
    event = next(event for event in __import__('json').load(stream)['events'] if event['frame'] == 1160)
with np.load(ROOT / protocol['detections'], allow_pickle=False) as stream:
    detector_classes = stream['cls'].copy()
row_ids = [row['tid'] for row in event['rows']]
columns = event['high_raw_ids']
original = np.array(event['corrected'], np.float32)
native = read(ROOT / protocol['native_protocol'])
threshold = native['native_config']['match_thresh']
assert threshold == .8

def solve(cost):
    _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=threshold)
    pairs = [(i, int(j)) for i, j in enumerate(x) if j >= 0]
    n, k = cost.shape
    augmented = np.zeros((n + k, n + k))
    augmented[:n, :k] = cost
    augmented[:n, k:] = .4
    augmented[n:, :k] = .4
    ii, jj = linear_sum_assignment(augmented)
    reference = math.fsum(float(augmented[i, j]) for i, j in zip(ii, jj))
    actual = math.fsum(float(cost[i, j]) for i, j in pairs) + .4 * (n + k - 2 * len(pairs))
    assert abs(reference - actual) < 1e-9
    return pairs, abs(reference - actual)

baseline, error = solve(original.copy())
assert sorted(baseline) == sorted(map(tuple, event['actual_first']))
incompatible = np.array([[int(row['cls']) != int(detector_classes[raw_id]) for raw_id in columns] for row in event['rows']])
masked = original.copy()
masked[incompatible] = 1e6
assert np.array_equal(masked[~incompatible], original[~incompatible])
retained = [i for i, tid in enumerate(row_ids) if tid != 2445]
records = []
for name, matrix, mapping in [('classmask_only', masked, list(range(len(row_ids)))), ('classmask_and_remove2445', masked[retained].copy(), retained)]:
    local, difference = solve(matrix)
    pairs = [(mapping[i], j) for i, j in local]
    before = {(row_ids[i], columns[j]) for i, j in baseline}
    after = {(row_ids[i], columns[j]) for i, j in pairs}
    records.append(dict(variant=name, matrix_shape=list(matrix.shape), objective_error=difference,
                        pairs=[dict(tid=row_ids[i], raw_id=columns[j], original_cost=float(original[i, j]), masked_cost=float(matrix[mapping.index(i), j])) for i, j in pairs],
                        removed_pairs=[dict(tid=tid, raw_id=raw) for tid, raw in sorted(before - after)], added_pairs=[dict(tid=tid, raw_id=raw) for tid, raw in sorted(after - before)],
                        raw109699_owner=[row_ids[i] for i, j in pairs if columns[j] == 109699],
                        target2179_assignment=[dict(raw_id=columns[j], cost=float(original[i, j])) for i, j in pairs if row_ids[i] == 2179],
                        matched_cross_class_pairs=[dict(tid=row_ids[i], raw_id=columns[j]) for i, j in pairs if incompatible[i, j]]))
for path, digest in protocol['inputs'].items():
    assert sha(ROOT / path) == digest, path
write('classmask_result.json', dict(created=now(), records=records, baseline_objective_error=error, mask_edges=int(incompatible.sum()),
                                    mask_edges_previously_within_limit=int((incompatible & (original <= threshold)).sum()), surviving_compatible_costs_unchanged=True,
                                    original_baseline_reproduced=True, no_GT1160_claim=True, policy_or_tracker_calls=0, model_calls=0,
                                    scope='Only the2registered saved-matrix controls. Removingactive2445 is not a deployedrepair. Cross-class detector metadata is not GT error. No weights/threshold changes or further variants.'))
print('FINAL_CLASSMASK_DIAGNOSTIC_COMPLETE')
