"""Independent scalar reconstruction; does not import the candidate policy."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def iou(a, b):
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0.

def cosine(a, b):
    return min(1., max(-1., math.fsum(float(x) * float(y) for x, y in zip(a, b)) / math.sqrt(math.fsum(float(x) ** 2 for x in a) * math.fsum(float(x) ** 2 for x in b))))

def nms(candidates):
    todo = sorted(range(len(candidates)), key=lambda i: (-candidates[i]['confidence'], candidates[i]['raw_id']))
    groups = []
    while todo:
        chosen = todo[0]
        aliases = [chosen]
        remaining = []
        for j in todo[1:]:
            if candidates[j]['cls'] == candidates[chosen]['cls'] and iou(candidates[j]['box'], candidates[chosen]['box']) > 0.7:
                aliases.append(j)
            else:
                remaining.append(j)
        groups.append(dict(representative_index=chosen, representative_raw_id=candidates[chosen]['raw_id'], alias_indices=aliases, alias_raw_ids=[candidates[j]['raw_id'] for j in aliases]))
        todo = remaining
    return groups

protocol = read(OUT / 'proposal_protocol.json')
binding = read(OUT / 'descriptor_binding.json')
result = read(OUT / 'result_episode.json')
assert sha(OUT / 'evaluation_arrays_episode.npz') == result['arrays_sha256']
policy_source = (OUT / 'appearance_recovery_policy_v4.py').read_text(encoding='utf8')
assert not any(forbidden in policy_source for forbidden in ["['identity']", "['hits']", '00115019', '35280', '574'])
snapshots = read(ROOT / 'research_identity_repairs_20261112/review/assignment_snapshots.json')['snapshots']
lowprotocol = read(ROOT / 'research_remaining_causes_20261112/independent_review/reid/low_protocol_v3.json')
max_cosine_error = 0.
max_iou_error = 0.
checked = 0
protected = 0
with np.load(OUT / 'evaluation_arrays_episode.npz') as saved:
    for record in result['records']:
        case_index = record['case']
        snapshot = snapshots[case_index]
        case = next(c for c in lowprotocol['cases'] if c['old_id'] == record['old_case'])
        eligible_tids = set()
        for receipt in snapshot['proposals']:
            if receipt.get('geometry') is not None:
                assert receipt.get('reason') in ['head_accept', 'head_refuse']
                eligible_tids.add(int(receipt['old_id']))
        assert record['eligible_tids'] == sorted(eligible_tids)
        rows = record['rows']
        candidates = record['candidates']
        assert all(set(row) == {'tid', 'state', 'cls', 'predicted_xyxy'} for row in rows)
        assert all(set(candidate) == {'raw_id', 'box', 'confidence', 'cls'} for candidate in candidates)
        expected_indices = [i for i, candidate in enumerate(case['candidates']) if record['mode'] != 'remove_all_target_own' or not any(hit['identity'] == case['identity'] for hit in candidate['hits'])]
        assert record['source_indices'] == expected_indices
        assert [candidate['raw_id'] for candidate in candidates] == [case['candidates'][i]['raw_id'] for i in expected_indices]
        groups = nms(candidates)
        assert groups == record['result']['groups']
        baseline_pairs = sorted((int(tid), int(raw)) for tid, raw in snapshot['assigned'].items())
        assert record['result']['baseline_pairs'] == [list(p) for p in baseline_pairs]
        occupied_rows = {tid for tid, _ in baseline_pairs}
        occupied_raw = {raw for _, raw in baseline_pairs}
        refs = saved[f'{case_index}_references']
        cur = saved[f'{case_index}_current'][expected_indices]
        if record['mode'] == 'descriptor_permutation':
            cur = np.roll(cur, 1, axis=0)
        trivial = record['mode'] in ['disabled', 'missing_competitors', 'no_descriptors']
        if trivial:
            assert record['result']['cosine'] is None and record['result']['group_iou'] is None
            assert all(d['action'] == 'defer' and d['raw_id'] is None for d in record['result']['decisions'])
            assert not record['result']['proposals']
            checked += len(rows)
            continue
        similarities = [[cosine(ref, cur[g['representative_index']]) for g in groups] for ref in refs]
        proximity = [[iou(row['predicted_xyxy'], candidates[g['representative_index']]['box']) for g in groups] for row in rows]
        if groups:
            max_cosine_error = max(max_cosine_error, float(abs(np.array(similarities) - record['result']['cosine']).max()))
            max_iou_error = max(max_iou_error, float(abs(np.array(proximity) - record['result']['group_iou']).max()))
        expected_proposals = []
        for i, row in enumerate(rows):
            actual = record['result']['decisions'][i]
            assert actual['tid'] == row['tid']
            action = 'defer'
            selected = None
            if row['tid'] in occupied_rows:
                reason = 'native_assignment_frozen'
                protected += 1
            elif row['state'] != 2:
                reason = 'row_not_lost'
            elif row['tid'] not in eligible_tids:
                reason = 'outside_observable_native_recovery_episode'
            elif not groups:
                action, reason = 'refuse', 'no_raw_candidate'
            else:
                potential = [j for j, group in enumerate(groups) if candidates[group['representative_index']]['cls'] == row['cls'] and candidates[group['representative_index']]['confidence'] <= 0.1 and (1 - similarities[i][j]) / 2 <= 1 - 0.8 and proximity[i][j] >= 0.5]
                if not potential:
                    action, reason = 'refuse', 'no_excluded_low_candidate_passes_fixed_appearance_and_proximity'
                else:
                    possible = [j for j, group in enumerate(groups) if candidates[group['representative_index']]['cls'] == row['cls']]
                    best = max(similarities[i][j] for j in possible)
                    maxima = [j for j in possible if similarities[i][j] == best]
                    if len(maxima) != 1:
                        reason = 'tied_best_candidate_group'
                    else:
                        j = maxima[0]
                        group = groups[j]
                        candidate = candidates[group['representative_index']]
                        reverse_rows = [k for k, r in enumerate(rows) if r['cls'] == candidate['cls']]
                        reverse_best = max(similarities[k][j] for k in reverse_rows)
                        reverse = [k for k in reverse_rows if similarities[k][j] == reverse_best]
                        if any(raw in occupied_raw for raw in group['alias_raw_ids']):
                            reason = 'candidate_alias_owned_by_frozen_native_assignment'
                        elif candidate['confidence'] > 0.1:
                            reason = 'best_group_is_native_candidate_not_recovery_branch'
                        elif j not in potential:
                            reason = 'best_group_fails_fixed_absolute_gates'
                        elif len(reverse) != 1 or reverse[0] != i:
                            reason = 'not_strict_mutual_nearest_against_all_competitor_rows'
                        else:
                            action, reason, selected = 'continue', 'unassigned_mutual_low_candidate_passes_fixed_gates', candidate['raw_id']
                            expected_proposals.append((row['tid'], selected))
            assert (action, reason, selected) == (actual['action'], actual['reason'], actual['raw_id']), (record['old_case'], record['mode'], row['tid'], action, actual)
            checked += 1
        assert expected_proposals == [(p['tid'], p['raw_id']) for p in record['result']['proposals']]
        assert all(p['tid'] not in occupied_rows and not set(p['alias_raw_ids']) & occupied_raw for p in record['result']['proposals'])
assert max_cosine_error < 1e-9 and max_iou_error < 1e-9
for container in [protocol['inputs'], binding['inputs']]:
    for name, digest in container.items():
        assert sha(ROOT / name) == digest, name
episode_addendum = read(OUT / 'episode_addendum_v4.json')
for name, digest in episode_addendum['new_sources'].items():
    assert sha(OUT / name) == digest, name
for name, digest in episode_addendum['additional_inputs'].items():
    assert sha(ROOT / name) == digest, name
scope_addendum = read(OUT / 'scope_addendum_v3.json')
for name, digest in scope_addendum['new_sources'].items():
    assert sha(OUT / name) == digest, name
for name, digest in protocol['sources'].items():
    assert sha(OUT / name) == digest, name
with (OUT / 'audit_episode.json').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(json.dumps(dict(passed=True, checked_row_decisions=checked, protected_native_row_decisions=protected, max_scalar_cosine_error=max_cosine_error, max_scalar_iou_error=max_iou_error,
                                 independent='Scalar cosine/IoU/NMS and complete policy reconstruction without importing production candidate; GT used only for remove-own control generation/evaluation.', shared='Archived descriptors, raw boxes, saved full-row snapshot and source label annotations.'), indent=2).rstrip() + '\n')
print('RECOVERY_AUDIT_PASS', checked, flush=True)
