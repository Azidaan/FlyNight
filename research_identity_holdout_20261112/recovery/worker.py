"""Frozen-policy multi-event holdout replay; CPU only, no tracker lifecycle."""
from common import *
import sys
import numpy as np
sys.path.insert(0, str(ROOT))
from tracking.appearance_recovery_policy import confirm_recovery, observable_episode_tids, overlap
from annotations import ExactAnnotations, classify, classify_historical

assert (OUT / 'GO.json').exists(), 'Await parent holdout registration and descriptor binding'
protocol, binding = verify()
protocol_hash = sha(OUT / 'protocol.json')
binding_hash = sha(OUT / 'descriptor_binding.json')
runtime = read(OUT / 'runtime_cases.json')['cases']
annotation = ExactAnnotations(read(OUT / 'evaluation_plan.json'))
histories = {(record['event_key'], record['tid']): record['observations'] for record in read(OUT / 'historical_observations.json')['records']}
bank = np.load(ROOT / binding['descriptor_file'], allow_pickle=False)
event_index = []

def vector(slot):
    value = bank[slot[0]][int(slot[1])].copy()
    assert value.shape == (256,) and np.isfinite(value).all() and np.linalg.norm(value) > 0
    return value

def evaluate(event, proposal, references, current):
    row = next(row for row in event['rows'] if row['tid'] == proposal['tid'])
    candidate = next(c for c in event['candidates'] if c['raw_id'] == proposal['raw_id'])
    source = references[row['tid']]
    destination = current[candidate['raw_id']]
    return dict(tid=row['tid'], raw_id=candidate['raw_id'], reference=source, candidate=destination, outcome=classify(source, destination))

for index, event in enumerate(runtime):
    output_name = f'events/{index:05d}.json'
    if (OUT / output_name).exists():
        prior = read(OUT / output_name)
        assert prior['event_key'] == event['event_key'] and prior['protocol_sha256'] == protocol_hash and prior['binding_sha256'] == binding_hash
        event_index.append(dict(event_key=event['event_key'], path=output_name, sha256=sha(OUT / output_name)))
        continue
    status('event', completed=index, total=len(runtime), event_key=event['event_key'])
    rows = [{key: row[key] for key in ['tid', 'state', 'cls', 'predicted_xyxy']} for row in event['rows']]
    candidates = [{key: c[key] for key in ['raw_id', 'cls', 'confidence', 'box']} for c in event['candidates']]
    eligible = observable_episode_tids(event['native_proposals'])
    assert eligible <= {row['tid'] for row in rows}
    complete = all(row['reference'] is not None for row in event['rows'])
    references = np.stack([vector(row['reference']['slot']) for row in event['rows']]) if complete else None
    current = np.stack([vector(c['slot']) for c in event['candidates']]) if candidates else np.empty((0, 256), np.float32)
    frozen = sorted([list(map(int, pair)) for pair in event['baseline_pairs']])
    reference_labels = {}
    for row in event['rows']:
        source = row['reference']
        reference_labels[row['tid']] = annotation.hits(event['video'], source['frame'], source['box']) if source is not None else dict(exact=False, reason='missing_observed_reference', hits=[])
    current_labels = {c['raw_id']: annotation.hits(event['video'], event['frame'], c['box']) for c in candidates}
    historical_expected = {row['tid']: annotation.historical_expected(event['video'], histories[(event['event_key'], row['tid'])], row['reference']['frame'] if row['reference'] is not None else None) for row in event['rows']}
    native_evaluation = []
    native_historical_evaluation = []
    for tid, raw_id in frozen:
        if tid in reference_labels and raw_id in current_labels:
            native_evaluation.append(dict(tid=tid, raw_id=raw_id, reference=reference_labels[tid], candidate=current_labels[raw_id], outcome=classify(reference_labels[tid], current_labels[raw_id])))
        else:
            native_evaluation.append(dict(tid=tid, raw_id=raw_id, outcome='unknown', reason='baseline_owner_or_current_candidate_outside_saved_car_row_pool'))
        expected = historical_expected.get(tid)
        destination = current_labels.get(raw_id)
        native_historical_evaluation.append(dict(tid=tid, raw_id=raw_id, outcome=classify_historical(expected, destination) if expected is not None and destination is not None else 'unknown',
                                                actual_source_anchor_frame=expected['actual_source_anchor_frame'] if expected is not None else None, source_pixel_identity_assumed=False))
    specifications = [dict(mode=mode, indices=list(range(len(candidates))), valid=True) for mode in ['baseline', 'fixed_repair', 'descriptor_permutation', 'missing_competitors', 'no_descriptors']]
    for row in rows:
        if row['state'] != 2 or row['tid'] not in eligible:
            continue
        source = reference_labels[row['tid']]
        valid = source['exact'] and len(source['hits']) == 1 and annotation.at(event['video'], event['frame'])['exact']
        if valid:
            identity = source['hits'][0]['identity']
            indices = [j for j, candidate in enumerate(candidates) if not any(h['identity'] == identity for h in current_labels[candidate['raw_id']]['hits'])]
            specifications.append(dict(mode='remove_all_own', control_tid=row['tid'], valid=True, indices=indices, removed_identity_evaluation_only=identity))
        else:
            specifications.append(dict(mode='remove_all_own', control_tid=row['tid'], valid=False, reason='no_exact_unambiguous_reference_identity_or_current_clock'))
        expected = historical_expected[row['tid']]
        valid_historical = expected['valid'] and annotation.at(event['video'], event['frame'])['exact']
        if valid_historical:
            identity = expected['expected_identity']
            indices = [j for j, candidate in enumerate(candidates) if not any(h['identity'] == identity for h in current_labels[candidate['raw_id']]['hits'])]
            specifications.append(dict(mode='remove_all_historical_own', control_tid=row['tid'], valid=True, indices=indices, removed_identity_evaluation_only=identity,
                                       actual_source_anchor_frame=expected['actual_source_anchor_frame']))
        else:
            specifications.append(dict(mode='remove_all_historical_own', control_tid=row['tid'], valid=False, reason='no_consistent_exact_historical_identity_or_current_clock'))
    modes = []
    for specification in specifications:
        if not specification['valid']:
            modes.append(dict(**specification, policy=None, evaluation=[], historical_evaluation=[]))
            continue
        ids = specification['indices']
        candidate_input = [candidates[j] for j in ids]
        vectors = current[ids].copy()
        if specification['mode'] == 'descriptor_permutation':
            vectors = np.roll(vectors, 1, axis=0)
        output = confirm_recovery(rows, candidate_input, None if specification['mode'] == 'no_descriptors' else references,
                                  None if specification['mode'] == 'no_descriptors' else vectors, frozen,
                                  competition_complete=complete and specification['mode'] != 'missing_competitors', eligible_tids=eligible, enabled=specification['mode'] != 'baseline')
        assert output['baseline_pairs'] == frozen
        evaluation = [evaluate(event, proposal, reference_labels, current_labels) for proposal in output['proposals']]
        historical_evaluation = [dict(tid=proposal['tid'], raw_id=proposal['raw_id'], outcome=classify_historical(historical_expected[proposal['tid']], current_labels[proposal['raw_id']]),
                                      actual_source_anchor_frame=historical_expected[proposal['tid']]['actual_source_anchor_frame'], descriptor_last_source_frame=historical_expected[proposal['tid']]['descriptor_last_source_frame'],
                                      source_pixel_identity_assumed=False) for proposal in output['proposals']]
        modes.append(dict(**specification, policy=output, evaluation=evaluation, historical_evaluation=historical_evaluation))
    primary = next(record for record in modes if record['mode'] == 'fixed_repair')
    decomposition = []
    raw_cosine = None
    if complete:
        a = references.astype(float)
        b = current.astype(float)
        raw_cosine = np.clip((a / np.linalg.norm(a, axis=1)[:, None]) @ (b / np.linalg.norm(b, axis=1)[:, None]).T, -1., 1.)
    by_raw_group = {raw_id: group for group in primary['policy']['groups'] for raw_id in group['alias_raw_ids']}
    occupied = {raw_id for _, raw_id in frozen}
    for i, row in enumerate(rows):
        if row['tid'] not in eligible or row['state'] != 2:
            continue
        truth = reference_labels[row['tid']]
        own_identity = truth['hits'][0]['identity'] if truth['exact'] and len(truth['hits']) == 1 else None
        observations = []
        for j, candidate in enumerate(candidates):
            if own_identity is None or not any(hit['identity'] == own_identity for hit in current_labels[candidate['raw_id']]['hits']):
                continue
            group = by_raw_group[candidate['raw_id']]
            observations.append(dict(raw_id=candidate['raw_id'], confidence=candidate['confidence'], is_excluded_low=candidate['confidence'] <= 0.1,
                                     raw_cosine=None if raw_cosine is None else float(raw_cosine[i, j]), raw_proximity_iou=overlap(row['predicted_xyxy'], candidate['box']),
                                     representative_raw_id=group['representative_raw_id'], representative_is_this_raw=group['representative_raw_id'] == candidate['raw_id'],
                                     aliases=group['alias_raw_ids'], any_alias_occupied=bool(set(group['alias_raw_ids']) & occupied)))
        group_trace = []
        if primary['policy']['cosine'] is not None:
            for j, group in enumerate(primary['policy']['groups']):
                candidate = candidates[group['representative_index']]
                similarity = primary['policy']['cosine'][i][j]
                costs = (1 - similarity) / 2
                proximity = primary['policy']['group_iou'][i][j]
                same_rows = [k for k, competitor in enumerate(rows) if competitor['cls'] == candidate['cls']]
                reverse_best = max(primary['policy']['cosine'][k][j] for k in same_rows)
                reverse = [rows[k]['tid'] for k in same_rows if primary['policy']['cosine'][k][j] == reverse_best]
                group_trace.append(dict(raw_id=candidate['raw_id'], aliases=group['alias_raw_ids'], cosine=similarity, appearance_cost=costs, proximity_iou=proximity,
                                        same_class=candidate['cls'] == row['cls'], excluded_low=candidate['confidence'] <= 0.1, appearance_pass=costs <= 1 - 0.8, proximity_pass=proximity >= 0.5,
                                        occupied=bool(set(group['alias_raw_ids']) & occupied), reverse_nearest_tids=reverse, strict_reverse_to_this_row=reverse == [row['tid']], candidate_truth=current_labels[candidate['raw_id']]))
        decision = next(d for d in primary['policy']['decisions'] if d['tid'] == row['tid'])
        decomposition.append(dict(tid=row['tid'], reference_evaluation=truth, decision=decision, all_raw_own_observations=observations, group_gate_trace=group_trace,
                                  note='Accounting of fixed gates and observed annotation availability; not a new threshold search or proof of physical pixel ownership.'))
    write(output_name, dict(event_key=event['event_key'], video=event['video'], frame=event['frame'], protocol_sha256=protocol_hash, binding_sha256=binding_hash,
                            rows=len(rows), raw_candidates=len(candidates), eligible_tids=sorted(eligible), complete_competitor_references=complete,
                            native_evaluation=native_evaluation, native_historical_evaluation=native_historical_evaluation, historical_expected={str(k): v for k, v in historical_expected.items()},
                            reference_evaluation={str(k): v for k, v in reference_labels.items()}, current_evaluation={str(k): v for k, v in current_labels.items()},
                            modes=modes, decomposition=decomposition, raw_cosine=None if raw_cosine is None else raw_cosine.tolist()))
    event_index.append(dict(event_key=event['event_key'], path=output_name, sha256=sha(OUT / output_name)))
    print('HOLDOUT_RECOVERY_EVENT', index + 1, len(runtime), event['event_key'], flush=True)
verify()
write('result.json', dict(passed=True, protocol_sha256=protocol_hash, binding_sha256=binding_hash, events=event_index, count=len(event_index), fixed_policy_sha256=sha(ROOT / 'tracking/appearance_recovery_policy.py'),
                         new_fits=0, neural_calls=0, core_calls=0, tracker_calls=0, selection_mutated=False, scope='Selected unseen event snapshots only; optional branch does not replay public ID lifecycle or full tracking metrics.'))
status('complete', completed=len(runtime), total=len(runtime))
