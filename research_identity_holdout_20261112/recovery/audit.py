"""Independent scalar policy, label and control audit for all holdout events."""
from common import *
import numpy as np
from independent import iou, scalar_cosines, reconstruct

protocol, binding = verify()
runtime = read(OUT / 'runtime_cases.json')['cases']
histories = {(record['event_key'], record['tid']): record['observations'] for record in read(OUT / 'historical_observations.json')['records']}
evaluation_plan = read(OUT / 'evaluation_plan.json')
result = read(OUT / 'result.json')
assert result['protocol_sha256'] == sha(OUT / 'protocol.json')
assert result['binding_sha256'] == sha(OUT / 'descriptor_binding.json')
bank = np.load(ROOT / binding['descriptor_file'], allow_pickle=False)
labels = {}
annotation_cache = {}
maximum_cosine_error = 0.
maximum_iou_error = 0.
checked_decisions = 0
checked_scalar_pairs = 0
checked_labels = 0
protected_checks = 0

def objects(video, frame):
    if video not in labels:
        metadata = evaluation_plan['videos'][video]
        clock_path = ROOT / metadata['annotation_clock']
        label_path = ROOT / metadata['annotation_boxes']
        if not clock_path.exists() or not label_path.exists():
            labels[video] = None
        else:
            clock = read(clock_path)['anchors']
            ground = read(label_path)
            labels[video] = clock, {int(row.get('frameIndex', index)): row for index, row in enumerate(ground)}
    key = video, frame
    if key not in annotation_cache:
        if labels[video] is None:
            annotation_cache[key] = False, 'missing_exact_annotation_source', []
        else:
            clock, ground = labels[video]
            if str(frame) not in clock:
                annotation_cache[key] = False, 'missing_exact_raw_frame_clock_anchor', []
            else:
                anchor = clock[str(frame)]
                row = ground[int(anchor['frameIndex'])]
                assert row['name'] == anchor['name']
                annotation_cache[key] = True, 'exact_clock_name_frameIndex_verified', [dict(identity=str(obj['id']), box=[float(obj['box2d'][key]) for key in ['x1', 'y1', 'x2', 'y2']]) for obj in row.get('labels') or [] if obj['category'] == 'car' and not (obj.get('attributes') or {}).get('crowd', False)]
    return annotation_cache[key]

def hits(video, frame, box):
    exact, reason, annotations = objects(video, frame)
    return dict(exact=exact, reason=reason, hits=[dict(identity=obj['identity'], box=obj['box'], iou=iou(box, obj['box'])) for obj in annotations if iou(box, obj['box']) >= .5])

def outcome(reference, candidate):
    if not reference['exact'] or not candidate['exact'] or len(reference['hits']) != 1 or len(candidate['hits']) != 1:
        return 'unknown'
    return 'own' if reference['hits'][0]['identity'] == candidate['hits'][0]['identity'] else 'foreign'

def history_expected(video, observations, descriptor_frame):
    unique = {(int(observation['frame']), tuple(observation['box'])): observation for observation in observations}
    anchors = []
    for _, observation in sorted(unique.items()):
        value = hits(video, observation['frame'], observation['box'])
        if value['exact']:
            anchors.append(dict(frame=int(observation['frame']), box=observation['box'], evaluation=value))
    identities = [anchor['evaluation']['hits'][0]['identity'] for anchor in anchors if len(anchor['evaluation']['hits']) == 1]
    problems = []
    if not anchors:
        problems.append('no_exact_past_historical_anchor')
    if any(len(anchor['evaluation']['hits']) != 1 for anchor in anchors):
        problems.append('annotated_history_unmatched_or_ambiguous')
    if len(set(identities)) > 1:
        problems.append('conflicting_annotated_history_identities')
    valid = bool(anchors) and not problems
    return dict(valid=valid, problems=problems, expected_identity=identities[0] if valid else None,
                actual_source_anchor_frame=max(anchor['frame'] for anchor in anchors) if valid else None,
                all_annotated_history=anchors, descriptor_last_source_frame=descriptor_frame, source_pixel_identity_assumed=False)

def historical_outcome(expected, candidate):
    if not expected['valid'] or not candidate['exact'] or len(candidate['hits']) != 1:
        return 'unknown'
    return 'own' if expected['expected_identity'] == candidate['hits'][0]['identity'] else 'foreign'

def compare_labels(expected, actual):
    global checked_labels, maximum_iou_error
    assert expected['exact'] == actual['exact'] and expected['reason'] == actual['reason']
    assert [h['identity'] for h in expected['hits']] == [h['identity'] for h in actual['hits']]
    for left, right in zip(expected['hits'], actual['hits']):
        assert left['box'] == right['box']
        maximum_iou_error = max(maximum_iou_error, abs(left['iou'] - right['iou']))
    checked_labels += 1

assert len(runtime) == len(result['events']) == result['count']
for event_index, (event, receipt) in enumerate(zip(runtime, result['events'])):
    assert receipt['event_key'] == event['event_key']
    assert sha(OUT / receipt['path']) == receipt['sha256']
    saved = read(OUT / receipt['path'])
    assert saved['event_key'] == event['event_key'] and saved['protocol_sha256'] == result['protocol_sha256'] and saved['binding_sha256'] == result['binding_sha256']
    status('scalar_audit', completed=event_index, total=len(runtime), event_key=event['event_key'])
    rows = [{key: row[key] for key in ['tid', 'state', 'cls', 'predicted_xyxy']} for row in event['rows']]
    candidates = [{key: c[key] for key in ['raw_id', 'cls', 'confidence', 'box']} for c in event['candidates']]
    assert all(set(row) == {'tid', 'state', 'cls', 'predicted_xyxy'} for row in rows)
    assert all(set(c) == {'raw_id', 'cls', 'confidence', 'box'} for c in candidates)
    eligible = set()
    for native in event['native_proposals']:
        assert not set(native) - {'old_id', 'geometry', 'reason'}
        if native.get('geometry') is not None:
            assert native.get('reason') in ['head_accept', 'head_refuse']
            eligible.add(int(native['old_id']))
    assert saved['eligible_tids'] == sorted(eligible)
    assert saved['rows'] == len(rows) and saved['raw_candidates'] == len(candidates)
    complete = all(row['reference'] is not None for row in event['rows'])
    assert saved['complete_competitor_references'] == complete
    reference_evaluation = {}
    for row in event['rows']:
        reference = row['reference']
        value = hits(event['video'], reference['frame'], reference['box']) if reference is not None else dict(exact=False, reason='missing_observed_reference', hits=[])
        reference_evaluation[row['tid']] = value
        compare_labels(value, saved['reference_evaluation'][str(row['tid'])])
    current_evaluation = {}
    for candidate in candidates:
        value = hits(event['video'], event['frame'], candidate['box'])
        current_evaluation[candidate['raw_id']] = value
        compare_labels(value, saved['current_evaluation'][str(candidate['raw_id'])])
    historical = {}
    for row in event['rows']:
        expected = history_expected(event['video'], histories[(event['event_key'], row['tid'])], row['reference']['frame'] if row['reference'] is not None else None)
        historical[row['tid']] = expected
        actual = saved['historical_expected'][str(row['tid'])]
        for key in ['valid', 'problems', 'expected_identity', 'actual_source_anchor_frame', 'descriptor_last_source_frame', 'source_pixel_identity_assumed']:
            assert expected[key] == actual[key], (event['event_key'], row['tid'], key)
        assert len(expected['all_annotated_history']) == len(actual['all_annotated_history'])
        for left, right in zip(expected['all_annotated_history'], actual['all_annotated_history']):
            assert left['frame'] == right['frame'] and left['box'] == right['box']
            compare_labels(left['evaluation'], right['evaluation'])
    reference_vectors = [bank[row['reference']['slot'][0]][row['reference']['slot'][1]] for row in event['rows']] if complete else None
    current_vectors = [bank[c['slot'][0]][c['slot'][1]] for c in event['candidates']]
    full_cosine = scalar_cosines(reference_vectors, current_vectors) if complete else None
    if complete:
        expected = np.asarray(full_cosine).reshape(len(rows), len(candidates))
        actual = np.asarray(saved['raw_cosine']).reshape(len(rows), len(candidates))
        maximum_cosine_error = max(maximum_cosine_error, float(abs(expected - actual).max(initial=0)))
        checked_scalar_pairs += len(rows) * len(candidates)
    else:
        assert saved['raw_cosine'] is None
    frozen = sorted([list(map(int, pair)) for pair in event['baseline_pairs']])
    for baseline, pair in zip(saved['native_evaluation'], frozen):
        tid, raw = pair
        assert (baseline['tid'], baseline['raw_id']) == (tid, raw)
        if tid in reference_evaluation and raw in current_evaluation:
            assert baseline['outcome'] == outcome(reference_evaluation[tid], current_evaluation[raw])
        else:
            assert baseline['outcome'] == 'unknown'
    assert len(saved['native_evaluation']) == len(frozen)
    assert len(saved['native_historical_evaluation']) == len(frozen)
    for baseline, (tid, raw) in zip(saved['native_historical_evaluation'], frozen):
        assert (baseline['tid'], baseline['raw_id']) == (tid, raw)
        expected = historical_outcome(historical[tid], current_evaluation[raw]) if tid in historical and raw in current_evaluation else 'unknown'
        assert baseline['outcome'] == expected and baseline['source_pixel_identity_assumed'] is False
        assert baseline['actual_source_anchor_frame'] == (historical[tid]['actual_source_anchor_frame'] if tid in historical else None)
    control_rows = {row['tid'] for row in rows if row['state'] == 2 and row['tid'] in eligible}
    assert {mode['control_tid'] for mode in saved['modes'] if mode['mode'] == 'remove_all_own'} == control_rows
    assert {mode['control_tid'] for mode in saved['modes'] if mode['mode'] == 'remove_all_historical_own'} == control_rows
    assert [mode['mode'] for mode in saved['modes'][:5]] == ['baseline', 'fixed_repair', 'descriptor_permutation', 'missing_competitors', 'no_descriptors']
    for mode in saved['modes']:
        indices = list(range(len(candidates)))
        if mode['mode'] in ['remove_all_own', 'remove_all_historical_own']:
            reference = reference_evaluation[mode['control_tid']]
            exact_current = objects(event['video'], event['frame'])[0]
            expected_history = historical[mode['control_tid']]
            valid = reference['exact'] and len(reference['hits']) == 1 and exact_current if mode['mode'] == 'remove_all_own' else expected_history['valid'] and exact_current
            assert mode['valid'] == valid
            if not valid:
                assert mode['policy'] is None and mode['evaluation'] == [] and mode['historical_evaluation'] == []
                continue
            identity = reference['hits'][0]['identity'] if mode['mode'] == 'remove_all_own' else expected_history['expected_identity']
            assert mode['removed_identity_evaluation_only'] == identity
            if mode['mode'] == 'remove_all_historical_own':
                assert mode['actual_source_anchor_frame'] == expected_history['actual_source_anchor_frame']
            indices = [j for j, candidate in enumerate(candidates) if not any(h['identity'] == identity for h in current_evaluation[candidate['raw_id']]['hits'])]
        assert indices == mode['indices']
        subset = [candidates[j] for j in indices]
        descriptor_indices = indices[-1:] + indices[:-1] if mode['mode'] == 'descriptor_permutation' else indices
        values = [[row[j] for j in descriptor_indices] for row in full_cosine] if complete else None
        expected = reconstruct(rows, subset, values, frozen, eligible, complete=complete and mode['mode'] != 'missing_competitors', enabled=mode['mode'] != 'baseline', no_descriptors=mode['mode'] == 'no_descriptors')
        grouped, decisions, proposals, similarity, proximity = expected
        actual = mode['policy']
        assert actual['baseline_pairs'] == frozen and actual['groups'] == grouped
        assert [(d['tid'], d['action'], d['reason'], d['raw_id']) for d in actual['decisions']] == decisions, (event['event_key'], mode['mode'])
        assert [(p['tid'], p['raw_id'], p['alias_raw_ids']) for p in actual['proposals']] == proposals
        occupied_rows = {tid for tid, _ in frozen}
        occupied_raw = {raw for _, raw in frozen}
        assert all(p['tid'] not in occupied_rows and not set(p['alias_raw_ids']) & occupied_raw for p in actual['proposals'])
        assert len({p['tid'] for p in actual['proposals']}) == len(actual['proposals'])
        assert sum(len(p['alias_raw_ids']) for p in actual['proposals']) == len({raw for p in actual['proposals'] for raw in p['alias_raw_ids']})
        protected_checks += sum(row['tid'] in occupied_rows for row in rows)
        checked_decisions += len(rows)
        if similarity is None:
            assert actual['cosine'] is None and actual['group_iou'] is None
        else:
            maximum_cosine_error = max(maximum_cosine_error, float(abs(np.array(similarity).reshape(len(rows), len(grouped)) - np.array(actual['cosine']).reshape(len(rows), len(grouped))).max(initial=0)))
            maximum_iou_error = max(maximum_iou_error, float(abs(np.array(proximity).reshape(len(rows), len(grouped)) - np.array(actual['group_iou']).reshape(len(rows), len(grouped))).max(initial=0)))
        assert len(mode['evaluation']) == len(actual['proposals'])
        assert len(mode['historical_evaluation']) == len(actual['proposals'])
        for evaluation, proposal in zip(mode['evaluation'], actual['proposals']):
            tid, raw = proposal['tid'], proposal['raw_id']
            assert (evaluation['tid'], evaluation['raw_id']) == (tid, raw)
            compare_labels(reference_evaluation[tid], evaluation['reference'])
            compare_labels(current_evaluation[raw], evaluation['candidate'])
            assert evaluation['outcome'] == outcome(reference_evaluation[tid], current_evaluation[raw])
        for evaluation, proposal in zip(mode['historical_evaluation'], actual['proposals']):
            tid, raw = proposal['tid'], proposal['raw_id']
            assert (evaluation['tid'], evaluation['raw_id']) == (tid, raw)
            assert evaluation['outcome'] == historical_outcome(historical[tid], current_evaluation[raw])
            assert evaluation['actual_source_anchor_frame'] == historical[tid]['actual_source_anchor_frame']
            assert evaluation['descriptor_last_source_frame'] == historical[tid]['descriptor_last_source_frame']
            assert evaluation['source_pixel_identity_assumed'] is False
    assert {d['tid'] for d in saved['decomposition']} == control_rows
    for decomposition in saved['decomposition']:
        tid = decomposition['tid']
        index = next(i for i, row in enumerate(rows) if row['tid'] == tid)
        primary = next(mode['policy'] for mode in saved['modes'] if mode['mode'] == 'fixed_repair')
        assert decomposition['decision'] == next(d for d in primary['decisions'] if d['tid'] == tid)
        reference = reference_evaluation[tid]
        own = reference['hits'][0]['identity'] if reference['exact'] and len(reference['hits']) == 1 else None
        own_indices = [j for j, candidate in enumerate(candidates) if own is not None and any(hit['identity'] == own for hit in current_evaluation[candidate['raw_id']]['hits'])]
        assert [observation['raw_id'] for observation in decomposition['all_raw_own_observations']] == [candidates[j]['raw_id'] for j in own_indices]
        for observation, j in zip(decomposition['all_raw_own_observations'], own_indices):
            group = next(g for g in primary['groups'] if candidates[j]['raw_id'] in g['alias_raw_ids'])
            assert observation['representative_raw_id'] == group['representative_raw_id'] and observation['aliases'] == group['alias_raw_ids']
            assert observation['is_excluded_low'] == (candidates[j]['confidence'] <= .1)
            if complete:
                maximum_cosine_error = max(maximum_cosine_error, abs(observation['raw_cosine'] - full_cosine[index][j]))
        for trace, group_index in zip(decomposition['group_gate_trace'], range(len(primary['groups']))):
            group = primary['groups'][group_index]
            candidate = candidates[group['representative_index']]
            similarity = primary['cosine'][index][group_index]
            proximity = iou(rows[index]['predicted_xyxy'], candidate['box'])
            assert trace['raw_id'] == candidate['raw_id'] and trace['aliases'] == group['alias_raw_ids']
            assert trace['same_class'] == (candidate['cls'] == rows[index]['cls']) and trace['excluded_low'] == (candidate['confidence'] <= .1)
            assert trace['appearance_pass'] == ((1 - similarity) / 2 <= 1 - .8) and trace['proximity_pass'] == (proximity >= .5)
            assert trace['occupied'] == bool(set(group['alias_raw_ids']) & {raw for _, raw in frozen})
    print('HOLDOUT_RECOVERY_AUDIT', event_index + 1, len(runtime), flush=True)
assert maximum_cosine_error < 1e-9 and maximum_iou_error < 1e-9
verify()
write('audit.json', dict(passed=True, events=len(runtime), checked_row_decisions=checked_decisions, scalar_pairs=checked_scalar_pairs, checked_annotation_records=checked_labels,
                         native_ownership_checks=protected_checks, maximum_scalar_cosine_error=maximum_cosine_error, maximum_scalar_iou_error=maximum_iou_error,
                         shared='Frozen descriptor bank, selected event metadata, underlying GT files and clock sources.', independent='No production policy/worker/annotation helper import. Scalar fsum cosine, scalar IoU/NMS, full policy branches, controls and exact-clock labels reconstructed independently.'))
status('audit_complete', completed=len(runtime), total=len(runtime))
