"""Reserved B annotation-readiness metadata only; no descriptors or policies.

All frozen B events retained. Does not load raw detection caches, image pixels,
MASA banks, Census features or any new repair-score/output file.
"""
from common import ROOT, OUT, read, sha, write, now
import gzip
from collections import defaultdict, Counter

REVIEW = ROOT / 'research_identity_holdout_20261112/review'
selection_path = REVIEW / 'selected_events_B_RESERVED_v2.json'
proposal_path = REVIEW / 'proposed_selection_protocol_v2.json'
inventory_path = REVIEW / 'inventory_v2.json'
selection = read(selection_path)
proposal = read(proposal_path)
inventory = read(inventory_path)
excluded = {video for video, roles in inventory['video_exposure'].items() if any(role != 'historical_native_validation_aggregate_and_baseline' for role, source in roles)}
allowed = set(proposal['rule']['B_reserved_videos'])
assert not allowed & excluded
assert {event['video'] for event in selection['events']} <= allowed
sources = {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in [selection_path, proposal_path, inventory_path]}
label_cache = {}
event_metadata = []
all_rows = []
targets = []

def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0., a[2] - a[0]) * max(0., a[3] - a[1]) + max(0., b[2] - b[0]) * max(0., b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.

def annotations(video, frame):
    if video not in label_cache:
        clock_path = f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json'
        labels_path = f'data/bdd100k/labels/box_track_20/val/{video}.json'
        alignment_path = f'runs/z13_val20_matched_20261024/cache/{video}/alignment.json'
        clock = read(ROOT / clock_path)['anchors']
        raw = read(ROOT / labels_path)
        aligned = read(ROOT / alignment_path)['matches']
        assert {int(key): (anchor['name'], anchor['frameIndex']) for key, anchor in clock.items()} == {item['video_frame_index']: (item['reference_name'], item['reference_index']) for item in aligned}
        by_index = {int(row.get('frameIndex', index)): row for index, row in enumerate(raw)}
        label_cache[video] = clock, by_index
        for path in [clock_path, labels_path, alignment_path]:
            sources[path] = sha(ROOT / path)
    clock, by_index = label_cache[video]
    if str(frame) not in clock:
        return None
    anchor = clock[str(frame)]
    row = by_index[anchor['frameIndex']]
    assert row['name'] == anchor['name']
    return [dict(identity=str(obj['id']), box=[obj['box2d'][key] for key in ['x1', 'y1', 'x2', 'y2']]) for obj in row.get('labels') or [] if obj['category'] == 'car' and not (obj.get('attributes') or {}).get('crowd', False)]

def hits(box, objects):
    return [dict(identity=obj['identity'], iou=iou(box, obj['box'])) for obj in objects if iou(box, obj['box']) >= .5]

grouped = defaultdict(list)
for selected in selection['events']:
    assert selected['video'] not in excluded
    grouped[selected['event_archive']].append(selected)
for archive_path, chosen_events in grouped.items():
    sources[archive_path] = sha(ROOT / archive_path)
    with gzip.open(ROOT / archive_path, 'rt', encoding='utf8') as stream:
        archive = {event['frame']: event for event in __import__('json').load(stream)['events']}
    for chosen in chosen_events:
        event = archive[chosen['frame']]
        video, frame = chosen['video'], chosen['frame']
        current = annotations(video, frame)
        current_exact = current is not None
        row_values = {}
        for row in event['rows']:
            history = row.get('history', [])
            last = history[-1] if history else None
            last_annotations = annotations(video, last['frame']) if last is not None else None
            last_hits = hits(last['box'], last_annotations) if last_annotations is not None else []
            unique_observations = {(int(observation['frame']), tuple(observation['box'])): observation for observation in history}
            bindings = []
            for _, observation in sorted(unique_observations.items()):
                past = annotations(video, observation['frame'])
                if past is not None:
                    bindings.append(dict(frame=observation['frame'], box=observation['box'], hits=hits(observation['box'], past)))
            identities = [binding['hits'][0]['identity'] for binding in bindings if len(binding['hits']) == 1]
            if not bindings:
                reason = 'no_exact_annotated_observation_in_saved_recent_history'
            elif any(len(binding['hits']) != 1 for binding in bindings):
                reason = 'annotated_history_observation_unmatched_or_ambiguous'
            elif len(set(identities)) != 1:
                reason = 'inconsistent_annotated_history_identities'
            else:
                reason = 'consistent_exact_anchored_history'
            expected = identities[0] if reason == 'consistent_exact_anchored_history' else None
            value = dict(event_key=chosen['event_key'], video=video, frame=frame, tid=row['tid'], state=row['state'], cls=row['cls'],
                         current_exact_anchor=current_exact, last_reference_frame=None if last is None else last['frame'],
                         last_reference_exact_anchor=last_annotations is not None, last_reference_unique_identity=last_hits[0]['identity'] if len(last_hits) == 1 else None,
                         strict_last_reference_evaluable=current_exact and len(last_hits) == 1,
                         annotated_history_observations=bindings, history_identity_status=reason, expected_anchored_identity=expected,
                         actual_source_anchor_frame=max(binding['frame'] for binding in bindings) if expected is not None else None,
                         anchored_history_evaluable=current_exact and expected is not None,
                         expected_identity_present_in_current_annotation=bool(current_exact and expected is not None and any(obj['identity'] == expected for obj in current)),
                         last_unannotated_source_labelled_by_history=False)
            row_values[row['tid']] = value
            all_rows.append(value)
        for key in chosen['proposal_keys']:
            assert key['old_id'] in row_values
            targets.append(dict(proposal_key=chosen['event_key'] + '|' + str(key['old_id']) + '|' + str(key['raw_id']), **row_values[key['old_id']]))
        event_metadata.append(dict(event_key=chosen['event_key'], current_exact_anchor=current_exact, rows=len(event['rows']), eligible_proposals=len(chosen['proposal_keys'])))
    del archive
summary = dict(events=len(selection['events']), proposals=len(targets), videos=len(allowed), all_rows=len(all_rows),
               current_exact=sum(row['current_exact_anchor'] for row in targets), last_reference_exact=sum(row['last_reference_exact_anchor'] for row in targets),
               strict_unique_last_reference_evaluable=sum(row['strict_last_reference_evaluable'] for row in targets),
               any_exact_history=sum(bool(row['annotated_history_observations']) for row in targets),
               current_exact_and_any_history=sum(row['current_exact_anchor'] and bool(row['annotated_history_observations']) for row in targets),
               anchored_history_evaluable=sum(row['anchored_history_evaluable'] for row in targets),
               anchored_identity_present_current=sum(row['expected_identity_present_in_current_annotation'] for row in targets),
               target_history_status=dict(Counter(row['history_identity_status'] for row in targets)))
write('B_metadata_readiness.json', dict(created=now(), summary=summary, targets=targets, rows=all_rows, events=event_metadata, inputs=sources,
                                        selection_sha256=sha(selection_path), all_frozen_B_events_retained=True, repair_outcomes_computed=False,
                                        raw_detection_cache_read=False, descriptors_read=False, policy_imported=False, image_or_neural_calls=0,
                                        scope='Metadata readiness only for reserved B. Exact historical-to-current identity continuity is separate from literal last descriptor-source identity. No nearest/interpolated GT, no candidate coverage or repair-score selection. B remains unexecuted for MASA/Census/recovery.'))
print(__import__('json').dumps(summary, ensure_ascii=False))
