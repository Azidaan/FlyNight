"""Saved first-pair detector-class metadata on ALL54 A; no GT/policy replay."""
from common import ROOT, OUT, read, sha, write, now
import numpy as np
from collections import Counter
manifest_path = 'research_identity_holdout_20261112/review/A_runtime_manifest_v2.json'
native_path = 'runs/domain_matched_native_20261109/protocol.json'
manifest = read(ROOT / manifest_path)
native = read(ROOT / native_path)
caches = {}
inputs = {manifest_path: sha(ROOT / manifest_path), native_path: sha(ROOT / native_path), 'datasets/bdd100k_loader.py': sha(ROOT / 'datasets/bdd100k_loader.py')}
events = []
mismatches = []
directions = Counter()
all_pairs = 0
for event in manifest['events']:
    video = event['video']
    if video not in caches:
        source_path = native['videos'][video]['detections']
        with np.load(ROOT / source_path, allow_pickle=False) as stream:
            caches[video] = {key: stream[key].copy() for key in ['frame_idx', 'xyxy', 'conf', 'cls']}
        inputs[source_path] = sha(ROOT / source_path)
    raw = caches[video]
    mismatch_count = 0
    records = []
    for row_index, column in event['actual_first']:
        row = event['rows'][row_index]
        raw_id = event['high_raw_ids'][column]
        assert int(raw['frame_idx'][raw_id]) == event['frame']
        source_class, target_class = int(row['cls']), int(raw['cls'][raw_id])
        record = dict(event_key=event['event_key'], video=video, frame=event['frame'], tid=row['tid'], row=row_index, column=column, row_state=row['state'],
                      row_class_before_update=source_class, current_raw_id=raw_id, current_detector_class=target_class, current_confidence=float(raw['conf'][raw_id]),
                      source_last_frame=row['reference']['frame'] if row['reference'] else None,
                      base_cost=event['base_cost'][row_index][column], corrected_cost=event['corrected_cost'][row_index][column], cross_class=source_class != target_class)
        records.append(record)
        all_pairs += 1
        directions[f'{source_class}->{target_class}'] += 1
        if record['cross_class']:
            mismatches.append(record)
            mismatch_count += 1
    events.append(dict(event_key=event['event_key'], first_pairs=len(records), cross_class_pairs=mismatch_count, saved_actual_first=records))
write('native_class_metadata_A.json', dict(created=now(), events=len(events), actual_first_pairs=all_pairs, cross_class_pairs=len(mismatches),
                                           events_with_cross_class=sum(event['cross_class_pairs'] > 0 for event in events), class_pair_counts=dict(directions),
                                           mismatches=mismatches, event_records=events, inputs=inputs, source_script_sha256=sha(__file__),
                                           GT_read=False, descriptors_read=False, new_solver_or_policy_calls=0, model_calls=0, full_tracker_calls=0,
                                           scope='Difference between saved track-class metadata BEFOREupdate and current detector class only. Does not prove these pairs physically wrong: detector classification can change. All54Aretained; Bnotread.'))
print('ACTUAL_FIRST_CLASS_METADATA', all_pairs, len(mismatches), sum(event['cross_class_pairs'] > 0 for event in events))
