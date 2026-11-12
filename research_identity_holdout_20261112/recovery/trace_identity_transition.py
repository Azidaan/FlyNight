"""Read the saved A native trace only; no tracker/model/policy replay."""
from common import ROOT, OUT, read, sha, write, now
import gzip
import numpy as np
from annotations import ExactAnnotations

video = 'b1ca2e5d-84cf9134'
archive_path = f'runs/domain_matched_native_20261109/rollouts/C34/{video}.events.json.gz'
native_path = 'runs/domain_matched_native_20261109/protocol.json'
native = read(ROOT / native_path)
cache_path = native['videos'][video]['detections']
with gzip.open(ROOT / archive_path, 'rt', encoding='utf8') as stream:
    archive = __import__('json').load(stream)['events']
by_frame = {event['frame']: event for event in archive}
with np.load(ROOT / cache_path, allow_pickle=False) as source:
    raw = {key: source[key].copy() for key in ['xyxy', 'conf', 'cls', 'frame_idx']}
annotation = ExactAnnotations(read(OUT / 'evaluation_plan.json'))
identities = [2179, 2445]
firsts = {}
for tid in identities:
    row_presence = [(event['frame'], row) for event in archive for row in event['rows'] if row['tid'] == tid]
    assigned = [(event['frame'], int(event['assigned'][str(tid)])) for event in archive if str(tid) in event['assigned']]
    emitted = [event['frame'] for event in archive if tid in event['emitted_ids']]
    firsts[str(tid)] = dict(first_saved_row_frame=min((frame for frame, row in row_presence), default=None),
                            stored_start_frames=sorted({row['start_frame'] for frame, row in row_presence}),
                            first_saved_assignment=assigned[0] if assigned else None, last_saved_assignment=assigned[-1] if assigned else None,
                            first_saved_public_emission=min(emitted) if emitted else None)

def detection(raw_id):
    if raw_id is None:
        return None
    confidence = float(raw['conf'][raw_id])
    return dict(raw_id=int(raw_id), raw_frame=int(raw['frame_idx'][raw_id]), cls=int(raw['cls'][raw_id]), confidence=confidence,
                box=raw['xyxy'][raw_id].astype(float).tolist(), native_first=confidence >= .25, native_second=.1 < confidence < .25,
                excluded_below_native_low=confidence <= .1)

def frame_record(frame):
    if frame not in by_frame:
        return dict(frame=frame, event_saved=False)
    event = by_frame[frame]
    ground = annotation.at(video, frame)
    cars = np.flatnonzero((raw['frame_idx'] == frame) & (raw['cls'] == 2))
    own = None
    if ground['exact']:
        own = []
        for raw_id in cars:
            candidate = detection(int(raw_id))
            hit = annotation.hits(video, frame, candidate['box'])
            if any(h['identity'] == '00122657' for h in hit['hits']):
                own.append(dict(**candidate, current_exact_hits=hit['hits']))
    tracks = []
    for tid in identities:
        match = next(((index, row) for index, row in enumerate(event['rows']) if row['tid'] == tid), None)
        assigned_raw = event['assigned'].get(str(tid))
        selected = []
        costs = []
        source_row = None
        if match is not None:
            index, row = match
            source_row = {key: row[key] for key in ['state', 'start_frame', 'mean', 'history']}
            source_row['row'] = index
            source_row['history_last'] = row['history'][-1] if row['history'] else None
            for i, j in event['actual_first']:
                if i == index:
                    selected.append(dict(column=j, raw_id=event['high_raw_ids'][j], base_cost=event['C'][i][j], corrected_cost=event['corrected'][i][j]))
            interesting = {int(value) for value in event['assigned'].values() if int(value) in [int(event['assigned'][str(other)]) for other in identities if str(other) in event['assigned']]}
            if own is not None:
                interesting.update(item['raw_id'] for item in own)
            for column, raw_id in enumerate(event['high_raw_ids']):
                if raw_id in interesting:
                    costs.append(dict(column=column, raw_id=raw_id, base_cost=event['C'][index][column], corrected_cost=event['corrected'][index][column],
                                      assigned_to=[int(owner) for owner, value in event['assigned'].items() if int(value) == raw_id]))
        proposal_records = [{key: p[key] for key in ['old_id', 'raw_id', 'row', 'column', 'reference_frame', 'reason', 'accepted', 'original_cost', 'geometry', 'logit'] if key in p} for p in event['proposals'] if p.get('old_id') == tid]
        assigned_detection = detection(int(assigned_raw)) if assigned_raw is not None else None
        if assigned_detection is not None:
            assert assigned_detection['raw_frame'] == frame
            assigned_detection['exact_current_evaluation'] = annotation.hits(video, frame, assigned_detection['box'])
        tracks.append(dict(tid=tid, row_before_assignment=source_row, current_assigned=assigned_detection, public_emitted=tid in event['emitted_ids'],
                           selected_actual_first=selected, relevant_native_costs=costs, native_proposals=proposal_records))
    return dict(frame=frame, event_saved=True, current_GT_exact=ground['exact'], GT_missing_reason=None if ground['exact'] else ground['reason'],
                exact_own_car_annotation=[obj for obj in ground['objects'] if obj['identity'] == '00122657'] if ground['exact'] else None,
                raw_car_count=len(cars), exact_own_raw_candidates=own, tracks=tracks,
                duplicate_records=[d for d in event.get('duplicates', []) if d.get('tracked_id') in identities or d.get('lost_id') in identities])

frames = [frame_record(frame) for frame in range(1150, 1166)]
birth_frames = {record['first_saved_assignment'][0] for record in firsts.values() if record['first_saved_assignment'] is not None}
birth_proofs = [frame_record(frame) for frame in sorted(birth_frames) if frame < 1150]
inputs = {archive_path: sha(ROOT / archive_path), native_path: sha(ROOT / native_path), cache_path: sha(ROOT / cache_path),
          f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json': sha(ROOT / f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json'),
          f'data/bdd100k/labels/box_track_20/val/{video}.json': sha(ROOT / f'data/bdd100k/labels/box_track_20/val/{video}.json')}
write('identity_transition_2179_2445.json', dict(created=now(), video=video, bounds=[1150, 1165], firsts=firsts, frames=frames, earlier_birth_proofs=birth_proofs, inputs=inputs,
                                               source_script_sha256=sha(__file__), model_calls=0, tracker_calls=0, policy_calls=0,
                                               evaluation_rule='Exact current GT only; between anchors own identity is never propagated. Whole-archive first-presence metadata only locates originalbirth; detailed diagnostic window1150to1165.',
                                               limits='Saved assignments/state/costs can localize the recorded transition; they do not by themselves prove the physical/neural root cause or causal utility of a new repair.'))
print(__import__('json').dumps(firsts, ensure_ascii=False))
