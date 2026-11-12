"""Normalize EVERY selected event; never select using labels or availability.

Only metadata conversion. No descriptor import, dot products, policy outcomes,
GT reads or tracker calls. Parent freezes selection and bank before evaluation.
"""
from common import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snapshots', required=True)
parser.add_argument('--bank-plan', required=True)
parser.add_argument('--exclusions', required=True)
args = parser.parse_args()
paths = [Path(value).resolve() for value in [args.snapshots, args.bank_plan, args.exclusions]]
for path in paths:
    path.relative_to(ROOT)
source = read(paths[0])
snapshots = source.get('snapshots', source.get('events'))
assert isinstance(snapshots, list) and snapshots
plan = read(paths[1])
frames = plan['frames']
exclusions = read(paths[2])
assert isinstance(exclusions.get('excluded_videos'), list), 'Canonical exclusions ledger must expose excluded_videos; do not guess.'
excluded = set(exclusions['excluded_videos'])
runtime = []
evaluation = {}
bank_usages = []
historical = []

def slot(video, frame, box, given=None):
    key = video + '|' + str(frame)
    boxes = frames[key]['boxes']
    index = given[1] if given is not None else boxes.index(list(map(float, box)))
    assert given is None or given[0] == key
    assert list(map(float, boxes[index])) == list(map(float, box))
    bank_usages.append([key, int(index)])
    return [key, int(index)]

for snapshot in snapshots:
    video = snapshot['video']
    frame = int(snapshot['frame'])
    assert video not in excluded, (video, 'Excluded old/development video')
    event_key = video + '|' + str(frame)
    rows = []
    for row in snapshot['rows']:
        history = [dict(frame=int(item['frame']), box=list(map(float, item['box'])), **{key: item[key] for key in ['public', 'raw_id'] if key in item}) for item in row.get('history', [])]
        assert all(item['frame'] < frame for item in history)
        historical.append(dict(event_key=event_key, video=video, current_frame=frame, tid=int(row['tid']), observations=history))
        predicted = row.get('predicted_xyxy')
        if predicted is None:
            mean = row['mean']
            predicted = [mean[0] - mean[2] / 2, mean[1] - mean[3] / 2, mean[0] + mean[2] / 2, mean[1] + mean[3] / 2]
        reference = row.get('reference')
        if reference is None and row.get('history'):
            reference = row['history'][-1]
        if reference is not None:
            assert int(reference['frame']) < frame
            reference = dict(frame=int(reference['frame']), box=list(map(float, reference['box'])),
                             slot=slot(video, int(reference['frame']), reference['box'], reference.get('slot')))
        rows.append(dict(tid=int(row['tid']), state=int(row['state']), cls=int(row['cls']), predicted_xyxy=list(map(float, predicted)), reference=reference))
    raw = snapshot.get('raw_current_car_candidates', snapshot.get('raw_candidates'))
    assert isinstance(raw, list)
    candidates = []
    for candidate in raw:
        if int(candidate['cls']) != 2:
            continue
        candidates.append(dict(raw_id=int(candidate['raw_id']), cls=2, confidence=float(candidate['confidence']), box=list(map(float, candidate['box'])),
                               slot=slot(video, frame, candidate['box'], candidate.get('slot', candidate.get('descriptor_slot')))))
    native = snapshot.get('proposals', snapshot.get('native_proposals', []))
    receipts = [{key: proposal[key] for key in ['old_id', 'geometry', 'reason'] if key in proposal} for proposal in native]
    assigned = snapshot.get('assigned')
    pairs = sorted((int(tid), int(raw_id)) for tid, raw_id in assigned.items()) if assigned is not None else sorted([list(map(int, pair)) for pair in snapshot['baseline_pairs']])
    runtime.append(dict(event_key=event_key, video=video, frame=frame, rows=rows, candidates=candidates, native_proposals=receipts, baseline_pairs=[list(pair) for pair in pairs]))
    metadata = snapshot.get('evaluation', {})
    inherited = snapshot.get('evaluation_sources', {})
    evaluation[video] = dict(annotation_clock=metadata.get('annotation_clock', f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json'),
                             annotation_boxes=metadata.get('annotation_boxes', f'data/bdd100k/labels/box_track_20/val/{video}.json'))
    evaluation[video]['annotation_clock'] = inherited.get('clock', evaluation[video]['annotation_clock'])
    evaluation[video]['annotation_boxes'] = inherited.get('labels', evaluation[video]['annotation_boxes'])
assert len({event['event_key'] for event in runtime}) == len(runtime)
for event in runtime:
    assert len({r['tid'] for r in event['rows']}) == len(event['rows'])
    assert len({c['raw_id'] for c in event['candidates']}) == len(event['candidates'])
write('runtime_cases.json', dict(created=now(), source_snapshots=str(paths[0].relative_to(ROOT)).replace('\\', '/'), source_sha256=sha(paths[0]),
                                 exclusions_sha256=sha(paths[2]), cases=runtime, scope='All selection-ledger events; policy runtime fields contain no annotation identities/hits or previous diagnostic case IDs.'))
write('evaluation_plan.json', dict(created=now(), videos=evaluation, inputs={path: sha(ROOT / path) for video in evaluation.values() for path in video.values() if (ROOT / path).exists()},
                                   unavailable_sources=[path for video in evaluation.values() for path in video.values() if not (ROOT / path).exists()],
                                   convention='Only exact persisted raw-frame clock anchors, matching GT name/frameIndex; IoU>=0.5. Missing/ambiguous/unmatched labels remain unknown. No nearest-clock guessing.'))
write('historical_observations.json', dict(created=now(), records=historical, scope='Separate evaluation-only observed history; never label the unannotated last descriptor source by carrying an earlier identity. No GT objects read during this conversion.'))
inherited_sources = source.get('source_sha256', {})
for name, digest in inherited_sources.items():
    assert sha(ROOT / name) == digest, name
write('runtime_preparation.json', dict(created=now(), inputs={**{str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in paths}, **inherited_sources},
                                       events=len(runtime), rows=sum(len(e['rows']) for e in runtime), current_cars=sum(len(e['candidates']) for e in runtime), usages=len(bank_usages), GT_read_for_selection=False,
                                       bank_schema='npz[video|raw_frame][box_slot] ->256-vector; exact boxes/slots from shared extraction plan.', metadata_only=True))
print('RUNTIME_METADATA_READY', len(runtime), flush=True)
