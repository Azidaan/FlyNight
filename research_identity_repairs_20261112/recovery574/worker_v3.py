import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import json, hashlib, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from appearance_recovery_policy_v3 import confirm_recovery

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    with (OUT / name).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).rstrip() + '\n')

def verify():
    protocol = read(OUT / 'proposal_protocol.json')
    for name, digest in protocol['inputs'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in protocol['sources'].items():
        assert sha(OUT / name) == digest, name
    binding = read(OUT / 'descriptor_binding.json')
    assert binding['descriptor_file'] == protocol['pending_descriptor_bank']
    assert binding['descriptor_file'] in binding['inputs']
    for name, digest in binding['inputs'].items():
        assert sha(ROOT / name) == digest, name
    return protocol, binding

assert (OUT / 'GO_lost_only.json').exists(), 'Await parent preregistration and descriptor binding'
protocol, binding = verify()
scope_addendum = read(OUT / 'scope_addendum_v3.json')
for name, digest in scope_addendum['new_sources'].items():
    assert sha(OUT / name) == digest, name
addendum = read(OUT / 'worker_addendum_v2.json')
for name, digest in addendum['new_sources'].items():
    assert sha(OUT / name) == digest, name
snapshots = read(ROOT / 'research_identity_repairs_20261112/review/assignment_snapshots.json')['snapshots']
lowdir = ROOT / 'research_remaining_causes_20261112/independent_review/reid'
lowprotocol = read(lowdir / 'low_protocol_v3.json')
lowresult = read(lowdir / 'low_result.json')
oldresult = read(lowdir / 'result.json')
bank = np.load(ROOT / binding['descriptor_file'], allow_pickle=False)
current = np.load(lowdir / 'low_descriptors.npz', allow_pickle=False)
oldbank = np.load(lowdir / 'descriptors.npz', allow_pickle=False)
records = []
baseline = []
arrays = {}
label_cache = {}

def source_hits(video, frame, box):
    key = (video, frame)
    if key not in label_cache:
        clocks = read(ROOT / f'runs/identity_assignment_20261108/full_precision/runtime_inputs/{video}.json')['anchors']
        if str(frame) not in clocks:
            # No exact evaluation annotation: retain unknown, never guess a clock.
            return []
        anchor = clocks[str(frame)]
        labels = read(ROOT / f'data/bdd100k/labels/box_track_20/val/{video}.json')
        row = next(row for i, row in enumerate(labels) if int(row.get('frameIndex', i)) == anchor['frameIndex'])
        assert row['name'] == anchor['name']
        label_cache[key] = row.get('labels') or []
    from appearance_recovery_policy_v3 import overlap
    hits = []
    for obj in label_cache[key]:
        if obj['category'] != 'car' or obj.get('attributes', {}).get('crowd', False):
            continue
        bb = [obj['box2d'][key] for key in ['x1', 'y1', 'x2', 'y2']]
        value = overlap(box, bb)
        if value >= 0.5:
            hits.append(dict(identity=str(obj['id']), iou=value))
    return hits

for case_index, snapshot in enumerate(snapshots):
    case = next(c for c in lowprotocol['cases'] if c['old_id'] == snapshot['old_case'])
    assert snapshot['video'] == case['video'] and snapshot['frame'] == case['current_frame']
    rows = [{key: r[key] for key in ['tid', 'state', 'cls', 'predicted_xyxy']} for r in snapshot['rows']]
    candidates = [{key: c[key] for key in ['raw_id', 'box', 'confidence', 'cls']} for c in snapshot['raw_current_car_candidates']]
    assert [c['raw_id'] for c in candidates] == [c['raw_id'] for c in case['candidates']]
    references = np.stack([bank[r['reference']['slot'][0]][r['reference']['slot'][1]] for r in snapshot['rows']]).copy()
    vectors = current[str(case['old_id'])].copy()
    assert len(vectors) == len(candidates)
    assert all(r['reference']['frame'] < snapshot['frame'] for r in snapshot['rows'])
    frozen = sorted((int(tid), int(raw)) for tid, raw in snapshot['assigned'].items())
    target_i = snapshot['target_row']
    slot = oldresult['slots'][str(case['old_id'])]['reference']
    old_reference = oldbank[slot[0]][slot[1]].astype(float)
    reference_error = float(abs(references[target_i].astype(float) - old_reference).max())
    assert reference_error < 1e-6, (case['old_id'], reference_error)
    normalized = vectors.astype(float) / np.linalg.norm(vectors.astype(float), axis=1)[:, None]
    witness = normalized @ (references[target_i].astype(float) / np.linalg.norm(references[target_i].astype(float)))
    expected = np.array([c['cosine'] for c in next(c for c in lowresult['records'] if c['old_id'] == case['old_id'])['candidates']])
    score_error = float(abs(witness - expected).max())
    assert score_error < 1e-6, (case['old_id'], score_error)
    baseline.append(dict(case=case_index, old_case=case['old_id'], reference_descriptor_error=reference_error, cosine_witness_error=score_error, all_raw=len(candidates), complete_reference_rows=len(rows)))
    arrays[f'{case_index}_references'] = references
    arrays[f'{case_index}_current'] = vectors
    own_identity = case['identity']
    removed = [i for i, c in enumerate(case['candidates']) if any(hit['identity'] == own_identity for hit in c['hits'])]
    modes = ['disabled', 'complete_recovery', 'descriptor_permutation', 'missing_competitors', 'no_descriptors', 'remove_all_target_own']
    for mode in modes:
        indices = [i for i in range(len(candidates)) if mode != 'remove_all_target_own' or i not in removed]
        candidate_input = [candidates[i] for i in indices]
        vector_input = vectors[indices].copy()
        if mode == 'descriptor_permutation':
            vector_input = np.roll(vector_input, 1, axis=0)
        result = confirm_recovery(rows, candidate_input, None if mode == 'no_descriptors' else references.copy(), None if mode == 'no_descriptors' else vector_input,
                                  frozen, competition_complete=mode != 'missing_competitors', enabled=mode != 'disabled')
        assert result['baseline_pairs'] == [list(p) for p in frozen]
        assert all(p['tid'] not in snapshot['assigned'] and str(p['tid']) not in snapshot['assigned'] for p in result['proposals'])
        by_raw = {c['raw_id']: c for c in case['candidates']}
        evaluation = []
        for proposal in result['proposals']:
            source = next(r for r in snapshot['rows'] if r['tid'] == proposal['tid'])
            reference_hits = source_hits(snapshot['video'], source['reference']['frame'], source['reference']['box'])
            candidate_hits = by_raw[proposal['raw_id']]['hits']
            ref_ids = [h['identity'] for h in reference_hits]
            current_ids = [h['identity'] for h in candidate_hits]
            outcome = 'own' if len(ref_ids) == len(current_ids) == 1 and ref_ids == current_ids else 'foreign' if len(ref_ids) == len(current_ids) == 1 else 'unknown'
            evaluation.append(dict(tid=proposal['tid'], raw_id=proposal['raw_id'], reference_hits=reference_hits, candidate_hits=candidate_hits, outcome=outcome))
        target_decision = next(d for d in result['decisions'] if d['tid'] == snapshot['target_tid'])
        target_hits = [] if target_decision['raw_id'] is None else by_raw[target_decision['raw_id']]['hits']
        target_own = target_decision['raw_id'] is not None and [h['identity'] for h in target_hits] == [own_identity]
        records.append(dict(case=case_index, old_case=case['old_id'], video=snapshot['video'], frame=snapshot['frame'], mode=mode,
                            rows=rows, candidates=candidate_input, source_indices=indices, removed_raw_ids=[candidates[i]['raw_id'] for i in removed] if mode == 'remove_all_target_own' else [],
                            result=result, evaluation=evaluation, target_tid=snapshot['target_tid'], target_decision=target_decision, target_evaluation_own=target_own, target_candidate_hits=target_hits))
    assert np.array_equal(vectors, current[str(case['old_id'])])
    print('RECOVERY_CASE', case_index, len(rows), len(candidates), flush=True)
verify()
with (OUT / 'evaluation_arrays_lost_only.npz').open('xb') as stream:
    np.savez_compressed(stream, **arrays)
flat_proposals = [dict(video=r['video'], frame=r['frame'], **p) for r in records if r['mode'] == 'complete_recovery' for p in r['result']['proposals']]
write('result_lost_only.json', dict(passed=True, records=records, proposals=flat_proposals, baseline=baseline, arrays_sha256=sha(OUT / 'evaluation_arrays_lost_only.npz'), new_fits=0, core_calls=0, tracker_calls=0, neural_calls=0,
                         reference_scope='Complete107 saved first-stage rows; freeze all public assigned IDs including subsequent native stages. Engineering evaluation on5 selected scenes.'))
