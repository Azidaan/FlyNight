"""Compose all four registered A variants, only after an exact parent GO."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from research_identity_holdout_20261112.review.compose_adapter_A import compose_event
from tracking.identity_repair_adapter import VARIANTS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--go', required=True)
    args, private = parser.parse_known_args()
    assert all(x.startswith('--audit-memory-worker=') for x in private)
    protocol_path = OUT/'proposed_compose_protocol.json'
    protocol = read(protocol_path)
    go = read(args.go)
    assert go['approved'] is True and go['protocol_sha256'] == sha(protocol_path)
    assert datetime.now(ZoneInfo('Europe/Moscow')) < datetime.fromisoformat(protocol['deadline'])
    for name, digest in protocol['inputs'].items():
        assert sha(ROOT/name) == digest, name
    assert set(go.get('pending_parent_binding', {})) == set(protocol['pending_parent_binding'])
    for name, digest in go['pending_parent_binding'].items():
        assert sha(ROOT/name) == digest, name
    runtime = read(ROOT/protocol['paths']['runtime'])
    clean = {e['event_key']: e for e in read(ROOT/protocol['paths']['clean_runtime'])['cases']}
    recovery = read(ROOT/protocol['paths']['recovery'])
    correspondence = read(ROOT/protocol['paths']['correspondence'])
    assert recovery['passed'] and correspondence['passed']
    assert read(ROOT/protocol['paths']['recovery_audit'])['passed']
    assert read(ROOT/protocol['paths']['correspondence_audit'])['passed']
    assert len(runtime['events']) == protocol['events'] == 54
    all_records, notes = [], []
    with np.load(ROOT/protocol['paths']['bank'], allow_pickle=False) as bank:
        for event in runtime['events']:
            assert len(event['eligible_proposal_keys']) == 1
            key = event['eligible_proposal_keys'][0]
            item = clean[event['event_key']]
            rows = [{k: r[k] for k in ['tid', 'state', 'cls', 'predicted_xyxy']} for r in item['rows']]
            candidates = [{k: c[k] for k in ['raw_id', 'cls', 'confidence', 'box']} for c in item['candidates']]
            references = np.stack([bank[r['reference']['slot'][0]][r['reference']['slot'][1]] for r in item['rows']])
            current = np.stack([bank[c['slot'][0]][c['slot'][1]] for c in item['candidates']])
            snapshot = {k: event[k] for k in ['event_key', 'video', 'frame', 'high_raw_ids', 'actual_first', 'assigned', 'base_cost', 'corrected_cost']}
            snapshot.update(target_tid=key['old_id'], target_raw_id=key['raw_id'], target_row=key['row'], target_column=key['column'], native_proposals=item['native_proposals'])
            rr = [r for r in correspondence['records'] if r['event_key'] == event['event_key'] and r['old_id'] == key['old_id'] and r['raw_id'] == key['raw_id'] and r['head'] == 'native']
            def score(coordinate, reference_mode):
                found = [r for r in rr if r['coordinate'] == coordinate and r['reference_mode'] == reference_mode]
                assert len(found) <= 1
                return found[0]['score'] if found else None
            original, census = score('original', 'index'), score('census', 'terminal_hold')
            receipt = next(r for r in recovery['events'] if r['event_key'] == event['event_key'])
            recovery_path = ROOT/protocol['paths']['recovery_directory']/receipt['path']
            assert sha(recovery_path) == receipt['sha256']
            recovery_event = read(recovery_path)
            policy = next(m['policy'] for m in recovery_event['modes'] if m['mode'] == 'fixed_repair')
            allowed_low = {(p['tid'], p['raw_id']) for p in policy['proposals']}
            for variant in VARIANTS:
                record, note = compose_event(snapshot, rows, candidates, references, current, original, census, variant)
                actual_low = {(p['tid'], p['raw_id']) for p in note.get('low_proposals', [])}
                assert actual_low <= allowed_low, 'Combined stricter ownership cannot introduce a recovery absent from frozen component.'
                all_records.append(record)
                details = dict(original_score=original, census_score=census)
                details.update(note)
                notes.append(dict(event_key=event['event_key'], variant=variant, **details))
    assert len(all_records) == 216
    for name, digest in protocol['inputs'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in go['pending_parent_binding'].items():
        assert sha(ROOT/name) == digest, name
    decisions = OUT/'decisions_A.json'
    write(decisions, dict(records=all_records, protocol_sha256=sha(protocol_path), GT_used_for_decisions=False, tracker_calls=0))
    write(OUT/'compose_notes_A.json', dict(records=notes, protocol_sha256=sha(protocol_path), execution_failures=sum('execution_failure' in r for r in all_records)))
    assert go.get('authorize_registered_matrix_audit') is True
    inputs = dict(protocol['inputs'])
    inputs.update(go['pending_parent_binding'])
    inputs[str(decisions.relative_to(ROOT)).replace('\\', '/')] = sha(decisions)
    inputs[str(protocol_path.relative_to(ROOT)).replace('\\', '/')] = sha(protocol_path)
    write(OUT/'matrix_GO_A.json', dict(approved=True, inputs=inputs, delegated_by_parent_GO_sha256=sha(args.go), scope='Only already registered frozen A matrix audit and post-audit; no new variants or tracker execution.'))
    print('COMPOSE_A_READY', len(all_records), flush=True)


if __name__ == '__main__':
    main()
