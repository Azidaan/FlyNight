"""Synchronize newly completed unchanged B outputs before a later cache switch."""
from pathlib import Path
from datetime import datetime
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.z13_cached_hota_prepare import SOURCE, RUN, read, digest, copy_checked
from tracking.z13_checkpoint import atomic_json


def main():
    assert read(RUN / 'status.json')['state'] == 'prepared'
    assert read(RUN / 'full_authorization.json')['approved'] is False
    identity = read(RUN / 'inference_identity.json')
    assert all(digest(ROOT / p) == value for p, value in identity.items()), 'Prepared code/data changed'
    source_identity = read(SOURCE / 'inference_identity.json')
    proof = read(RUN / 'reuse_proof.json')
    added = []
    for source_marker in sorted((SOURCE / 'completed_jobs').glob('B_sig_*.json')):
        marker = read(source_marker)
        assert marker['identity'] == source_identity and marker['job']['arm'] == 'B'
        target_marker = RUN / 'completed_jobs' / source_marker.name
        if target_marker.exists():
            assert read(target_marker)['identity'] == identity
            continue
        video = marker['job']['video']
        hashes = {}
        for name in ('signatures.npz', 'signatures.json'):
            source = SOURCE / 'B/signatures' / video / name
            target = RUN / 'B/signatures' / video / name
            copy_checked(source, target)
            hashes[name] = digest(target)
        atomic_json(target_marker, {'identity': identity, 'job': marker['job'],
            'completed': marker['completed'], 'reused_from': str(source_marker.relative_to(ROOT)),
            'reuse_proof': str((RUN / 'reuse_proof.json').relative_to(ROOT))})
        added.append({'video': video, 'source_marker': str(source_marker.relative_to(ROOT)), 'copied_sha256': hashes})
    proof['reused'] += added
    proof['last_sync'] = datetime.now().astimezone().isoformat()
    atomic_json(RUN / 'reuse_proof.json', proof)
    atomic_json(RUN / 'status.json', {'state': 'prepared', 'completed_jobs': len(proof['reused']),
                                    'total_jobs': 109, 'reason': 'Ready, not launched; B provenance verified'})
    print(json.dumps({'added_B': len(added), 'reused_B': len(proof['reused']), 'GPU_run_started': False}))


if __name__ == '__main__':
    main()
