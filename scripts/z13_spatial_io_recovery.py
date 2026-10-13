"""One audited recovery of the 2026-10-13 Windows status-write interruption."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import msvcrt
    import numpy as np
    import torch
    from scripts.z13_spatial_train import RUN, identity
    from tracking.z13_checkpoint import atomic_json, replace_with_retry

    lock = (RUN / 'controller.lock').open('r+b')
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    folder = RUN / 'B/fold0'
    destination = RUN / 'recovery_20261013_io'
    if destination.exists():
        raise RuntimeError('Recovery already recorded; refusing to overwrite backups')
    old_identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    protocol = json.loads((RUN / 'protocol.json').read_text(encoding='utf-8'))
    new_identity = identity(protocol, protocol['jobs'][0])
    assert old_identity.keys() == new_identity.keys()
    for key in old_identity:
        if key != 'files_sha256':
            assert old_identity[key] == new_identity[key], key
    old_hashes, new_hashes = old_identity['files_sha256'], new_identity['files_sha256']
    assert old_hashes.keys() == new_hashes.keys()
    changes = {k: {'old': old_hashes[k], 'new': new_hashes[k]}
               for k in old_hashes if old_hashes[k] != new_hashes[k]}
    assert {k.replace('\\', '/') for k in changes} == {
        'tracking/z13_checkpoint.py', 'scripts/z13_spatial_train.py',
        'scripts/z13_spatial_runner.py'}
    state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert state['identity'] == old_identity
    assert state['step'] == 3610 and len(state['history']) == 3610

    def same(a, b):
        assert type(a) is type(b), (type(a), type(b))
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                same(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for left, right in zip(a, b):
                same(left, right)
        else:
            assert a == b

    destination.mkdir()
    for name in ('checkpoint.pt', 'identity.json', 'status.json'):
        shutil.copy2(folder / name, destination / name)
    shutil.copy2(RUN / 'status.json', destination / 'controller_status.json')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(folder / 'checkpoint.pt') == digest(destination / 'checkpoint.pt')
    updated = {**state, 'identity': new_identity}
    temporary = folder / 'checkpoint.pt.recovery'
    with temporary.open('wb') as stream:
        torch.save(updated, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checked = torch.load(temporary, map_location='cpu', weights_only=False)
    for key in state:
        if key != 'identity':
            same(state[key], checked[key])
    assert checked['identity'] == new_identity
    report = {
        'created': datetime.now().astimezone().isoformat(), 'step': state['step'],
        'reason': 'WinError 5 while replacing status.json; infrastructure-only retry fix',
        'changed_files': changes, 'data_and_protocol_unchanged': True,
        'all_training_state_exactly_preserved': True,
        'backup_checkpoint_sha256': digest(destination / 'checkpoint.pt'),
        'updated_checkpoint_sha256': digest(temporary),
        'cpu_tests': 'test_z13_atomic_retry.py + test_z13_spatial.py: 6 passed',
        'technical_gates_retained': 'Training equations unchanged; Windows replacement tested with real reader lock',
        'failure_log': 'runs/z13_spatial_v2_20261013/controller_20261013_204411/log.txt',
    }
    atomic_json(destination / 'report.json', {**report, 'state': 'validated_before_replacement'})
    replace_with_retry(temporary, folder / 'checkpoint.pt')
    atomic_json(folder / 'identity.json', new_identity)
    atomic_json(destination / 'report.json', {**report, 'state': 'recovered'})
    print(json.dumps({'state': 'recovered', 'step': state['step'], 'state_exact': True}))
    lock.close()


if __name__ == '__main__':
    main()
