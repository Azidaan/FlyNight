"""Admit a compiled full run only after the saved parity and speed probe."""
from datetime import datetime
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/z13_triton_hota_v1_20261014'
OLD = ROOT / 'runs/z13_spatial_hota_v1_20261014'


def main():
    report = json.loads((PARENT / 'report.json').read_text(encoding='utf-8'))
    assert report['passed'] is True
    assert all(report['video_comparisons'][a]['wall_speedup'] > 1 for a in ('B', 'D'))
    auth = json.loads((OLD / 'full_authorization.json').read_text(encoding='utf-8'))
    deadline = datetime.fromisoformat(auth['deadline'])
    assert datetime.now().astimezone() < deadline
    full = PARENT / 'full'; full.mkdir()
    for name in ('protocol.json', 'videos.json'):
        shutil.copy2(OLD / name, full / name)
    for arm in ('B', 'C', 'D'):
        (full / arm).mkdir()
        for name in ('model.pt', 'policy.json'):
            shutil.copy2(OLD / arm / name, full / arm / name)
    (full / 'technical').mkdir()
    shutil.copy2(PARENT / 'report.json', full / 'technical/report.json')
    environment = {'python': __import__('sys').version, 'torch': report['torch'],
                   'triton': report['triton'], 'wheel': 'triton-windows-3.6.0.post26',
                   'packages': str(PARENT / 'packages'),
                   'compiler': str(PARENT / 'packages/triton/runtime/tcc/tcc.exe'),
                   'source_probe': str(PARENT / 'report.json')}
    (full / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n', encoding='utf-8')
    # Reuse only the separately compiled probe outputs, never a mixed backend cache.
    video = report['paired_video']
    for arm in ('B', 'D'):
        source = PARENT / f'technical/compiled_{arm}' / video
        destination = full / arm / 'signatures' / video
        destination.mkdir(parents=True)
        for name in ('signatures.npz', 'signatures.json'):
            shutil.copy2(source / name, destination / name)
    auth.update({'backend': 'Separate torch.compile/Inductor/Triton execution',
                 'backend_requested': json.loads((PARENT / 'protocol.json').read_text(encoding='utf-8'))['request'],
                 'maximum_seconds': max(1, int((deadline - datetime.now().astimezone()).total_seconds()))})
    (full / 'full_authorization.json').write_text(json.dumps(auth, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (full / 'status.json').write_text(json.dumps({'state': 'authorized', 'completed_jobs': 0, 'total_jobs': 109}) + '\n', encoding='utf-8')
    print(json.dumps(auth, ensure_ascii=False))


if __name__ == '__main__':
    main()
