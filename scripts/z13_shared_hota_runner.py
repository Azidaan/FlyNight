"""Isolated fixed HOTA execution with shared immutable MASA frame features."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scope():
    source = (ROOT / 'scripts/z13_spatial_hota_runner.py').read_text(encoding='utf-8')
    source = source.replace('runs/z13_spatial_hota_v1_20261014', 'runs/z13_shared_masa_hota_v1_20261015/full')
    source = source.replace("'scripts/z13_spatial_hota_worker.py', *switches", "'scripts/z13_shared_hota_worker.py', *switches")
    old_jobs = "jobs += [{'id': f'{a}_masa', 'stage': 'masa', 'arm': a} for a in protocol['arms']]"
    assert source.count(old_jobs) == 1
    source = source.replace(old_jobs, "jobs += [{'id': 'shared_masa', 'stage': 'shared_masa'}]")
    source = source.replace("'masa_jobs': 3", "'masa_jobs': 1")
    result = {'__file__': str(ROOT / 'scripts/z13_spatial_hota_runner.py'), '__name__': 'isolated_shared_runner'}
    exec(compile(source, 'isolated_shared_runner', 'exec'), result)
    result['FILES'] += ['tracking/z13_compiled_backend.py', 'scripts/z13_shared_masa.py',
        'scripts/z13_shared_hota_worker.py', 'scripts/z13_shared_hota_runner.py',
        'runs/z13_shared_masa_hota_v1_20261015/full/environment.json',
        'runs/z13_triton_hota_v1_20261014/packages/triton_windows-3.6.0.post26.dist-info/RECORD']
    return result


def main():
    from tracking.z13_compiled_backend import setup
    setup()
    scope()['main']()


if __name__ == '__main__':
    main()
