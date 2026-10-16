"""Prepared alternate execution; no implicit authority to start a second run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scope():
    source = (ROOT / 'scripts/z13_spatial_hota_runner.py').read_text(encoding='utf-8')
    source = source.replace('runs/z13_spatial_hota_v1_20261014', 'runs/z13_cached_hota_v1_20261015/full')
    source = source.replace("'scripts/z13_spatial_hota_worker.py', *switches", "'scripts/z13_cached_hota_worker.py', *switches")
    result = {'__file__': str(ROOT / 'scripts/z13_spatial_hota_runner.py'), '__name__': 'isolated_cached_runner'}
    exec(compile(source, 'isolated_cached_runner', 'exec'), result)
    result['FILES'] += ['tracking/z13_compiled_backend.py', 'tracking/z13_cached_params_backend.py',
        'tracking/z13_compiled_cached_backend.py', 'scripts/z13_cached_hota_worker.py',
        'scripts/z13_cached_hota_runner.py', 'runs/z13_cached_hota_v1_20261015/full/environment.json',
        'runs/z13_triton_hota_v1_20261014/packages/triton_windows-3.6.0.post26.dist-info/RECORD']
    return result


def main():
    from tracking.z13_compiled_backend import setup
    setup()
    scope()['main']()


if __name__ == '__main__':
    main()
