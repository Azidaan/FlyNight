"""Own inference controller derived from the fixed runner, isolated outputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_compiled_backend import setup
setup()


def main():
    # Replace only the run directory, identity paths and child entrypoint.
    source = (ROOT / 'scripts/z13_spatial_hota_runner.py').read_text(encoding='utf-8')
    source = source.replace("runs/z13_spatial_hota_v1_20261014", "runs/z13_triton_hota_v1_20261014/full")
    source = source.replace("'scripts/z13_spatial_hota_worker.py', *switches", "'scripts/z13_triton_hota_worker.py', *switches")
    scope = {'__file__': str(ROOT / 'scripts/z13_spatial_hota_runner.py'), '__name__': 'isolated_runner'}
    exec(compile(source, 'isolated_z13_triton_runner', 'exec'), scope)
    scope['FILES'] += ['tracking/z13_compiled_backend.py', 'scripts/z13_triton_hota_worker.py',
                       'scripts/z13_triton_hota_runner.py',
                       'runs/z13_triton_hota_v1_20261014/full/environment.json',
                       'runs/z13_triton_hota_v1_20261014/packages/triton_windows-3.6.0.post26.dist-info/RECORD']
    scope['main']()


if __name__ == '__main__':
    main()
