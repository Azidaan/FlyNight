"""New isolated execution root; source pipeline, models and action unchanged."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_compiled_cached_backend import setup
setup()


def main():
    from tracking.z13_compiled_cached_backend import install
    from scripts import z13_spatial_hota_worker as worker
    from scripts import z13_spatial_hota_eval as evaluation
    install()
    worker.RUN = ROOT / 'runs/z13_cached_hota_v1_20261015/full'
    evaluation.RUN = worker.RUN
    worker.main()


if __name__ == '__main__':
    main()
