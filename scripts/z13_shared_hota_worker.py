"""Existing Triton signature/evaluation path; one shared MASA stage for all arms."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.z13_compiled_backend import setup
setup()
RUN = ROOT / 'runs/z13_shared_masa_hota_v1_20261015/full'


def main():
    if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage') + 1] == 'shared_masa':
        from tracking.gpu_lock import GpuLock
        from scripts.z13_shared_masa import run_full
        with GpuLock('flyvis_fovea_gpu'):
            run_full(RUN)
        return
    from tracking.z13_compiled_backend import install
    from scripts import z13_spatial_hota_worker as worker
    from scripts import z13_spatial_hota_eval as evaluation
    install()
    worker.RUN = RUN
    evaluation.RUN = RUN
    worker.main()


if __name__ == '__main__':
    main()
