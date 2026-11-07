"""Тот же экспорт; только кэш неизменных массивов NPZ, прежний срок."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b1_npz_recovery_20261026 import install
from scripts import b1_export_base_20261027 as worker
if __name__ == '__main__':
    install()
    worker.OUT = ROOT / 'runs/b123_20261027/b1_base_export_recovery1'
    worker.main()
