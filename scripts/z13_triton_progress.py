"""Read-only progress view for the isolated compiled full run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import z13_hota_console_progress as console

if __name__ == '__main__':
    console.RUN = ROOT / 'runs/z13_triton_hota_v1_20261014/full'
    try:
        console.main()
    except KeyboardInterrupt:
        pass
