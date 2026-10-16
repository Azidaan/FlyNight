"""Read-only progress for the separately prepared cached execution."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import z13_hota_console_progress as console

if __name__ == '__main__':
    console.RUN = ROOT / 'runs/z13_cached_hota_v1_20261015/full'
    try:
        console.main()
    except KeyboardInterrupt:
        pass
