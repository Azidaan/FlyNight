"""Required metadata and protected graph checks before new fitting."""
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'research_visible_geometry_head_20261109/preflight.py'
exec(compile(source.read_text(encoding='utf8'),str(source),'exec'))
