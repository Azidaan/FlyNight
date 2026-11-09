"""Check newly registered metadata and preserve the protected graph."""
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'research_visible_geometry_head_20261109/preflight.py'
exec(compile(source.read_text(encoding='utf8'),str(source),'exec'))
