"""Reuse mandatory catalog/protected-graph QA, with local output receipts."""
from pathlib import Path
source=Path(__file__).resolve().parents[1]/'research_visible_geometry_head_20261109/preflight.py'
exec(compile(source.read_text(encoding='utf8'),str(source),'exec'))
