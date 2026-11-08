"""Reuse unexecuted pipeline plumbing in a separately frozen comparison version."""
from pathlib import Path
root=Path(__file__).resolve().parents[1];source=root/'research_bbox_spatial_signals_20261108';target=Path(__file__).parent
for name in ['setup_data.py','inputs.py','features.py','score.py']:
    text=(source/name).read_text(encoding='utf-8').replace('bbox_spatial_signals_20261108','bbox_normalized_signals_20261108')
    with (target/name).open('x',encoding='utf-8') as f:f.write(text)
