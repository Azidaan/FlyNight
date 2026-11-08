from pathlib import Path
root=Path(__file__).resolve().parents[1];text=(root/'research_bbox_singleton_pilot_20261108/audit.py').read_text(encoding='utf-8');old="assert abs(c['cost']-(1-value*c['confidence']))<2e-6"
assert text.count(old)==1
text=text.replace('from research_bbox_singleton_pilot_20261108.common import *','from research_bbox_singleton_pilot_20261108.common import *\nfrom research_bbox_singleton_audit_recovery1_20261108.native_cost import native_cost')
text=text.replace(old,"assert native_cost(row['row_data']['mean'],c['box'],c['confidence'])==c['cost']")
with (Path(__file__).parent/'audit.py').open('x',encoding='utf-8') as f:f.write(text)
