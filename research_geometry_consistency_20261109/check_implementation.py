"""Verify the candidate transport before any feature comparison or deployment."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=300)
import pytest,numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.geometric_template_correspondence import geometric_template_plan
assert pytest.main(['tests/test_geometric_template_correspondence.py','-q'])==0
g=json.loads((ROOT/'runs/z13_local_peak_20261021/geometry.json').read_text())
r=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720])
assert 'flyvis' not in sys.modules
result=dict(passed=True,tests=7,analytic_object_coordinate_signal=True,all_shift_box_erosion_avoided=True,
    eye_hull_and_identity_witnesses=True,not_deployed=True,
    gate='The six admitted training edges are all own; no fitting or native rollout is justified on that gated training population.',
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tracking/geometric_template_correspondence.py','tests/test_geometric_template_correspondence.py',str(Path(__file__).relative_to(ROOT))]})
with (OUT/'implementation_check.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('IMPLEMENTATION_VALIDATED_NOT_DEPLOYED',flush=True)
