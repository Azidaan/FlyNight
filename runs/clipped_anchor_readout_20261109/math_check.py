import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=300)
import pytest
assert pytest.main(['tests/test_clipping_aware_correspondence.py','-q'])==0
with (OUT/'mathematical_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,tests=4,
    model='A rigid translated interval is intersected with the fixed image bounds. If only the right edge is clipped, the observed left-edge displacement equals true translation; the observed center displacement generally does not. Mirror for other image edges.',
    domain='A shared visible edge and locally unchanged object scale are assumptions. Detector jitter, occlusion and perspective can violate them. This is a geometry/readout correction, not a claim about FlyVis core dynamics.',
    intervention='Override only translation components with an observable common unclipped image edge. Other axes and uncut cases preserve the original prediction exactly. No GT, overlap cutoff, scale fitting or learned threshold.'),f,indent=2)
print('Clipping mathematics passed before new raw-observation recovery')
