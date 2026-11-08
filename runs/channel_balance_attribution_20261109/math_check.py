import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=300)
import pytest
assert pytest.main(['tests/test_channel_balanced_correspondence.py','-q'])==0
with (OUT/'mathematical_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,tests=4,old_identity='S=sum_pc x_pc*y_pc/(||X||F||Y||F)=sum_c w_c*rho_c; sum(w)<=1.',
        candidate='Mean rho over exactly nonconstant reference channels. Current-constant channel contributes0 and remains in the denominator. No amplitude floor or threshold tuning.',
        properties=['Bounded[-1,1] by Cauchy-Schwarz.','Invariant to positive channel gains/offsets independently in either patch.','Equal-energy limit equals original centered correlation.','Zero-padding and uniform channel duplication do not increase evidence.'],
        caveat='Equal weights can magnify weak noisy channels. No proof of object identity or benefit follows from these identities. Attribution is not a causal neuron ablation.'),f,indent=2)
print('Mathematics passed before natural-map attribution')
