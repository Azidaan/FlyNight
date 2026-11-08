import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=300)
import pytest
assert pytest.main(['tests/test_channel_balanced_correspondence.py::test_spatial_and_amplitude_terms_sum_to_the_exact_channel_difference','-q'])==0
with (OUT/'delta_mathematical_checks.json').open('x',encoding='utf8') as f:
    json.dump(dict(passed=True,identity='wF*rF-wO*rO=(wF+wO)*(rF-rO)/2+(rF+rO)*(wF-wO)/2',
        role='Separate relative amplitude weighting and spatial correlation in the exact foreign-minus-own score gap. Both terms can be signed; percentages can exceed100% under cancellation.',
        not_causal='Per-channel centering and a shared denominator couple observations. The identity does not establish biological signal ownership or a defective neuron.'),f,indent=2)
print('Delta identity passed before natural-map attribution')
