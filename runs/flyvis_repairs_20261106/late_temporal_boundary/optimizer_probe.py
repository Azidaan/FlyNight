"""Bounded CPU-only malformed optimizer probe; no saved model files changed."""
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import copy
import hashlib
import json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(1)
from tracking.b1_timestamped import make_timestamped_modules,timestamped_checkpoint,restore_timestamped_checkpoint
OUT=Path(__file__).resolve().parent
assert not (OUT/'optimizer_report.json').exists()
checks=[]
for field,value in [('lr',np.float32(np.nan)),('lr',np.float32(np.inf)),
                    ('eps',np.float32(np.nan)),('weight_decay',np.float32(np.nan)),
                    ('betas',(np.float32(np.nan),.999)),('lr',float('nan'))]:
    bb,head=make_timestamped_modules('R1',device='cpu')
    opt=torch.optim.Adam(head.parameters(),lr=.001)
    sum(p.sum() for p in head.parameters()).backward();opt.step();opt.zero_grad(set_to_none=True)
    checkpoint=timestamped_checkpoint(bb,head,cursor=1,optimizer=opt)
    checkpoint['head']['output.weight'].add_(10)
    checkpoint['optimizer']['param_groups'][0][field]=value
    old={k:v.clone() for k,v in head.state_dict().items()}
    before_rng=torch.random.get_rng_state().clone()
    flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    error=None;step_error=None
    try:cursor=restore_timestamped_checkpoint(checkpoint,bb,head,optimizer=opt)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    changed=any(not torch.equal(v,old[k]) for k,v in head.state_dict().items())
    after_restore_rng_equal=torch.equal(before_rng,torch.random.get_rng_state())
    if error is None:
        sum(p.sum() for p in head.parameters()).backward()
        try:opt.step()
        except Exception as exc:step_error=f'{type(exc).__name__}: {exc}'
    nonfinite=[k for k,v in head.state_dict().items() if not torch.isfinite(v).all()]
    checks.append(dict(field=field,value_repr=repr(value),value_type=type(value).__name__,
        restore_error=error,restore_success=error is None,head_changed_on_restore=changed,
        next_step_error=step_error,nonfinite_parameters_after_step=nonfinite,
        RNG_unchanged_by_restore=after_restore_rng_equal,
        deterministic_flags_unchanged=flags==(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())))
report=dict(checks=checks,cuda_initialized=torch.cuda.is_initialized(),source_sha256=hashlib.sha256((ROOT/'tracking/b1_timestamped.py').read_bytes()).hexdigest(),
    boundary='Only newly constructed diagnostic CPU heads and in-memory checkpoints; no historical weights, production sources or real training data changed.')
assert not report['cuda_initialized']
(OUT/'optimizer_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
