"""CPU-only geometry/baseline audit before the new candidate protocol is frozen."""
from pathlib import Path
import sys, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.z13_candidates_night import contextual_data
from tracking.z13_checkpoint import atomic_json

p=json.loads((ROOT/'runs/z13_distribution_pilot_20261015/protocol.json').read_text(encoding='utf-8'))
counts={}
for name,key in [('train104','fit'),('train26','evaluation')]:
    data=contextual_data(p[key])
    assert len(data.crops)==len(data.hex)
    assert np.isfinite(data.crops).all()
    assert (data.crops[:,2:]>data.crops[:,:2]).all()
    assert len(data.videos)==len(data.widths)
    counts[name]=dict(videos=len(p[key]),episodes=len(data.widths),frames=len(data.hex),labels=len(data.labels),
                      last_observed_size_positive=True,crops_valid=True)
    del data
atomic_json(Path(__file__).with_name('candidate_data_check.json'),dict(passed=True,counts=counts,gpu_used=False))
print(json.dumps(counts))
