"""Independent saved-artifact verification; imports no tracking implementation."""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=OUT/'real_D_v5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


protocol=read(RUN/'protocol.json')
report=read(RUN/'report.json')
row=read(ROOT/'runs/b1_real_inputs_20261026/gpu_samples.json')['rows'][0]
with np.load(ROOT/row['retinal_source']) as data:
    frames=data['fr_frame'][row['prefix_start']:row['current_array_index']+1]
with np.load(RUN/'outputs.npz') as data:
    assert np.array_equal(data['frames'],frames)
    assert np.array_equal(data['expected'],data['actual'],equal_nan=True)
    assert data['actual'].shape==(len(row['choices']),)
    assert np.isnan(data['actual'][0]) and np.isfinite(data['actual'][1:]).all()
    real_max_delta=float(np.max(np.abs(data['actual'][1:]-data['expected'][1:])))
fps=read(ROOT/f'runs/train_cache/{row["video"]}/motion/flyvis_tiled.json')['fps']
gaps=np.diff(frames)
durations=gaps/fps
steps=np.ceil(durations/.02-1e-12).astype(int)
dt=durations/steps
assert np.all(gaps>0) and np.all(dt<=.02)
assert abs(durations.sum()-report['elapsed_input_seconds'])<1e-12
assert report['source_fps']==fps
assert report['adjacent_max_abs_difference']==0
assert report['diagnostic_rejected'] and not report['cuda_initialized']
assert report['checkpoint_restore_after_mutation_exact'] and report['optimizer_restore_exact']
assert not report['hota_measured'] and not report['scientific_hypothesis_accepted']
assert all(value>0 for value in report['core_gradient_norms'][-1].values())
current_hashes={name:sha(ROOT/name)==digest for name,digest in protocol['sha256'].items()}
assert all(current_hashes.values())
for stem in ('checkpoint','outputs','protocol'):
    filename={'checkpoint':'diagnostic_checkpoint.pt','outputs':'outputs.npz','protocol':'protocol.json'}[stem]
    assert sha(RUN/filename)==report[f'{stem}_sha256']
checkpoint=torch.load(RUN/'diagnostic_checkpoint.pt',map_location='cpu',weights_only=False)
assert checkpoint['provenance']['diagnostic_only'] is True
assert checkpoint['provenance']['protocol_sha256']==sha(RUN/'protocol.json')
assert checkpoint['temporal_semantics']['schema']=='b1_observed_frame_time_v1'
assert checkpoint['temporal_semantics']['source_clock']=='frame_delta / actual_source_fps'
assert checkpoint['cursor']==2
for section in ('backbone','head'):
    assert checkpoint[section] and all(torch.isfinite(v).all() for v in checkpoint[section].values())
early=read(OUT/'real_D/protocol.json')
early_snapshots={name:sha(OUT/'real_D/sources'/Path(name).name)==digest
    for name,digest in early['sha256'].items()
    if Path(name).name in ('b1_timestamped.py','b1_timestamped_smoke_20261106.py')}
assert all(early_snapshots.values())
result=dict(passed=True,scope='Independent formulas and saved-output/hash verification; shares primary saved data; does not rerun tracker or establish HOTA.',
    frames=frames.tolist(),gaps=gaps.tolist(),source_fps=fps,interval_seconds=durations.tolist(),
    integration_counts=steps.tolist(),integration_dt=dt.tolist(),elapsed_seconds=float(durations.sum()),
    legacy_four_intervals_at_30fps_seconds=4/30,streamed_vs_offline_max_abs_delta=real_max_delta,
    protocol_sources_current_hash_match=current_hashes,early_source_snapshots_match=early_snapshots,
    diagnostic_checkpoint_finite=True,cuda_initialized=torch.cuda.is_initialized(),script_sha256=sha(__file__))
assert not result['cuda_initialized']
(OUT/'independent_validation_v5.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
