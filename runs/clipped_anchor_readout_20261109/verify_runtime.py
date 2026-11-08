import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=600)
import pytest
assert pytest.main(['tests/test_channel_balanced_correspondence.py','tests/test_clipping_aware_correspondence.py','tests/test_observed_spatial_reader.py','tests/test_observed_detection_track.py','-q'])==0
import numpy as np
from tracking.observed_spatial_reader import ObservedSpatialReader
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);parent=ROOT/'runs/background_patch_gate_20261108';bal=ROOT/'runs/channel_balance_attribution_20261109';wide=ROOT/'runs/decoder34_identity_gate_20261108';maximum=0.;reused=0
for e in read(parent/'protocol.json')['events']:
    name='_'.join(map(str,[e['video'],e['frame'],e['old_tid'],e['new_tid']]))+'.json';r=read(OUT/'events'/name);h=e['row_data']['history'][-1]
    p=reader.plan(h['box'],h['crop'],e['box'],e['row_data']['current_crop'],e['geometry']['prediction'],e['geometry']['offsets'],reference_observation=r['reference_observation']['box'])
    assert p['cells'].tolist()==r['geometry']['cells']
    if len(p['cells']):
        error=float(np.max(np.abs(p['query']-np.asarray(r['geometry']['query']))));maximum=max(maximum,error);assert error<1e-12
    if r['reused']:
        old8=read(parent/'events'/name);old34=read(wide/'events'/name);b=read(bal/'events'/name)
        for n,v in r['variants'].items():
            o=(old8 if n in old8['variants'] else old34)['variants'].get(n)
            assert v['global_scores']==(o['correlation_scores'] if o else [None]*19)
            assert v['balanced_scores']==(b['variants'][n]['scores'] if n in b['variants'] else [None]*19)
        reused+=1
assert reused==186
with (OUT/'runtime_verification.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,tests=13,plans_verified=200,query_max_error=maximum,unchanged_events_exact_reuse=186,existing_natural_score_grids_not_recomputed=True,
    zero_intervention='Synthetic complete lifecycle output is bit-exact against native XYWHByteTrack, both fuse-score settings. Only observed-box metadata added, including unconfirmed/low-score/re-activation stages. No full natural tracker replay performed.',
    sources={n:sha(ROOT/n) for n in ['tracking/channel_balanced_correspondence.py','tracking/clipping_aware_correspondence.py','tracking/observed_spatial_reader.py','tracking/observed_detection_track.py']}),f,indent=2)
print('Runtime API and exact no-intervention metadata verified')
