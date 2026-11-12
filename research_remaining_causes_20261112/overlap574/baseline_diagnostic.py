import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from tracking.observed_spatial_reader import ObservedSpatialReader
from tracking.full_template_correspondence import relative_template_plan
from tracking.hex_linear_interpolation import sample_linear
from tracking.channel_balanced_correspondence import _center_pair
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
p=read(ROOT/'research_stage_localization_20261112/protocol.json');c=p['cases'][0];r=c['case'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');reader=ObservedSpatialReader(g['points'],g['eye_size'],[1280,720]);plan=relative_template_plan(reader,r['reference_box'],r['reference_crop'],r['current_box'],r['current_crop'],r['offsets']);query=plan['query'][:,r['old_variants']['C34']['winner']];h=read(ROOT/'research_candidate_support_probe_20261112/scores.json')['heads']['C34']['native'];scores={};errors={}
with np.load(ROOT/'research_input_precision_probe_20261112/maps.npz') as z:
 for mode in ['replayed','archived']:
  a,b=[z[f'index_0_{i}_C34'].astype(float) if mode=='replayed' else np.load(ROOT/c['endpoints'][i]['paths']['C34']).astype(float) for i in [0,1]];x,y,an,bn,active,activeb=_center_pair(a[plan['cells']],sample_linear(reader.mesh,b,query));q=np.divide((x*y).sum(0),an*bn,out=np.zeros(34),where=an*bn>0)*34/active.sum();features=np.r_[(np.array(r['geometry'])-h['geometry']['mean'])/h['geometry']['scale'],q];t=np.array(h['theta']);scores[mode]=float(features@t[:-1]+t[-1])
 for i in [0,1]:errors[str(i)]=float(abs(z[f'index_0_{i}_C34'].astype(float)-np.load(ROOT/c['endpoints'][i]['paths']['C34']).astype(float)).max())
expected=next(x['score'] for x in read(ROOT/'research_candidate_support_probe_20261112/scores.json')['records'] if x['old_id']==232 and x['representation']=='C34' and x['head']=='native' and x['mode']=='full')
out=dict(case=232,representation='C34',head='native',scores=scores,expected_frozen_score=expected,replayed_minus_expected=scores['replayed']-expected,archived_minus_expected=scores['archived']-expected,map_max_errors=errors,root_cause='maps.npz index is an independently replayed neural output whose original parity gate allowed1e-4; prior candidate-support frozen score used archived endpoint files directly. Comparing them with1e-9 is invalid baseline provenance. Fix is archived files, not relaxed tolerance.',intervention_outcomes_computed=False)
with (OUT/'baseline_diagnostic.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
