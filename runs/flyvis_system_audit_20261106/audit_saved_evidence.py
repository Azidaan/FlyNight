"""Read saved timing metadata; independently recompute boundary identities."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r={}
samples=ROOT/'runs/b1_real_inputs_20261026/gpu_samples.json'
row=json.loads(samples.read_text(encoding='utf-8'))['rows'][0]
p=ROOT/row['retinal_source']
with np.load(p) as z:
    frames=z['fr_frame'][row['prefix_start']:row['current_array_index']+1].tolist()
r['saved_training_sample']={'uid':row['uid'],'frames':frames,'gaps':np.diff(frames).tolist(),'source':str(p.relative_to(ROOT)),'source_sha256':sha(p),'selection':str(samples.relative_to(ROOT)),'selection_sha256':sha(samples)}
folder=ROOT/'runs/recovery_research_20261027/b1_tracker/continuation_recovery1/events/D'
p=min(folder.glob('*.json'),key=lambda x:x.stat().st_size)
events=json.loads(p.read_text(encoding='utf-8'))
first={}
for e in events:
    key=(e['track'],e['reference_frame'])
    first.setdefault(key,e)
gaps=collections.Counter(e['frame']-e['reference_frame'] for e in first.values())
r['saved_live_events']={'source':str(p.relative_to(ROOT)),'sha256':sha(p),'selection':'smallest existing D event file by byte size; descriptive single-video audit','events':len(events),'episodes':len(first),'first_query_gap_histogram':dict(gaps)}
b=json.loads((OUT/'boundary_report.json').read_text())
checks=[]
for x in b['flow_inverse_resize']:
    checks.append(abs(x['ratio']-x['height']/x['width'])<1e-6)
calls=b['b1_timestamp_trace']['calls']
r['independent_boundary_validation']={'conversion_ratio_equals_height_over_width':all(checks),'square_control_ratio_one':abs(b['flow_inverse_resize'][0]['ratio']-1)<1e-6,'first_state_jump_two_frames':calls[1]['advance_to']-calls[1]['advance_from']==2,'reported_interval_one_frame':abs(calls[1]['fixed_backbone_steps']*calls[1]['fixed_backbone_dt']-1/30)<1e-12,'shared_inputs':'Reads boundary_report.json; independent formulas, not independent execution of network or tracker.'}
r['source_hashes_unchanged']={p:sha(ROOT/p)==h for p,h in b['source_sha256'].items()}
r['script_sha256']=sha(Path(__file__))
(OUT/'saved_evidence.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(r,ensure_ascii=False,indent=2))
