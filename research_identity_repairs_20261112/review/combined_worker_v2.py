"""CPU-only proposal composition after exact parent protocol GO; no solver."""
import os,sys,json,hashlib,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_identity_repairs_20261112/recovery574'));sys.path.insert(0,str(ROOT/'research_identity_repairs_20261112/geometry832'))
import numpy as np
from combined_adapter_v2 import compose,VARIANTS
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(n,x):
 with (OUT/n).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True);args,private=parser.parse_known_args();assert all(x.startswith('--audit-memory-worker=') for x in private);assert sha(OUT/'combined_protocol_v2.json')==args.go_protocol_sha256
def verify():
 p=read(OUT/'combined_protocol_v2.json')
 for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
 return p
p=verify();assert read(OUT/'bank_audit.json')['passed'];snapshots=read(OUT/'assignment_snapshots.json')['snapshots'];scores=read(ROOT/'research_identity_repairs_20261112/geometry832/result.json')['records'];records=[];notes=[]
with np.load(OUT/'reference_bank.npz') as bank,np.load(ROOT/'research_remaining_causes_20261112/independent_review/reid/low_descriptors.npz') as current:
 for snapshot in snapshots:
  rows=[{k:r[k] for k in ['tid','state','cls','predicted_xyxy']} for r in snapshot['rows']];candidates=[{k:r[k] for k in ['raw_id','box','confidence','cls']} for r in snapshot['raw_current_car_candidates']]
  references=np.stack([bank[r['reference']['slot'][0]][r['reference']['slot'][1]] for r in snapshot['rows']]);vectors=current[str(snapshot['old_case'])].copy()
  original=next(r['score'] for r in scores if r['old_id']==snapshot['old_case'] and r['representation']=='C34' and r['head']=='native' and r['coordinate']=='original' and r['reference_mode']=='index')
  census=next(r['score'] for r in scores if r['old_id']==snapshot['old_case'] and r['representation']=='C34' and r['head']=='native' and r['coordinate']=='census' and r['reference_mode']=='terminal_hold')
  minimal={k:snapshot[k] for k in ['video','frame','target_tid','target_raw_id','target_row','target_column','high_raw_ids','actual_first','assigned','base_cost','corrected_cost']}
  minimal['native_proposals']=[{k:q[k] for k in ['old_id','reason','geometry'] if k in q} for q in snapshot['proposals']]
  for variant in VARIANTS:
   record,note=compose(minimal,rows,candidates,references.copy(),vectors.copy(),original,census,variant);records.append(record);notes.append(dict(video=snapshot['video'],frame=snapshot['frame'],variant=variant,**note))
verify();save('combined_decisions_v2.json',dict(records=records,scope=p['scope']));save('combined_policy_notes_v2.json',dict(passed=True,records=notes,model_calls=0,tracker_calls=0,new_fits=0,source_protocol_sha256=sha(OUT/'combined_protocol_v2.json')))
assert p['authorize_downstream_assignment_audit'] is True
names=['assignment_audit_v2.py','assignment_snapshots.json','combined_decisions_v2.json','combined_protocol_v2.json'];save('combined_assignment_GO_v2.json',dict(approved=True,authorization='Parent executes preregistered combined_worker_v2 with exact protocol hash; protocol authorizes downstream saved-matrix audit.',source_protocol_sha256=sha(OUT/'combined_protocol_v2.json'),inputs={str((OUT/n).relative_to(ROOT)).replace('\\','/'):sha(OUT/n) for n in names}))
print('COMBINED_PROPOSALS_PREPARED',len(records))
