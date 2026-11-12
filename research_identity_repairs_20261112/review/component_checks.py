"""Synthetic checks of the actual dormant component, never the five outcomes."""
import os,sys,json,hashlib,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'research_identity_repairs_20261112/recovery574'));sys.path.insert(0,str(ROOT/'research_identity_repairs_20261112/geometry832'))
import numpy as np
from combined_adapter_v2 import compose,complete_support
from appearance_recovery_policy_v4 import observable_episode_tids
parser=argparse.ArgumentParser();parser.add_argument('--go-protocol-sha256',required=True);args,private=parser.parse_known_args();assert all(x.startswith('--audit-memory-worker=') for x in private);assert hashlib.sha256((OUT/'combined_protocol_v2.json').read_bytes()).hexdigest()==args.go_protocol_sha256
rows=[dict(tid=1,state=2,cls=2,predicted_xyxy=[0,0,10,10]),dict(tid=2,state=1,cls=2,predicted_xyxy=[20,0,30,10])]
candidates=[dict(raw_id=10,box=[20,0,30,10],confidence=.8,cls=2),dict(raw_id=11,box=[0,0,10,10],confidence=.05,cls=2)]
a=np.eye(2,dtype=float);b=np.array([[0.,1.],[1.,0.]])
snapshot=dict(video='synthetic',frame=10,target_tid=1,target_raw_id=10,target_row=0,target_column=0,high_raw_ids=[10],actual_first=[[1,0]],assigned={'2':10},base_cost=[[.95],[.05]],corrected_cost=[[.75],[.05]],native_proposals=[dict(old_id=1,geometry=[0]*7,reason='head_refuse')])
variant='guarded_census_appearance_low_native_domain';checks=[]
r,n=compose(snapshot,rows,candidates,a,b,.5,.6,variant);assert r['freeze']=='all' and len(r['append_columns'])==1 and r['append_columns'][0]['raw_id']==11 and r['append_columns'][0]['costs'][0]==0 and n['action']=='recover_excluded_low';assert all(u['tid']!=2 for u in r['edge_updates']);checks.append('Freeunmatchedlostrow recovers lowbox while activepair frozen')
assert observable_episode_tids([dict(old_id=1,geometry=[0]*7,reason='head_accept'),dict(old_id=2,geometry=[0]*7,reason='head_refuse'),dict(old_id=3,reason='outside_observed_training_gap')])==frozenset([1,2]);checks.append('Nativeepisodeeligibility independentofpositive/negativeheadsign')
blocked=dict(snapshot,native_proposals=[]);r,n=compose(blocked,rows,candidates,a,b,.5,.6,variant);assert not r['append_columns'] and r['defer_tids']==[1];checks.append('Emissionrestrictedwhile fullcompetitionretained')
tie=np.array([[1.,0.],[1.,0.]]);r,n=compose(snapshot,rows,candidates,tie,b,.5,.6,variant);assert not r['append_columns'];checks.append('Incompleteuniquenesscannotauthorize recovery')
r,n=compose(snapshot,rows,candidates[:1],a,b[:1],.5,.6,variant);assert not r['append_columns'] and r['defer_tids']==[1];checks.append('Missingownappearancealternative defers ratherthanforcedmax')
r,n=compose(snapshot,rows,candidates,a,b,-100.,-100.,'targetdecision_without_core_appearance_low_native_domain');assert len(r['append_columns'])==1;checks.append('Appearancecontrol doesnotuse coreheadvalues')
frozen=dict(snapshot,assigned={'2':10,'3':11});r,n=compose(frozen,rows,candidates,a,b,.5,.6,variant);assert not r['append_columns'];checks.append('OccupieddownstreamIDoutsidefirstrows protected')
with (OUT/'component_checks.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,checks=checks,cases=len(checks),scope='Syntheticactualcomponentchecks, notfive-scene scientificoutcomes; no GPU/model/tracker.'),f,ensure_ascii=False,indent=2)
print('COMPONENT_CHECKS_PASS',len(checks))
