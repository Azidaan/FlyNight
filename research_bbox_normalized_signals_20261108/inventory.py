"""New alignment, same boxes and cost; measure readiness before image/core work."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_normalized_signals_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.box_normalized_correspondence import box_transform
    from tracking.own_template_verification import correspondence
    from tracking.spatial_identity_assignment import frame_points
    from research_identity_assignment_run_20261108.full_planner import dis_domain
    phase();prior=read(ROOT/'runs/bbox_spatial_readiness_20261108/inventory.json');g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');ready={};manifests={};requests={};events=[];counts=defaultdict(Counter);labels={}
    for prefix in ['singleton_verify_20261108','identity_assignment_20261108','native_confirm_policy_20261108']:
        folder=ROOT/'runs'/prefix;path=folder/'features_complete.json';manifests[path.relative_to(ROOT).as_posix()]=sha(path)
        for key,h in read(path)['files'].items():ready.setdefault(key,dict(path=(folder/'features'/(key+'.npz')).relative_to(ROOT).as_posix(),sha256=h))
    for old in prior['events']:
        e={k:v for k,v in old.items() if k not in ['reference_key','current_key']};row=e['row_data'];transform=box_transform(row['history'][-1]['box'],e['box']);reason=None;support=0
        if len(row['history'])<2:reason='reference_sequence_shorter_than_two_observations'
        else:
            cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pair=correspondence(transform,row['history'],row['current_crop'],e['box'],g,cv,rv)
            if pair is None:reason='fewer_than_three_common_reference_cells'
            else:support=len(pair['unique'])
        e.update(transform=transform,unavailable=reason,support=support);counts[e['split']]['cost_qualified_edges']+=1
        if e['video'] not in labels:labels[e['video']]=read(PRIOR_PILOT/'labels'/(e['video']+'.json'))['rows']
        l=labels[e['video']][f"{e['frame']}|{e['old_tid']}|{e['new_tid']}"];kind='correct' if l['known'] and l['positive'] else 'foreign' if l['known'] else 'unknown'
        if reason:counts[e['split']][reason]+=1
        else:
            counts[e['split']]['spatially_available_edges']+=1;counts[e['split']]['supported_'+kind]+=1;ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];keys=[]
            for role,seq in [('reference',ref),('current',cur)]:
                key=sequence_key(e['video'],seq);keys.append(key)
                if key not in requests:requests[key]=dict(key=key,video=e['video'],sequence=seq,role=role,current_uid=None,ready=ready.get(key))
            e['reference_key'],e['current_key']=keys
        events.append(e)
    reused=0
    for r in requests.values():
        if r['ready']:assert sha(ROOT/r['ready']['path'])==r['ready']['sha256'];reused+=1
    missing=len(requests)-reused;views={(r['video'],int(f),tuple(c)) for r in requests.values() if not r['ready'] for f,c in r['sequence']};proceed=counts['fit20']['supported_correct']>0 and counts['evaluate26']['spatially_available_edges']>=10 and missing<=200 and len(views)<=1000
    write(RUN/'inventory.json',dict(passed=True,events=events,requests=requests,counts={k:dict(v) for k,v in counts.items()},ready_manifests=manifests,ready_sequences=reused,missing_sequences=missing,views_upper_bound=len(views),input_restore_gate=dict(proceed=bool(proceed)),scope='New observed-box normalization. Native boxes, alpha0.025, age2 and eye crop fixed. No biological signal or quality claim; labeled fit support is a logical readiness condition only.'),new=True);status('bbox_normalized_readiness',counts={k:dict(v) for k,v in counts.items()},missing=missing,proceed=bool(proceed),guard=obs.read());phase()


if __name__=='__main__':main(guard())
