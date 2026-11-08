"""Measure spatial availability and exact saved-map reuse before rendering inputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_spatial_readiness_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.bbox_singleton_gate import proposals
    from tracking.spatial_identity_assignment import geometric_match,frame_points
    from tracking.own_template_verification import correspondence
    from research_identity_assignment_run_20261108.full_planner import dis_domain
    p=phase();selection=read(COST/'selection.json');assert selection['selected']=='A025_C2';g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');allvideos=read(INVENTORY/'protocol.json')['videos'];requests={};events=[];counts=defaultdict(Counter);ready={};ready_manifests={}
    for name in ['singleton_verify_20261108','identity_assignment_20261108','native_confirm_policy_20261108']:
        folder=ROOT/'runs'/name;manifest=folder/'features_complete.json'
        if not manifest.exists():continue
        ready_manifests[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
        for key,h in read(manifest)['files'].items():ready.setdefault(key,dict(path=(folder/'features'/(key+'.npz')).relative_to(ROOT).as_posix(),sha256=h))
    for video,entry in allvideos.items():
        if entry['split']=='val20':continue
        data=read(INVENTORY/'native'/(video+'.json'));split=entry['split']
        for d in data['events']:
            for e in proposals(d,cap=2,threshold=.2,bidirectional=False):
                if np.float32(e['cost']-.025)>.8:continue
                counts[split]['cost_qualified_edges']+=1;row=e['row_data'];c=next(c for r in d['rows'] if r['old_tid']==e['old_tid'] for c in r['candidates'] if c['new_tid']==e['new_tid']);reason=None;transform=None;support=0
                if len(row['history'])<2:reason='reference_sequence_shorter_than_two_observations'
                else:
                    transform=geometric_match(row['mean'],row['covariance'],row['history'],row['current_crop'],c['box'],g)
                    cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pair=correspondence(transform,row['history'],row['current_crop'],c['box'],g,cv,rv)
                    if pair is None:reason='no_valid_common_three_reference_cells'
                    else:support=len(pair['unique'])
                item=dict(video=video,split=split,frame=d['frame'],old_tid=e['old_tid'],new_tid=e['new_tid'],cost=e['cost'],column=e['column'],box=c['box'],age=e['age'],row_data=row,transform=transform,support=support,unavailable=reason)
                if reason:counts[split][reason]+=1
                else:
                    counts[split]['spatially_available_edges']+=1;ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];keys=[]
                    assert cur[-1][0]==d['frame'] and max(t for t,crop in ref)<d['frame']
                    for role,seq in [('reference',ref),('current',cur)]:
                        key=sequence_key(video,seq);keys.append(key)
                        if key not in requests:requests[key]=dict(key=key,video=video,sequence=seq,role=role,current_uid=None,ready=ready.get(key))
                    item['reference_key'],item['current_key']=keys
                events.append(item)
        status('bbox_spatial_metadata_inventory',video=video,split=split,counts={k:dict(v) for k,v in counts.items()},guard=obs.read())
    reused=0
    for r in requests.values():
        if r['ready']:assert sha(ROOT/r['ready']['path'])==r['ready']['sha256'];reused+=1
    views={(r['video'],int(f),tuple(c)) for r in requests.values() if not r['ready'] for f,c in r['sequence']};missing=len(requests)-reused
    proceed=counts['fit20']['spatially_available_edges']>=5 and counts['evaluate26']['spatially_available_edges']>=10 and missing<=200 and len(views)<=1000
    phase();write(RUN/'inventory.json',dict(passed=True,events=events,requests=requests,counts={k:dict(v) for k,v in counts.items()},ready_manifests=ready_manifests,ready_sequences=reused,missing_sequences=missing,views_upper_bound=len(views),input_restore_gate=dict(proceed=bool(proceed),criterion=p['gate']),new_core_calls=0,new_image_reads=0,new_detector_calls=0,scope='Spatial availability only for the fit-selected small-cost singleton. No labels used here, no new signal quality claim. SUPPORT must remain a no-FlyVis control.'),new=True)


if __name__=='__main__':main(guard())
