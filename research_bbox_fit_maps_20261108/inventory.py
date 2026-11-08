import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_fit_maps_20261108.common import *


def main(obs):
    import numpy as np
    from tracking.bbox_singleton_gate import proposals
    from tracking.box_normalized_correspondence import box_transform
    from tracking.own_template_verification import correspondence
    from tracking.spatial_identity_assignment import frame_points
    from research_identity_assignment_run_20261108.full_planner import dis_domain
    phase();entries=read(INVENTORY/'protocol.json')['videos'];g=read(ROOT/'runs/z13_local_peak_20261021/geometry.json');ready={};requests={};events=[];counts=Counter();manifests={}
    for prefix in ['bbox_normalized_signals_20261108','singleton_verify_20261108','native_confirm_policy_20261108','identity_assignment_20261108']:
        folder=ROOT/'runs'/prefix;path=folder/'features_complete.json';manifests[path.relative_to(ROOT).as_posix()]=sha(path)
        for key,h in read(path)['files'].items():ready.setdefault(key,dict(path=(folder/'features'/(key+'.npz')).relative_to(ROOT).as_posix(),sha256=h))
    for video,entry in entries.items():
        if entry['split']!='fit20':continue
        labels=read(PRIOR_PILOT/'labels'/(video+'.json'))['rows']
        for d in read(INVENTORY/'native'/(video+'.json'))['events']:
            for edge in proposals(d,cap=6,threshold=.2,bidirectional=False):
                row=edge['row_data'];c=next(c for r in d['rows'] if r['old_tid']==edge['old_tid'] for c in r['candidates'] if c['new_tid']==edge['new_tid']);transform=box_transform(row['history'][-1]['box'],c['box']);cv=dis_domain(frame_points(g['points'],row['current_crop'],g['eye_size']));rv=dis_domain(frame_points(g['points'],row['history'][-1]['crop'],g['eye_size']));pairs=correspondence(transform,row['history'],row['current_crop'],c['box'],g,cv,rv);reason='reference_sequence_shorter_than_two_observations' if len(row['history'])<2 else 'fewer_than_three_reference_cells' if pairs is None else None
                e=dict(edge,video=video,split='fit20',frame=d['frame'],box=c['box'],transform=transform,unavailable=reason,support=0 if reason else len(pairs['unique']));label=labels[f"{d['frame']}|{edge['old_tid']}|{edge['new_tid']}"];kind='positive' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';counts['all_'+kind]+=1
                if not reason:
                    counts['supported_'+kind]+=1;ref=[[h['frame'],h['crop']] for h in row['history']];cur=ref+row['search'];keys=[]
                    for role,seq in [('reference',ref),('current',cur)]:
                        key=sequence_key(video,seq);keys.append(key)
                        if key not in requests:requests[key]=dict(key=key,video=video,sequence=seq,role=role,current_uid=None,ready=ready.get(key))
                    e['reference_key'],e['current_key']=keys
                events.append(e)
    reused=0
    for r in requests.values():
        if r['ready']:assert sha(ROOT/r['ready']['path'])==r['ready']['sha256'];reused+=1
    missing=len(requests)-reused;views={(r['video'],int(f),tuple(c)) for r in requests.values() if not r['ready'] for f,c in r['sequence']};proceed=counts['supported_positive']>=3 and counts['supported_foreign']>=3 and missing<=400 and len(views)<=2000
    phase();write(RUN/'inventory.json',dict(passed=True,events=events,requests=requests,counts=dict(counts),ready_manifests=manifests,ready_sequences=reused,missing_sequences=missing,views_upper_bound=len(views),input_restore_gate=dict(proceed=bool(proceed)),scope='All200fit ordinary-singleton proposals, including154unknown pairs; no GT-based runtime or data selection. Labels describe support only. No26or20valdata used.'),new=True);status('fit_map_restoration_inventory',counts=dict(counts),missing=missing,proceed=bool(proceed),guard=obs.read())


if __name__=='__main__':main(guard())
