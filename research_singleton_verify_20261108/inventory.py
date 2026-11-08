"""Find exact reusable own-track sequences before choosing a new intervention."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
import numpy as np
from research_singleton_verify_20261108.common import *


def main():
    assert read(PREVIOUS/'independent_audit.json')['passed'];exports=read(BASE/'export_complete.json')['videos']
    cached=read(BASE/'features_complete.json')['files'];queries=read(ROOT/'research_identity_assignment_protocol_20261108/queries.json')['rows']
    qlookup={(q['video'],q['frame'],q['tid']):q for q in queries};needed={};events=[];counts=Counter();sources={}
    for video,entry in exports.items():
        path=ROOT/entry['path'];assert sha(path)==entry['sha256'];data=read(path);sources[path.relative_to(ROOT).as_posix()]=entry['sha256']
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
            for frame,snap in data['snapshots'].items():
                native=snap['native_matches'];used_rows={i for i,j in native};used_columns={j for i,j in native};C=z['cost_'+frame]
                for i,j in snap['votes']:
                    row=snap['rows'][i];q=qlookup.get((video,int(frame),row['tid']));free=i not in used_rows and j not in used_columns and C[i,j]<1
                    calibrate=bool(q is not None and q['split']=='fit20' and C[i,j]<1)
                    if not free and not calibrate:continue
                    ref=[[h['frame'],h['crop']] for h in row['history']];current=ref+row['search']
                    keys=[]
                    for role,seq in [('reference',ref),('current',current)]:
                        key=sequence_key(video,seq);keys.append(key)
                        r=dict(key=key,video=video,sequence=seq,role=role,current_uid=row.get('current_uid') if role=='current' else None)
                        if key in needed:assert needed[key]['sequence']==seq and needed[key]['video']==video
                        else:needed[key]=r
                    events.append(dict(video=video,frame=int(frame),row=i,column=j,tid=row['tid'],query_uid=None if q is None else q['uid'],
                        calibration=calibrate,free=bool(free),reference_key=keys[0],current_key=keys[1]))
        relevant=[r for r in needed.values() if r['video']==video]
        if not relevant:continue
        archive=ROOT/f'runs/z13_data/{video}.npz';archive_sha=read(ROOT/'runs/z13_temporal_pair_readiness_20261025/protocol.json')['sha256'][archive.relative_to(ROOT).as_posix()]
        assert sha(archive)==archive_sha;sources[archive.relative_to(ROOT).as_posix()]=archive_sha
        with np.load(archive,allow_pickle=False) as z:frames=z['fr_frame'];crops=z['fr_crop']
        metadata={r['uid']:r for r in read(ROOT/f'runs/z13_temporal_pair_readiness_20261025/inputs/{video}.json')['rows']}
        for r in relevant:
            if r['key'] in cached:r['reuse_parent_sha256']=cached[r['key']];counts['parent_complete_maps']+=1;continue
            uid=r['current_uid']
            if uid is not None:
                m=metadata[uid];lo=m['state_prefix_start'];hi=m['current_array_index']+1
                saved=[[int(f),list(map(int,c))] for f,c in zip(frames[lo:hi],crops[lo:hi])]
                if saved==r['sequence']:r['reuse_current_uid']=uid;counts['current_sequence_matches_cache']+=1;continue
            counts['needs_new_reference' if r['role']=='reference' else 'needs_new_current']+=1
    result=dict(passed=True,requests=needed,events=events,counts=dict(counts),sources=sources,
        scope='No labels or new maps; only free-runtime edges plus fit20sole-candidate calibration examples, including already-owned boxes for negative identity examples.')
    json.dumps(result,allow_nan=False)
    write(RUN/'inventory_recovery1.json',result,new=True);print(json.dumps(dict(requests=len(needed),events=len(events),counts=dict(counts)),indent=2))


if __name__=='__main__':main()
