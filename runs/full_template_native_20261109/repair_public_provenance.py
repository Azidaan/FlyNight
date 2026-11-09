"""Repair exported pointers from exact public observations, never nearest boxes."""
from collections import defaultdict,Counter
import numpy as np


def repair(raw,flat,assignments):
    public=defaultdict(list);by_frame=defaultdict(lambda:defaultdict(list))
    for row in flat:public[(int(row[0]),int(row[5]))].append(row)
    for i,(frame,score,cls) in enumerate(zip(raw['frame_idx'],raw['conf'],raw['cls'])):
        by_frame[int(frame)][(float(np.float32(score)),int(cls))].append(i)
    counts=Counter();result=[];witnesses=[]
    for original in assignments:
        item=dict(original);rid=int(item['raw_id']);frame=item['frame'];rows=public.get((frame,item['tid']),[])
        assert not item['emitted'] or len(rows)==1,'Published assignment must have one actual output row'
        valid=0<=rid<len(raw['frame_idx']) and int(raw['frame_idx'][rid])==frame
        if valid and item['emitted']:
            row=rows[0]
            valid=np.float32(raw['conf'][rid])==np.float32(row[6]) and int(raw['cls'][rid])==int(row[7])
        if valid:
            item['raw_provenance']='logged_current_frame';counts['logged_current_frame']+=1
        else:
            counts['invalid_logged_pointer']+=1;item['logged_raw_id']=rid
            source_frame=int(raw['frame_idx'][rid]) if 0<=rid<len(raw['frame_idx']) else None
            counts['invalid_at_first_confirmation']+=int(frame==item['start_frame'])
            hits=by_frame[frame].get((float(np.float32(rows[0][6])),int(rows[0][7])),[]) if item['emitted'] else []
            if len(hits)==1:
                item['raw_id']=int(hits[0]);item['raw_provenance']='unique_exact_public_frame_score_class'
                counts['recovered_unique_public_pointer']+=1
            else:
                item['raw_id']=None;item['raw_provenance']='unknown_public_pointer_ambiguous_or_not_emitted'
                counts['unknown_pointer']+=1
            if len(witnesses)<12:
                witnesses.append(dict(frame=frame,tid=item['tid'],start_frame=item['start_frame'],
                    logged_raw_id=rid,logged_raw_frame=source_frame,recovered_raw_id=item['raw_id'],
                    exact_candidate_count=len(hits),emitted=item['emitted']))
        result.append(item)
    return result,dict(counts),witnesses
