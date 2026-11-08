"""Identical public-ID mechanism; one bounded gate shared by signal and controls."""
import hashlib,json
import numpy as np
from tracking.native_confirm_identity import NativeConfirmedIdentity


def accepted(config,edge,frame,scores):
    if not config.get('enabled',True) or edge['age']>config['cap']:return False
    kind=config.get('score_name')
    if kind is None:return True
    threshold=config['threshold'];value=scores[f"{frame}|{edge['old_tid']}|{edge['new_tid']}"]['scores'][kind]
    return threshold!='off' and value is not None and np.isfinite(value) and value>=threshold


def gate_signature(config,events,scores):
    values=[bool(accepted(config,e,d['frame'],scores)) for d in events for e in d['edges']]
    return hashlib.sha256(json.dumps([bool(config.get('enabled',True)),values],separators=(',',':')).encode()).hexdigest()


def simulate(native,metadata,scores,config,*,keep_outputs=True):
    policy=NativeConfirmedIdentity(enabled=config.get('enabled',True));events={r['frame']:r for r in metadata['events']};counts=np.bincount(native[:,0].astype(int),minlength=metadata['frames']);offsets=np.r_[0,counts.cumsum()];assert np.all(native[1:,0]>=native[:-1,0]);flat=[];decisions=[]
    for frame in range(metadata['frames']):
        rows=native[offsets[frame]:offsets[frame+1],1:];event=events.get(frame,dict(newly_confirmed=[],edges=[]));edges=[dict(old_tid=e['old_tid'],new_tid=e['new_tid'],cost=e['cost'],accepted=bool(accepted(config,e,frame,scores))) for e in event['edges']];out,d=policy.update(frame,rows,event['newly_confirmed'],edges)
        if keep_outputs and len(out):
            assert np.array_equal(rows[:,[0,1,2,3,5,6,7]],out[:,[0,1,2,3,5,6,7]])
            flat.append(np.column_stack([np.full(len(out),frame),out]).astype(np.float32))
        if d['links'] or d['revoked']:decisions.append(dict(frame=frame,**d))
    result=None if not keep_outputs else np.concatenate(flat) if flat else np.empty((0,9),np.float32)
    if keep_outputs:assert np.array_equal(result[:,[0,1,2,3,4,6,7,8]],native[:,[0,1,2,3,4,6,7,8]])
    return result,decisions
