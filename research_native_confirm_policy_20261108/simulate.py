"""One causal label policy on fixed native boxes and fixed, past-only features."""
import numpy as np
from tracking.native_confirm_identity import NativeConfirmedIdentity
from tracking.own_template_verification import passes


def simulate(arm,outputs,events,scores,protocol,frames):
    policy=NativeConfirmedIdentity(enabled=arm!='B0');byframe={r['frame']:r for r in events};flat=[];decisions=[]
    for frame in range(frames):
        native=outputs[outputs[:,0]==frame,1:];event=byframe.get(frame,dict(newly_confirmed=[],edges=[]));proposals=[]
        for e in event['edges']:
            if arm=='GEOMETRY':accepted=True
            elif arm=='AGE':accepted=e['age']<=protocol['recency_cap']
            elif arm=='B0':accepted=False
            else:
                key=f"{frame}|{e['old_tid']}|{e['new_tid']}";accepted=passes(scores[key]['scores'][arm],protocol['thresholds'][arm])
            proposals.append(dict(old_tid=e['old_tid'],new_tid=e['new_tid'],cost=e['cost'],accepted=accepted))
        out,d=policy.update(frame,native,event['newly_confirmed'],proposals)
        if len(out):
            assert np.array_equal(out[:,[0,1,2,3,5,6,7]],native[:,[0,1,2,3,5,6,7]])
            flat.append(np.column_stack([np.full(len(out),frame),out]).astype(np.float32))
        if d['links'] or d['revoked']:decisions.append(dict(frame=frame,**d))
    result=np.concatenate(flat) if flat else np.empty((0,9),np.float32)
    assert np.array_equal(result[:,[0,1,2,3,4,6,7,8]],outputs[:,[0,1,2,3,4,6,7,8]])
    if arm=='B0':assert np.array_equal(result,outputs)
    return result,decisions
