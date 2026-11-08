"""Investigate a failed independent IoU check without recomputing tracker outputs."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_bbox_singleton_pilot_20261108.common import *


def main():
    import numpy as np
    from research_bbox_singleton_pilot_20261108.audit import overlap
    from ultralytics.trackers.bot_sort import BOTrack,parse_bboxes
    from tracking.offline_association import _DetSlice
    from ultralytics.trackers.utils.matching import iou_distance,fuse_score
    errors=[];native_error=0;checks=0;changed_gate=0
    for video in read(INVENTORY/'protocol.json')['videos']:
        data=read(INVENTORY/'native'/(video+'.json'))
        for d in data['events']:
            for r in d['rows']:
                mean=np.array(r['row_data']['mean']);pred=np.r_[mean[:2]-mean[2:4]/2,mean[:2]+mean[2:4]/2]
                for c in r['candidates']:
                    a=_DetSlice(np.asarray([c['box']],np.float32),np.array([c['confidence']],np.float32),np.array([r['row_data']['cls']],np.float32));det=BOTrack(parse_bboxes(a)[0],a.conf[0],a.cls[0]);track=BOTrack(np.r_[mean[:4],0],a.conf[0],a.cls[0]);track.mean=mean.copy();value=float(fuse_score(iou_distance([track],[det]),[det])[0,0]);error=abs(value-c['cost']);native_error=max(native_error,error);approx=overlap(pred,c['box']);errors.append(dict(video=video,frame=d['frame'],cost_error_float64_raw_box=abs(c['cost']-(1-approx*c['confidence'])),native_reconstruction_error=error,box=c['box'],pred=pred.tolist()));checks+=1
                    changed_gate+=((approx>=.2)!=(c['forward_iou']>=.2))
    result=dict(passed=native_error==0,checked=checks,max_native_cost_error=native_error,max_raw_float64_cost_error=max(e['cost_error_float64_raw_box'] for e in errors),ordinary_point2_gate_decisions_changed=changed_gate,worst=sorted(errors,key=lambda x:x['cost_error_float64_raw_box'],reverse=True)[:5],explanation='Native matching casts both boxes to float32 after detection xyxy→xywh→tlwh→xyxy. Raw float64 boxes are not the exact input used by the native IoU cost.')
    write(ROOT/'runs/bbox_singleton_audit_recovery1_20261108/precision_probe.json',result,new=True);print({k:v for k,v in result.items() if k!='worst'})


if __name__=='__main__':main()
