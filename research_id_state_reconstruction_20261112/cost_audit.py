from common import *
import gzip
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.trackers.utils import matching
p=verify();rec=read(OUT/'reconstruction_v2.json');assert read(OUT/'audit.json')['passed'];native=ROOT/'runs/domain_matched_native_20261109';np_=read(native/'protocol.json');checked=0;entries=0;maximum=0.
for case in p['cases']:
    v=case['video']
    with gzip.open(native/'rollouts/C34'/(v+'.events.json.gz'),'rt',encoding='utf8') as f:logs=json.load(f)
    with np.load(ROOT/np_['videos'][v]['detections'],allow_pickle=False) as z:raw={k:z[k] for k in ['frame_idx','xyxy','conf','cls']}
    for event in logs['events']:
        detections=[]
        for rid in event['high_raw_ids']:
            b=raw['xyxy'][rid];xywh=np.r_[(b[:2]+b[2:])/2,b[2:]-b[:2]];detections.append(BOTrack(np.r_[xywh,0].astype(np.float64),raw['conf'][rid],raw['cls'][rid]))
        for i,row in enumerate(event['rows']):
            if row['tid'] not in case['ids']:continue
            m=np.array(row['mean']);b=np.r_[m[:2]-m[2:4]/2,m[:2]+m[2:4]/2];d=matching.iou_distance([b],[x.xyxy for x in detections]);C=matching.fuse_score(d,detections)[0];old=np.array(event['C'][i],np.float32);assert np.array_equal(C,old),(v,event['frame'],row['tid'],float(np.max(abs(C-old))));checked+=1;entries+=len(C)
save('cost_audit.json',dict(passed=True,rows=checked,entries=entries,all_costs_float32_exact=True,script_sha256=sha(__file__),scope='The installed distance implementation reproduces all original row costs for all five selected histories wherever complete matrices were archived. Missing birth rows remain reconstructed, not retroactively logged.'))
print('COSTS_EXACT',checked,entries)
