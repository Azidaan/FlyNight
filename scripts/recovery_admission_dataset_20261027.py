"""Подготовить отдельные данные допуска; разметка используется только для меток."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import collections,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import read,write,sha
RUN=ROOT/'runs/recovery_research_20261027/admission_dataset'

def run():
    import numpy as np,cv2
    from tracking.xywh_bytetrack import XYWHByteTrack
    from tracking.offline_association import _DetSlice
    from tracking.detection_supported_kalman import DetectionSupportedKalmanOutput
    from tracking.recovery_admission_oracle import PastIdentityAdmissionOracle
    from tracking.recovery_admission_features import CausalAdmissionMemory,FEATURE_NAMES
    from scripts.run_bdd_video_rate_experiment import load_detection_cache,detections_for_frame,scalabel_frame,CLASS_NAMES
    from evaluation.hota_eval import VALID
    protocol=read(RUN/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();start=time.monotonic()
    for n,h in protocol['sha256'].items():assert sha(ROOT/n)==h,n
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Dataset stage deadline')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    manifest=[];videos=[(split,v) for split in ['train','dev'] for v in protocol['videos'][split]]
    for index,(split,v) in enumerate(videos):
        began=time.monotonic();out=RUN/split/v;assert not out.exists();out.mkdir(parents=True)
        root=ROOT/'runs/train_cache'/v
        dets=load_detection_cache(root/'detections/raw_video.npz');align=read(root/'alignment.json')['matches']
        truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json');assert len(truth)==len(align)
        anchors={int(a['video_frame_index']):i for i,a in enumerate(align)}
        assert len(anchors)==len(align) and all(b>a for a,b in zip(anchors,list(anchors)[1:])),v
        meta=read(root/'detections/raw_video.json')['identity'];assert meta['weights_sha256']==protocol['detector_sha256'] and meta['imgsz']==960 and meta['conf']==.01
        tracker=XYWHByteTrack(SimpleNamespace(**protocol['config']));tracker.reset_id()
        branch=DetectionSupportedKalmanOutput(CLASS_NAMES,enabled=True);memory=CausalAdmissionMemory();oracle=PastIdentityAdmissionOracle()
        cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{v}.mov'));assert cap.isOpened()
        xs=[];ys=[];records=[];counts=collections.Counter();pairs={};baseline_rows=[];supported_rows=[]
        for frame in range(int(dets['frame_count'])):
            if frame%100==0:status(video=v,split=split,index=index,total=len(videos),frame=frame,candidates=len(xs))
            ok,image=cap.read();assert ok,(v,frame)
            xyxy,conf,classes=detections_for_frame(dets,frame)
            base=np.asarray(tracker.update(_DetSlice(xyxy,conf,classes)),dtype=np.float32).reshape(-1,8)
            if frame in anchors:
                ref=anchors[frame];extended=branch.output(tracker,base,xyxy,conf,classes)
                assert np.array_equal(extended[:len(base)],base)
                labels=[branch._label(row) for row in base];candidates=[branch._label(row) for row in extended[len(base):]]
                tracks={t.track_id:t for t in tracker.lost_stracks};features=[]
                # Сначала признаки из реальных наблюдений, затем отдельно метки с разметкой.
                for c in candidates:features.append(memory.describe(frame,c,tracks[int(c['id'])],labels,xyxy,conf,classes,image))
                known=[(c['category'],str(c['id'])) in oracle.history for c in candidates]
                accepted=oracle.step(labels,candidates,[g for g in truth[ref]['labels'] if g['category'] in VALID]);accepted_ids={c['id'] for c in accepted}
                for c,is_known,(x,current,past,past_frame) in zip(candidates,known,features):
                    target=-1 if not is_known else int(c['id'] in accepted_ids)
                    ident=str(c['id']);key=str(frame)+'_'+ident
                    current_name=key+'_current.jpg';past_name=str(past_frame)+'_'+ident+'_past.jpg'
                    assert cv2.imwrite(str(out/current_name),current,[cv2.IMWRITE_JPEG_QUALITY,95])
                    if past_name not in pairs:
                        assert cv2.imwrite(str(out/past_name),past,[cv2.IMWRITE_JPEG_QUALITY,95]);pairs[past_name]=True
                    xs.append(x);ys.append(target);counts[str(target)]+=1
                    records.append(dict(frame=frame,reference=ref,id=ident,past_frame=past_frame,current=current_name,past=past_name,target=target))
                baseline_rows.append(scalabel_frame(v,truth[ref],base));supported_rows.append(scalabel_frame(v,truth[ref],extended))
            memory.remember(frame,base,image,{t.track_id for t in tracker.tracked_stracks+tracker.lost_stracks})
        cap.release()
        x=np.asarray(xs,np.float32).reshape(-1,len(FEATURE_NAMES));y=np.asarray(ys,np.int8)
        np.savez_compressed(out/'features.npz',X=x,y=y)
        write(out/'records.json',records);write(out/'baseline.json',baseline_rows);write(out/'supported.json',supported_rows)
        entry=dict(video=v,split=split,candidates=len(xs),labels=dict(counts),strict_past=all(a['past_frame']<a['frame'] for a in records),
            seconds=time.monotonic()-began,source_cache_sha256=sha(root/'detections/raw_video.npz'),features_sha256=sha(out/'features.npz'),records_sha256=sha(out/'records.json'))
        manifest.append(entry);write(RUN/'completed_videos.json',manifest)
        if index==0:
            cost=2*entry['seconds']*(len(videos)-1)+120;available=end-time.time()
            write(RUN/'price.json',dict(first_video_seconds=entry['seconds'],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available))
            assert cost<available,'Dataset price gate failed; do not extend budget'
    write(RUN/'report.json',dict(complete=True,feature_names=FEATURE_NAMES,videos=manifest,seconds=time.monotonic()-start,
        unknown_label=-1,unknown_not_negative=True,features_do_not_receive_truth=True,detector_seen_training_videos=True,
        heldout_val_not_used_for_fitting=True,flyvis_core_used=False,limitation=protocol['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
