"""Причинные области фактического выхода, обучение рамок отдельно от источника изображений."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime,timezone
import sys,time,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
RUN=BASE/'pixel_box_dataset'

def run():
    import cv2,numpy as np
    from tracking.pixel_box_refinement import coordinates,region,metadata
    from scripts.hota_error_budget_20261027 import interventions
    from evaluation.hota_eval import VALID
    from scipy.optimize import linear_sum_assignment
    from tracking.recovery_detection_support import iou
    p=read(RUN/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp();started=time.monotonic()
    for name,h in p['sha256'].items():assert sha(ROOT/name)==h,name
    used={Path(x.strip()).name for x in (ROOT/p['detector_training_list']).read_text(encoding='utf-8-sig').splitlines()}
    complete=[]
    def status(**kw):
        if time.time()>end-5 or (RUN/'stop.request').exists():raise TimeoutError('Pixel box dataset original budget')
        write(RUN/'status.json',dict(state='running',updated=datetime.now(timezone.utc).isoformat(),**kw),required=False)
    videos=[(s,v) for s in ['train','dev'] for v in p['videos'][s]]
    for index,(split,v) in enumerate(videos):
        began=time.monotonic();folder=RUN/split/v;assert not folder.exists();folder.mkdir(parents=True)
        base=read(BASE/'admission_dataset'/split/v/'baseline.json');truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json')
        align=read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches'];assert len(base)==len(truth)==len(align)
        frame_to_ref={int(x['video_frame_index']):i for i,x in enumerate(align)};assert len(frame_to_ref)==len(align)
        cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{v}.mov'));assert cap.isOpened()
        images=[];metas=[];targets=[];known=[];records=[];past={};excluded=0
        for frame in range(max(frame_to_ref)+1):
            ok,image=cap.read();assert ok
            if frame not in frame_to_ref:continue
            ref=frame_to_ref[frame];labels=base[ref]['labels'];assert base[ref]['name']==truth[ref]['name']
            patches={x['id']:region(image,coordinates(x),p['size'],p['context']) for x in labels}
            selected=ref%p['query_stride']==p['query_offset']
            if selected and base[ref]['name'] in used:excluded+=1;selected=False
            if selected:
                gt=[x for x in truth[ref]['labels'] if x['category'] in VALID and not x.get('attributes',{}).get('crowd',False)]
                scores=np.array([[iou(coordinates(a),coordinates(b)) if a['category']==b['category'] else 0 for b in gt] for a in labels]).reshape(len(labels),len(gt))
                pairs={}
                if labels and gt:
                    rr,cc=linear_sum_assignment(-scores);pairs={int(r):int(c) for r,c in zip(rr,cc) if scores[r,c]>=.5}
                for j,label in enumerate(labels):
                    ident=label['id'];old=past.get(ident);cur=patches[ident];previous=cur if old is None else old[1]
                    b=coordinates(label);scale=np.array([max(b[2]-b[0],1),max(b[3]-b[1],1)]*2,np.float32)
                    target=np.zeros(4,np.float32) if j not in pairs else np.clip((coordinates(gt[pairs[j]])-b)/scale,-.25,.25)
                    images.append(np.stack([previous,cur]));metas.append(metadata(label,old is not None,0 if old is None else frame-old[0]));targets.append(target);known.append(j in pairs)
                    records.append(dict(reference=ref,frame=frame,name=base[ref]['name'],id=ident,past_frame=None if old is None else old[0],known=j in pairs,box=b.tolist(),target=target.tolist()))
            # История по реальным назначениям текущего запроса, после формирования входа.
            past={x['id']:(frame,patches[x['id']]) for x in labels}
            if ref%20==0:status(video=v,index=index,total=len(videos),reference=ref,records=len(records))
        cap.release();n=len(records)
        np.save(folder/'images.npy',np.asarray(images,np.uint8).reshape(n,2,p['size'],p['size'],3))
        np.savez_compressed(folder/'targets.npz',M=np.asarray(metas,np.float32).reshape(n,12),Y=np.asarray(targets,np.float32).reshape(n,4),known=np.asarray(known,bool))
        write(folder/'records.json',records)
        entry=dict(video=v,split=split,records=n,known=int(sum(known)),unknown=int(n-sum(known)),excluded_training_frames=excluded,
            seconds=time.monotonic()-began,strict_past=all(x['past_frame'] is None or x['past_frame']<x['frame'] for x in records),
            images_sha256=sha(folder/'images.npy'),targets_sha256=sha(folder/'targets.npz'),baseline_sha256=sha(BASE/'admission_dataset'/split/v/'baseline.json'))
        complete.append(entry);write(RUN/'completed_videos.json',complete)
        if index==0:
            cost=2*entry['seconds']*(len(videos)-1)+120;available=end-time.time();write(RUN/'price.json',dict(first_video_seconds=entry['seconds'],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    write(RUN/'report.json',dict(complete=True,videos=complete,records=sum(x['records'] for x in complete),known=sum(x['known'] for x in complete),seconds=time.monotonic()-started,
        exact_detector_training_images_excluded=True,unknown_excluded_from_loss=True,causal_images=True,flyvis_core_used=False,limitation=p['limitation']))
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
