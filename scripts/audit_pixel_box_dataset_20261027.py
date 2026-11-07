"""Сверить разделение, все метки геометрии и выборочные реальные пиксели до обучения."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now
RUN=BASE/'pixel_box_audit'

def run():
    import cv2,numpy as np
    from scipy.optimize import linear_sum_assignment
    from tracking.pixel_box_refinement import coordinates,region,metadata
    from evaluation.hota_eval import VALID
    from scripts.oracle_detection_ladder import iou_mat
    data=BASE/'pixel_box_dataset';p=read(data/'protocol.json');report=read(data/'report.json');done=read(data/'controller_result.json')
    assert report['complete'] and done['exit_code']==0 and done['all_owned_ended']
    assert not set(p['videos']['train'])&set(p['videos']['dev'])
    used={Path(x.strip()).name for x in (ROOT/p['detector_training_list']).read_text(encoding='utf-8-sig').splitlines()}
    count=0;pixel_checks=0;began=time.monotonic()
    for entry in report['videos']:
        v=entry['video'];split=entry['split'];folder=data/split/v
        assert sha(folder/'images.npy')==entry['images_sha256'] and sha(folder/'targets.npz')==entry['targets_sha256']
        a=np.load(folder/'images.npy',mmap_mode='r');z=np.load(folder/'targets.npz');records=read(folder/'records.json')
        assert a.shape==(len(records),2,96,96,3) and a.dtype==np.uint8
        assert z['M'].shape==(len(records),12) and z['Y'].shape==(len(records),4) and np.isfinite(z['M']).all() and np.isfinite(z['Y']).all()
        assert np.max(np.abs(z['Y']),initial=0)<=.250001
        baseline=read(BASE/'admission_dataset'/split/v/'baseline.json');truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{v}.json');align=read(ROOT/f'runs/train_cache/{v}/alignment.json')['matches']
        by={};past={};k=0
        for ref,row in enumerate(baseline):
            frame=int(align[ref]['video_frame_index']);labels=row['labels']
            if ref%p['query_stride']==p['query_offset'] and row['name'] not in used:
                gt=[g for g in truth[ref]['labels'] if g['category'] in VALID and not g.get('attributes',{}).get('crowd',False)]
                scores=iou_mat(np.array([coordinates(x) for x in labels]).reshape(-1,4),np.array([coordinates(x) for x in gt]).reshape(-1,4))
                if labels and gt:
                    scores*=np.array([[a['category']==b['category'] for b in gt] for a in labels]);rr,cc=linear_sum_assignment(-scores)
                    pairs={int(r):int(c) for r,c in zip(rr,cc) if scores[r,c]>=.5}
                else:pairs={}
                for j,label in enumerate(labels):
                    r=records[k];assert r['name'] not in used and r['reference']==ref and r['id']==label['id'] and r['frame']==frame
                    previous=past.get(label['id']);assert r['past_frame']==(None if previous is None else previous[0])
                    assert previous is None or previous[0]<frame
                    b=coordinates(label);scale=np.array([max(b[2]-b[0],1),max(b[3]-b[1],1)]*2,np.float32)
                    y=np.zeros(4,np.float32) if j not in pairs else np.clip((coordinates(gt[pairs[j]])-b)/scale,-.25,.25)
                    assert np.array_equal(z['Y'][k],y) and bool(z['known'][k])==(j in pairs)
                    assert np.array_equal(z['M'][k],metadata(label,previous is not None,0 if previous is None else frame-previous[0]))
                    by[k]=(label,previous);k+=1
            past={x['id']:(frame,x) for x in labels}
        assert k==len(records);count+=k
        if records:
            cap=cv2.VideoCapture(str(ROOT/f'data/bdd100k/videos/train/{v}.mov'))
            for i in sorted({0,len(records)-1}):
                label,previous=by[i];r=records[i]
                for side,fr,lab in [(1,r['frame'],label),(0,r['frame'] if previous is None else previous[0],label if previous is None else previous[1])]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,fr);ok,image=cap.read();assert ok
                    expected=region(image,coordinates(lab),p['size'],p['context']);assert np.array_equal(a[i,side],expected),(v,i,side)
                    pixel_checks+=1
            cap.release()
        write(RUN/'status.json',dict(state='running',updated=now(),video=v,checked_records=count),required=False)
    assert count==report['records']
    write(data/'audit.json',dict(passed=True,videos=len(report['videos']),records=count,pixel_checks=pixel_checks,all_labels_recomputed=True,
        strict_past=True,train_dev_disjoint=True,exact_training_frames_excluded=True,finite_bounded_targets=True,hashes_verified=True,seconds=time.monotonic()-began))
    write(RUN/'report.json',read(data/'audit.json'));write(RUN/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':run()
