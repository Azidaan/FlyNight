"""Уточнить только немонотонную привязку по исходным пикселям, без разметки."""
from pathlib import Path
import cv2,numpy as np
from tracking.video_alignment import decode_compact_video,load_compact_images


def build_alignment(video,images,output,search_radius):
    import json
    frames,metadata=decode_compact_video(video);refs=load_compact_images(images);rows=[];ranges={}
    for index,(name,ref) in enumerate(zip(images,refs)):
        expected=round(index*(len(frames)-1)/max(len(refs)-1,1));lo=max(0,expected-search_radius);hi=min(len(frames),expected+search_radius+1)
        costs=np.array([cv2.absdiff(ref,frames[k]).mean() for k in range(lo,hi)]);order=np.argsort(costs)
        rows.append(dict(reference_index=index,reference_name=Path(name).name,video_frame_index=lo+int(order[0]),mean_absolute_error=float(costs[order[0]]),second_best_error=float(costs[order[1]])))
        ranges[index]=range(lo,hi)
    invalid=np.flatnonzero(np.diff([x['video_frame_index'] for x in rows])<=0);affected=sorted(set(int(j) for i in invalid for j in (i,i+1)));changes=[]
    if affected:
        wanted=set(k for i in affected for k in ranges[i]);native={};cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        for k in range(max(wanted)+1):
            ok,frame=cap.read();assert ok
            if k in wanted:native[k]=frame
        cap.release()
        for i in affected:
            ref=cv2.imread(str(images[i]));assert ref is not None
            candidates=list(ranges[i]);checks=[]
            for divisor in (1,2,4):
                size=(ref.shape[1]//divisor,ref.shape[0]//divisor);r=cv2.resize(ref,size,interpolation=cv2.INTER_AREA)
                costs=np.array([cv2.absdiff(r,cv2.resize(native[k],size,interpolation=cv2.INTER_AREA)).mean() for k in candidates]);order=np.argsort(costs)
                assert len(np.flatnonzero(costs==costs.min()))==1,'Native-resolution ambiguity remains'
                checks.append(dict(divisor=divisor,best=candidates[int(order[0])],mae=float(costs[order[0]]),second_best_mae=float(costs[order[1]])))
            assert len({x['best'] for x in checks})==1,'Independent resolutions disagree'
            before=rows[i]['video_frame_index'];chosen=checks[0]['best']
            rows[i].update(video_frame_index=chosen,mean_absolute_error=float(cv2.absdiff(refs[i],frames[chosen]).mean()),
                second_best_error=min(float(cv2.absdiff(refs[i],frames[k]).mean()) for k in candidates if k!=chosen),
                match_basis='native/half/quarter-resolution agreement')
            changes.append(dict(reference_index=i,old_compact_best=before,accepted=checks[0]['best'],native_checks=checks))
    assert np.all(np.diff([x['video_frame_index'] for x in rows])>0),'No verified monotonic mapping'
    result=dict(video=metadata,reference_frames=len(images),search_radius=search_radius,matches=rows,
        recovery='Only conflicting compact matches refined at native/half/quarter resolution in same candidate window',native_refinements=changes,
        object_annotations_used=False,tracker_results_used=False,coarse_cost_fields_describe_selected_matches=True)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return rows,metadata
