"""Неизменный трекер и опциональная выдача lostKalman с текущей поддержкой."""
import numpy as np
from tracking.recovery_detection_support import supported_candidates


class DetectionSupportedKalmanOutput:
    def __init__(self,class_names,enabled=False,low_confidence=.1,iou_threshold=.5,max_lost_age=30):
        self.class_names=class_names;self.enabled=enabled;self.low_confidence=low_confidence
        self.iou_threshold=iou_threshold;self.max_lost_age=max_lost_age

    def _label(self,row):
        return dict(id=str(int(row[4])),category=self.class_names[int(row[6])],score=float(row[5]),
            box2d=dict(zip(('x1','y1','x2','y2'),map(float,row[:4]))))

    def output(self,tracker,base,xyxy,confidence,classes):
        base=np.asarray(base,dtype=np.float32).reshape(-1,8)
        if not self.enabled:return base.copy()
        active={int(r[4]) for r in base};extra=[]
        for track in tracker.lost_stracks:
            age=tracker.frame_id-track.end_frame
            if 1<=age<=self.max_lost_age and track.is_activated and track.track_id not in active:
                bb=np.asarray(track.xyxy,dtype=np.float32)
                if np.isfinite(bb).all() and bb[2]>bb[0] and bb[3]>bb[1]:extra.append([*bb,track.track_id,track.score,track.cls,-1])
        extra=np.asarray(extra,dtype=np.float32).reshape(-1,8)
        candidates=[self._label(row) for row in extra]
        detections=[dict(category=self.class_names[int(cls)],box2d=dict(zip(('x1','y1','x2','y2'),map(float,bb)))) for bb,score,cls in zip(xyxy,confidence,classes) if score>self.low_confidence]
        accepted,_=supported_candidates([self._label(row) for row in base],candidates,detections,self.iou_threshold)
        mapping={str(int(row[4])):row for row in extra};rows=[mapping[label['id']] for label in accepted]
        return np.concatenate([base,np.asarray(rows,dtype=np.float32).reshape(-1,8)]) if rows else base.copy()
