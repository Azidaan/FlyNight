"""Unchanged native association/readout with new heads and explicit origin metadata."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'runs/full_template_native_20261109'
sys.path.insert(0,str(OLD))
import common as archived_common
assert Path(archived_common.__file__).resolve()==OLD/'common.py'
# The archived runtime imports these destination helpers; its mathematical code
# remains the fingerprinted original. New writes must never enter the old run.
archived_common.OUT=OUT
import native_runtime as archived
assert archived.OUT==OUT
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
Detections=archived.Detections


class FeatureBank(archived.FeatureBank):
    def __init__(self,video,arm,backbone=None):
        super().__init__(video,arm,backbone)
        # Same closed G_PERM receipt extension already verified in the first pilot.
        source=archived.LEGACY/'rollouts/G_PERM'/(video+'.receipt.json')
        if source.exists():
            for name,digest in read(source).get('feature_files',{}).items():
                if name in self.legacy_sources:assert self.legacy_sources[name]==digest,name
                self.legacy_sources[name]=digest
            self.files[source.relative_to(ROOT).as_posix()]=sha(source)


class Planner(archived.Planner):
    def __init__(self,video,arm,models=None,bank=None):
        self.outer_arm=arm
        super().__init__(video,'G7' if arm=='G_DOMAIN' else arm,models,bank)
        self.raw=None;self.raw_ids=None

    def logit(self,arm,geometry,channels=None):
        return 1. if self.outer_arm=='G_DOMAIN' else super().logit(arm,geometry,channels)

    def origin(self,track):
        assert self.raw is not None and self.raw_ids is not None
        frame=self.frame;raw=self.raw
        old=getattr(track,'export_raw_id',-1)
        if old is not None and 0<=int(old)<len(raw['frame_idx']):
            old=int(old)
            if (int(raw['frame_idx'][old])==frame and int(raw['cls'][old])==int(track.cls)
                    and np.float32(raw['conf'][old])==np.float32(track.score)):
                return old,'native_current_detection_pointer'
        ids=self.raw_ids[(raw['conf'][self.raw_ids].astype(np.float32)==np.float32(track.score))&(raw['cls'][self.raw_ids]==int(track.cls))]
        return (int(ids[0]),'unique_exact_frame_score_class') if len(ids)==1 else (-1,'ambiguous_raw_origin')

    def after(self,tracker):
        super().after(tracker)
        public={int(t.track_id) for t in tracker.tracked_stracks if t.is_activated and t.frame_id==tracker.frame_id}
        for t in tracker.tracked_stracks:
            if t.frame_id!=tracker.frame_id:continue
            tid=int(t.track_id);rid,method=self.origin(t)
            self.histories[tid][-1].update(raw_id=None if rid<0 else rid,raw_origin_method=method,public=tid in public)


LAST_TRACKER=None
class LiveTracker(archived.LiveTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        global LAST_TRACKER
        LAST_TRACKER=self;self.origin_counts={};self.ambiguous_origins=[]

    def update(self,results,*args,**kwargs):
        # Input IDs are source metadata only, not extra association features.
        self.planner.raw_ids=np.asarray(results.raw_ids)
        output=super().update(results,*args,**kwargs)
        for t in self.tracked_stracks:
            if t.frame_id!=self.frame_id:continue
            rid,method=self.planner.origin(t)
            previous=getattr(t,'export_raw_id',-1)
            if rid>=0:
                assert int(self.planner.raw['frame_idx'][rid])==self.planner.frame
                assert int(self.planner.raw['cls'][rid])==int(t.cls)
                assert np.float32(self.planner.raw['conf'][rid])==np.float32(t.score)
            else:self.ambiguous_origins.append(dict(frame=self.planner.frame,tid=int(t.track_id)))
            self.origin_counts[method]=self.origin_counts.get(method,0)+1
            if rid>=0 and previous!=rid:self.origin_counts['stale_export_pointer_repaired']=self.origin_counts.get('stale_export_pointer_repaired',0)+1
            t.export_raw_id=rid;self.assigned[int(t.track_id)]=rid
        return output


# The first/second associations and lifecycle must be literally inherited.
assert LiveTracker._first_association is archived.LiveTracker._first_association
assert LiveTracker._second_association is archived.LiveTracker._second_association
