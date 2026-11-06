"""Independent replay of root motion/birth repairs and bounded counterexamples."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
from tracking.birth_policy import BirthHead,QBirthByteTrack,DeferredBirthByteTrack
from tracking.decision_log import IndexedDetections
from tracking.motion_source_bytetrack import make_deferred_cascade
from tracking.motion_augmented_bytetrack import MotionAugmentedByteTrack
from tracking.flyvis_tracker import MotionField
from tracking.offline_association import make_tracker_args
from ultralytics.trackers.basetrack import TrackState,BaseTrack
from ultralytics.trackers.byte_tracker import STrack,BYTETracker
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources=['tracking/decision_log.py','tracking/birth_policy.py','tracking/motion_source_bytetrack.py','tracking/motion_augmented_bytetrack.py']
hashes={name:sha(ROOT/name) for name in sources}
snap=ROOT/'runs/flyvis_repairs_20261106/motion_birth_before'
def original(filename):
    name='review_original_'+Path(filename).stem
    spec=importlib.util.spec_from_file_location(name,snap/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
old_motion=original('motion_augmented_bytetrack.py').MotionAugmentedByteTrack
old_birth=original('birth_policy.py')
head=BirthHead(('conf',),np.zeros(1),np.ones(1),np.zeros(1),0.)

def context(t,gray=None):
    t.set_frame_context(gray=np.full((160,160),128,np.uint8) if gray is None else gray,
        prev_propagated=np.zeros((0,4)),frame_size=(160,160))

def det(x=30,score=.9):
    return IndexedDetections(np.array([[x,30,x+25,60]],np.float32),np.array([score]),np.array([2]))

def state(t):
    return dict(frame=t.frame_id,tracks=[dict(id=s.track_id,state=s.state,mean=s.mean.tolist(),
        covariance=s.covariance.tolist()) for s in t.tracked_stracks+t.lost_stracks],
        events=copy.deepcopy(t.events),predictions=copy.deepcopy(t.predictions),
        births=copy.deepcopy(t.birth_log),stage=t.stage)

factories=[lambda:QBirthByteTrack(make_tracker_args(),head=head,tau=.4),
    lambda:DeferredBirthByteTrack(make_tracker_args(),head=head,tau=.6,confirm_hits=3),
    lambda:make_deferred_cascade(make_tracker_args(),buffer=.1,cascade_buffer=.2,head=head,tau=.6,confirm_hits=3)]
resets=[]
for factory in factories:
    tracker=factory()
    for i in range(4):context(tracker);tracker.update(det(30+i))
    tracker._context=None;before=state(tracker)
    try:tracker.update(det(35))
    except RuntimeError:pass
    else:raise AssertionError('Missing context was accepted')
    assert before==state(tracker)
    tracker.reset()
    assert tracker._context is None and not any((tracker.birth_log,tracker.events,tracker.predictions)) and tracker.stage==''
    if hasattr(tracker,'_previous_xywh'):
        assert not tracker._previous_xywh and not tracker._p_bad
        assert tracker.stats==dict(switched=0,tracks_seen=0)
    context(tracker);actual=tracker.update(det())
    fresh=factory();context(fresh);expected=fresh.update(det())
    assert np.array_equal(actual,expected)
    resets.append(dict(class_name=type(tracker).__name__,mro=[c.__name__ for c in type(tracker).mro()],
        missing_context_preserves_existing_state=True,reset_equals_fresh=True))

def valid_replay(cls,kwargs):
    tracker=cls(make_tracker_args(),head=head,**kwargs);out=[]
    for i in range(6):
        context(tracker);out.append(tracker.update(det(30+i,.9 if i!=3 else .2)))
    return out
normal_birth={}
for old,new,kwargs in ((old_birth.QBirthByteTrack,QBirthByteTrack,dict(tau=.4)),
                      (old_birth.DeferredBirthByteTrack,DeferredBirthByteTrack,dict(tau=.6,confirm_hits=3))):
    a,b=valid_replay(old,kwargs),valid_replay(new,kwargs)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    normal_birth[new.__name__]=True

field=MotionField(np.array([[80,80],[100,100],[120,120]],np.float32),np.array([[4,2],[5,3],[6,4]],np.float32))
def track(lost=False):
    t=STrack(np.array([100,100,40,40,0],np.float32),.9,2)
    t.activate(KalmanFilterXYAH(),frame_id=1);t.mean[4:6]=[3.,-2.]
    if lost:t.mark_lost()
    return t
normal_motion=0
for confidence,lost,apply,agreement,ungated in itertools.product((False,True),(False,True),('all','lost'),(None,.5),(False,True)):
    kwargs=dict(blend=.7,use_confidence=confidence,apply_to=apply,agreement_scale=agreement,ungated_lost=ungated)
    old,new=old_motion(make_tracker_args(),**kwargs),MotionAugmentedByteTrack(make_tracker_args(),**kwargs)
    old.set_motion_field(field);new.set_motion_field(field)
    one=track(lost);two=copy.deepcopy(one)
    old.multi_predict([one]);new.multi_predict([two])
    assert np.array_equal(one.mean,two.mean) and np.array_equal(one.covariance,two.covariance)
    assert old.motion_audit==new.motion_audit
    normal_motion+=1
empty=[]
for confidence in (False,True):
    t=track();expected=copy.deepcopy(t);old=copy.deepcopy(t)
    standard=BYTETracker(make_tracker_args());standard.multi_predict([expected])
    new=MotionAugmentedByteTrack(make_tracker_args(),blend=1.,use_confidence=confidence)
    legacy=old_motion(make_tracker_args(),blend=1.,use_confidence=confidence)
    new.multi_predict([t]);legacy.multi_predict([old])
    assert np.array_equal(t.mean,expected.mean)
    empty.append(dict(confidence=confidence,new_equals_kalman=True,
        legacy_equals_kalman=bool(np.array_equal(old.mean,expected.mean)),new_weight=new.motion_audit[-1]['weight']))

# Explicit limits: malformed present context still fails late; None-only guard
# makes no promise of validating every field or consuming the context once.
t=factories[0]();context(t);t.update(det())
t.set_frame_context(gray=None,prev_propagated=np.zeros((0,4)),frame_size=(160,160))
before_frame=t.frame_id
try:t.update(det(100))
except Exception as exc:malformed=dict(error=type(exc).__name__,before_frame=before_frame,after_frame=t.frame_id)
else:malformed=dict(error=None,before_frame=before_frame,after_frame=t.frame_id)
t=factories[0]();context(t);t.update(det());t.update(det(31))
stale_context_accepted=t.frame_id==2
saved_count=BaseTrack._count
try:
    BaseTrack._count=987;t.reset();reset_global_count=BaseTrack._count
finally:BaseTrack._count=saved_count

assert all(sha(ROOT/name)==digest for name,digest in hashes.items())
report=dict(passed_targeted_repairs=True,resets=resets,valid_birth_old_new_exact=normal_birth,
    nonempty_motion_old_new_exact_cases=normal_motion,empty_motion=empty,
    limits=dict(malformed_present_context=malformed,stale_context_accepted=stale_context_accepted,
        inherited_reset_global_count_after_setting987=reset_global_count,
        scope='These limits predate the fixes; no claim of fresh context validation or global counter isolation.'),
    source_sha256=hashes,script_sha256=sha(__file__))
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
