"""Compare repaired history scheduling with the exact pre-edit source snapshot."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import copy,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/flyvis_repairs_20261106'
import numpy as np
import torch
from tracking.history_pair_model import HistoryPairHead
from tracking.history_pair_runtime import HistoryAssociationTracker
from ultralytics.trackers.basetrack import BaseTrack
torch.set_num_threads(2);torch.set_default_device('cpu');torch.manual_seed(123)
source='tracking/history_pair_runtime.py'
with zipfile.ZipFile(OUT/'parent_baseline.zip') as z:old=z.read(source)
module=ModuleType('history_before_20261106');sys.modules[module.__name__]=module
exec(compile(old.decode('utf-8'),source,'exec'),module.__dict__)
args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=4,match_thresh=.8,fuse_score=True)
model=HistoryPairHead()
a=module.HistoryAssociationTracker(args,copy.deepcopy(model),'134',.2)
b=HistoryAssociationTracker(args,copy.deepcopy(model),'134',.2)
records=[]
for frame in range(100):
    boxes=np.array([[10+frame*.1,10,40+frame*.1,40],[15+frame*.1,10,45+frame*.1,40]],np.float32)
    if 25<=frame<36 or 65<=frame<76:boxes=boxes[:0]
    scores=np.full(len(boxes),.9,np.float32);classes=np.zeros(len(boxes),np.float32)
    context=dict(rois={i:dict(embedding=np.eye(1,256,i,dtype=np.float32)[0],image=np.zeros((3,64,64),np.uint8)) for i in range(len(boxes))},
                 camera=np.zeros(2,np.float32),epoch=0,video_size=np.array([100,100]))
    x=a.update_frame(frame,boxes,scores,classes,context)
    y=b.update_frame(frame,boxes,scores,classes,context)
    assert np.array_equal(x,y),(frame,'outputs')
    for field in ('tracked_stracks','lost_stracks','removed_stracks'):
        left,right=getattr(a,field),getattr(b,field);assert len(left)==len(right)
        for l,r in zip(left,right):
            assert l.track_id==r.track_id and np.array_equal(l.mean,r.mean) and np.array_equal(l.covariance,r.covariance)
    for ident in b.real_history:
        assert [o.frame for o in b.real_history[ident]]==[o.frame for o in a.real_history[ident]]
    records.extend([[frame,*row] for row in y])
assert a.events==b.events and len(b.events)>0
np.savez_compressed(OUT/'lifecycle_parity_outputs.npz',outputs=np.asarray(records,np.float32))
report=dict(passed=True,frames=100,nonzero_alpha=.2,events=len(b.events),
    outputs_and_all_tracker_states_exact=True,events_exact=True,
    history_keys_before=len(a.real_history),history_keys_after=len(b.real_history),
    original_source_sha256=hashlib.sha256(old).hexdigest(),
    repaired_source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
    output_sha256=hashlib.sha256((OUT/'lifecycle_parity_outputs.npz').read_bytes()).hexdigest(),
    scope='Synthetic scheduling regression with fixed nonzero neural head; no HOTA or learned quality claim',
    cuda_initialized=torch.cuda.is_initialized())
(OUT/'lifecycle_parity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
