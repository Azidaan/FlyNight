"""Execute the exact archived rollout functions with the bound new model adapter."""
import os,sys,json,hashlib,gzip,time,ast
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'runs/full_template_native_20261109';LEGACY=ROOT/'runs/identity_assignment_20261108/full_precision'
MSK=timezone(timedelta(hours=3));STOP=datetime(2026,11,9,23,54,tzinfo=MSK)
mode=sys.argv[1]
if mode!='C34':os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def cast(v):
    if isinstance(v,dict):return {k:cast(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [cast(x) for x in v]
    if hasattr(v,'tolist'):return v.tolist()
    return v
def write(name,obj,*,new=True):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    if new:
        with path.open('x',encoding='utf8') as f:json.dump(cast(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    else:
        temp=path.with_suffix(path.suffix+'.pending');temp.write_text(json.dumps(cast(obj),ensure_ascii=False,indent=2),encoding='utf8')
        for _ in range(30):
            try:temp.replace(path);return
            except PermissionError:time.sleep(.1)
        raise PermissionError(str(path))
def status(stage,**extra):write('status.json',dict(stage=stage,updated=datetime.now(MSK).isoformat(),**extra),new=False)
if not (OUT/'status.json').exists():status('starting')
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=min(14400,(STOP-datetime.now(MSK)).total_seconds()))
import psutil,numpy as np,torch
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS);torch.set_num_threads(2)
from datasets.bdd100k_loader import CLASS_NAMES
from tracking.offline_association import make_tracker_args
from tracking.spatial_identity_assignment import corrected_cost
import runtime_adapter as adapter
from runtime_adapter import Planner,LiveTracker,FeatureBank
p=read(OUT/'protocol.json');phase=read(OUT/'execution_protocol.json')
def verify(passport):
    for name,digest in passport['sources'].items():assert sha(ROOT/name)==digest,name
verify(phase)
assert read(OUT/'cache_preparation.json')['passed']
# Hook the raw source at Tracker construction, preserving the archived loop AST.
RAW_BY_VIDEO={}
for video,entry in p['videos'].items():
    with np.load(ROOT/entry['detections'],allow_pickle=False) as z:RAW_BY_VIDEO[video]={k:z[k] for k in ['frame_idx','conf','cls','xyxy']}
class BoundPlanner(Planner):
    def __init__(self,video,*args,**kwargs):super().__init__(video,*args,**kwargs);self.raw=RAW_BY_VIDEO[video]
Planner=BoundPlanner;Detections=adapter.Detections
tree=ast.parse((OLD/'native_run.py').read_text(encoding='utf8'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['scalabel','rollout']]
assert len(functions)==2
exec(compile(ast.Module(body=functions,type_ignores=[]),str(OLD/'native_run.py'),'exec'),globals())
def run(video,arm,backbone=None):
    receipt=rollout(video,arm,backbone)
    origin=OUT/'origin_receipts'/arm/(video+'.json')
    if not origin.exists():
        tracker=adapter.LAST_TRACKER
        if tracker is not None and tracker.planner.video==video:
            evidence=dict(counts=tracker.origin_counts,ambiguous=tracker.ambiguous_origins,origin_method_counts_available=True)
        else:
            # Ready rollout receipts must never be recomputed merely to recover
            # an optional method counter after a prior writer interruption.
            evidence=dict(origin_method_counts_available=False,reason='Ready rollout reused; transient in-memory counters unavailable. Primary saved assignments/history retain literal raw IDs, negative means unknown.')
        write(origin.relative_to(OUT),dict(passed=True,video=video,arm=arm,**evidence,
            metadata_only=True,negative_raw_id_means_unknown=True,association_function_unchanged=True))
    return receipt
if mode=='zero':
    receipts=[run(video,'zero_empty') for video in p['zero_witness_videos']]
    verify(phase)
    write('zero_complete.json',dict(passed=True,new_full_wrapper_witnesses=len(receipts),videos=p['zero_witness_videos'],
        all_full_raw_arrays_and_annotated_JSON_exact=True,active_empty_solver_frames=sum(r['active_empty_solver_frames'] for r in receipts),
        old_all20_both_zero_controls_reused=read(OLD/'zero_complete.json')['passed'],old_zero_receipt_sha256=sha(OLD/'zero_complete.json'),
        justification='Same actual archived rollout AST and inherited association/lifecycle. Only readout model file and source metadata adapter changed. Two predefined complete-video wrapper witnesses supplement, not repeat, the40readyall20zero controls.'))
    status('zero_complete',new_witnesses=2,old_ready_controls=40)
else:
    assert mode in p['arms'] and read(OUT/'zero_complete.json')['passed']
    if mode=='C34':
        from tracking.gpu_lock import GpuLock
        from tracking.flyvis_adapter import activate_flyvis
        from scripts.z13_target_model import build_arm
        with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
            activate_flyvis(device='cuda');_,_,bb=build_arm('C',torch,None)
            assert bb.dev.type=='cuda' and not bb.net.training and bb.steps==2 and abs(bb.dt-1/60)<1e-12
            weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
            receipts=[run(v,mode,bb) for v in p['videos']]
            assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    else:receipts=[run(v,mode) for v in p['videos']]
    verify(phase)
    write(mode+'_complete.json',dict(passed=True,arm=mode,videos=20,weights_unchanged=True,events=sum(r['events'] for r in receipts),execution_sha256=sha(OUT/'execution_protocol.json')))
    status('arm_complete',arm=mode,videos=20)
