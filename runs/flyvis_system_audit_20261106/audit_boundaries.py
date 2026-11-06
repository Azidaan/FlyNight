"""Small deterministic contract reproductions; no real model training or fixes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,hashlib
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
torch.set_num_threads(2);torch.set_default_device('cpu')
from tracking.flyvis_motion import FlyVisMotionEstimator,TileLayout,fixed_tile_bounds
from tracking.b1_tracker_runtime import B1Tracker,CausalMasa
from tracking.z13_readout import HexGeometry
from tracking.offline_association import _DetSlice
report={}
# The inverse geometric transform of the exact same square-eye displacement
# must scale its x component with crop width. A synthetic decoder tensor
# isolates the conversion from network prediction errors.
model=FlyVisMotionEstimator.__new__(FlyVisMotionEstimator)
model.eye_height=model.eye_width=391
model._local_centers_yx=np.array([[0,0]],np.float32)
model.flow_gain=1.;model.kernel_area=169
flow=torch.tensor([[[169/391],[0.]]],dtype=torch.float32)
rows=[]
for width,height in [(720,720),(1280,720),(492,400)]:
    model._bounds=[(0,0,width,height)]
    out=model._points_and_vectors(flow).vectors_xy[0]
    rows.append(dict(width=width,height=height,decoded_one_eye_pixel=flow.tolist(),actual_x=float(out[0]),inverse_resize_x=width/391,ratio=float(out[0]/(width/391))))
report['flow_inverse_resize']=rows

# Observe real tracker calls with lightweight stand-ins only for expensive
# rendering, appearance extraction and residual inference. Tracker is unchanged.
class Bank:
    eye_size=391
    def _render(self,frame,crops):return torch.full((len(crops),1,4),float(frame[0,0]))
class Server(CausalMasa):
    def embed(self,frame,boxes):return np.zeros((len(boxes),256),np.float32)
class Base:
    def probability(self,geo,region,coarse,snapped,old,gap):
        return np.full(len(geo),.5),geo,np.ones(len(geo),bool),{},np.zeros((len(geo),1))
    def decision(self,p,geo,mask):return None,np.full(len(geo),.5)
class Engine:
    def __init__(self):self.calls=[];self.last_pooled={}
    def start(self,h):
        self.calls.append({'start_frames':[x['frame'] for x in h]})
        return {'last_frame':h[-1]['frame']}
    def advance(self,eye,current):
        frame=int(current[0]);self.calls.append({'advance_from':eye['last_frame'],'advance_to':frame,'fixed_backbone_steps':2,'fixed_backbone_dt':1/60})
        eye['last_frame']=frame
    def correction(self,eye,masks,ref,p,zero):return p[:len(masks)],np.zeros(len(masks))
bank=Bank();engine=Engine();server=Server(None)
geom=HexGeometry(np.array([[160,160],[230,160],[160,230],[230,230]],np.float32),391)
args=SimpleNamespace(track_high_thresh=.25,track_low_thresh=.1,new_track_thresh=.25,track_buffer=30,match_thresh=.8,fuse_score=True)
tracker=B1Tracker(args,bank,geom,server,Base(),engine)
for frame in range(8):
    boxes=np.array([[30,30,70,70]],np.float32) if frame<4 or frame==7 else np.zeros((0,4),np.float32)
    conf=np.full(len(boxes),.9,np.float32);cls=np.zeros(len(boxes),np.float32)
    gray=np.full((100,100),frame,np.uint8)
    server.start(frame,gray,boxes,conf);tracker.begin(frame,gray)
    tracker.update(_DetSlice(boxes,conf,cls));tracker.after()
report['b1_timestamp_trace']={'calls':engine.calls,'events':[{k:e[k] for k in ('frame','reference_frame','track')} for e in tracker.events], 'scope':'Actual B1Tracker scheduling; neural/appearance stand-ins. Timing constants from ResidualEngine and FlyBackbone, not measured neural outputs.'}

# Saved checkpoints: inspect values without constructing or changing a network.
def state_summary(state):
    return {k:dict(shape=list(v.shape),finite=bool(torch.isfinite(v).all()),negative=int((v<0).sum()),min=float(v.min()),max=float(v.max())) for k,v in state.items() if k in ('nodes_bias','nodes_time_const','edges_syn_strength')}
report['all_stock_checkpoints']=[]
for p in sorted((ROOT/'runs/flyvis/results/flow/0000').glob('*/best_chkpt')):
    ck=torch.load(p,map_location='cpu',weights_only=False)
    report['all_stock_checkpoints'].append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),has_network='network' in ck,has_flow='flow' in ck.get('decoder',{}),summary=state_summary(ck.get('network',{}))))
report['cuda_initialized']=torch.cuda.is_initialized()
report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tracking/b1_tracker_runtime.py',ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_motion.py']}
(OUT/'boundary_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='all_stock_checkpoints'},ensure_ascii=False,indent=2))
print('stock checkpoints',len(report['all_stock_checkpoints']))
