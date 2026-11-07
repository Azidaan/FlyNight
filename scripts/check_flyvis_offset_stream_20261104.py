"""Независимый потоковый повтор одной последовательности каждой версии."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,importlib,json,sys
import cv2,numpy as np,psutil,torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_offset_recovery1_20261104';sys.path.insert(0,str(ROOT))
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
from tracking.flyvis_adapter import activate_flyvis
fv=activate_flyvis()
for m in [fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')]:m.device=torch.device('cpu')
torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
from tracking.flyvis_motion import FlyVisMotionEstimator
arm=sys.argv[1];assert arm in ('stock','fast');old=FlyVisMotionEstimator.__init__
def init(self,*args,**kwargs):
 old(self,*args,**kwargs);self.network.double();self.decoder.double();self.eye.conv.double()
 if arm=='fast':
  with torch.no_grad():self.network.nodes_time_const.mul_(.5)
def render(self,frame):
 gray=np.asarray(frame);h,w=gray.shape;self._prepare_layout(w,h)
 a=[cv2.resize(gray[y1:y2,x1:x2],(self.eye_width,self.eye_height),interpolation=cv2.INTER_AREA).astype(np.float64)/255 for x1,y1,x2,y2 in self._bounds]
 return self.eye(torch.from_numpy(np.stack(a))[:,None].to('cpu'))
FlyVisMotionEstimator.__init__=init;FlyVisMotionEstimator._render=render;FlyVisMotionEstimator._synchronize=lambda self:None
p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));case=p['cases'][1];input_path=OUT/'inputs'/(case['id']+'.npz');output_path=OUT/arm/'outputs'/(case['id']+'.npz')
mark=json.loads(output_path.with_suffix('.json').read_text());assert hashlib.sha256(output_path.read_bytes()).hexdigest()==mark['sha256']
inp=np.load(input_path)['frames'];output=np.load(output_path)['flow'];model=FlyVisMotionEstimator(mode='full',source_fps=30,flow_gain=1,temporal_resampling='hold',subtract_initial_baseline=False)
diff=[]
for j,image in enumerate(inp):
 model.observe(image)
 if j:
  d=model.last_decoded.cpu().numpy()[0];both=np.stack([d.mean(0),d[-1]],axis=0).transpose(0,2,1)*np.array([512/169,-512/169],np.float64)
  diff.append(float(np.abs(both-output[0,:,j]).max()))
record=dict(passed=max(diff)<=1e-5,arm=arm,case=case['id'],frames_checked=len(diff),max_abs_diff=max(diff),tolerance=1e-5,
 output_sha256=mark['sha256'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cuda_initialized=torch.cuda.is_initialized(),
 scope='Один случай на CPU через потоковый интерфейс; не полное повторение всех последовательностей.')
(OUT/(arm+'_stream_reference.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert record['passed'] and not record['cuda_initialized'];print(json.dumps(record,ensure_ascii=False))
