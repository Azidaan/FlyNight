"""Technical one-case spatial T4/T5 capture and complete decoder replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/flyvis_full_spatial_capture_probe_20261105';SRC=ROOT/'runs/flyvis_y1_latency_fp64_20261104';sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def register():
 OUT.mkdir(exist_ok=False);p=json.loads((SRC/'protocol.json').read_text(encoding='utf-8'));a=json.loads((SRC/'independent_audit.json').read_text(encoding='utf-8'));assert a['passed'];case=next(x for x in p['cases'] if x['event']=='reverse');q=SRC/'outputs'/(case['id']+'.npz');m=json.loads(q.with_suffix('.json').read_text());assert sha(q)==m['sha256'];crit='CPU FP64 single case: all full 8 T4/T5 per-column maps finite; all decoder vectors for both temporal modes and both aggregation modes reproduce saved output to absolute tolerance <=1e-5; central averages reproduce saved 8 type summaries <=1e-5; network and decoder parameter hashes unchanged.'
 doc=dict(id='flyvis_full_spatial_capture_probe_v1',registered_msk=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),case=case,reference_output_sha256=sha(q),reference_protocol_sha256=sha(SRC/'protocol.json'),source_inference_sha256=sha(ROOT/'scripts/flyvis_y1_latency_fp64_20261104.py'),self_sha256=sha(Path(__file__)),criterion=crit,scope='One fixed existing synthetic reversal sequence, precomputed eye inputs, stock FlyVis only, CPU FP64. Captures both modes for 79 frames; no training, no new real video, no HOTA, no scientific claim or correction of reversal latency.',policy='Do not modify the pretrained model, sources, dataset or earlier outputs; report both runtime and size; preserve 5GB free disk.')
 save(OUT/'protocol.json',doc);cat=ROOT/'experiments/project_history.json';d=json.loads(cat.read_text(encoding='utf-8'));assert not any(x['id']==doc['id'] for x in d['experiments']);d['experiments'].append(dict(id=doc['id'],title='FlyVis — проверка сохранения пространственной активности T4/T5',area='flyvis',state='planned',started='2026-11-05',tags=['Y1','motion-diagnostic'],question='Можно ли сохранить полные карты T4/T5, сохранив точный прежний выход?',hypothesis='Изменение записи внутренних состояний не изменит вывод модели при тех же входах.',expected='Техническая проверка одной последовательности и её стоимости.',acceptance=[dict(criterion=crit,met=None)],done='',obtained='',interpretation='',decision='',cannot_claim='',metrics=[],evidence=[dict(kind='json',ref='runs/flyvis_full_spatial_capture_probe_20261105/protocol.json',claim='Протокол технической пробы')]))
 cat.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Registered')
def run():
 import importlib,shutil,torch
 from tracking.flyvis_adapter import activate_flyvis
 from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha
 p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(Path(__file__))==p['self_sha256'];q=SRC/'outputs'/(p['case']['id']+'.npz');assert sha(q)==p['reference_output_sha256'] and sha(SRC/'protocol.json')==p['reference_protocol_sha256'];assert shutil.disk_usage(OUT).free>=5_000_000_000
 fv=activate_flyvis()
 for m in (fv,importlib.import_module('flyvis.network.initialization'),importlib.import_module('flyvis.task.decoder')):m.device=torch.device('cpu')
 torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
 from tracking.flyvis_motion import FlyVisMotionEstimator
 model=FlyVisMotionEstimator(mode='full',source_fps=30,flow_gain=1,temporal_resampling='hold',subtract_initial_baseline=False);model.network.double();model.decoder.double();model.eye.conv.double();net,dec=model.network,model.decoder;net.clamp();frozen=parameter_sha(net,dec);ref=np.load(q);hexes=ref['hexes'];mask=np.load(SRC/'geometry.npz')['mask'];names=('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d');idx=[np.asarray(net.stimulus.layer_index[n]) for n in names];assert all(len(x)==721 for x in idx);maps=np.empty((2,79,8,721),np.float64);flow=np.empty((2,2,79,721,2),np.float64);start=time.perf_counter()
 with torch.inference_mode():
  first=torch.from_numpy(hexes[0])[None,None];state=net.fade_in_state(1.0,model.dt,first.repeat(2,1,1));initial=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1);flow[:,:,0]=initial[:,None]*np.array([512/169,-512/169]);maps[:,0]=np.stack([state.nodes.activity[:,i].cpu().numpy() for i in idx],axis=1);last=torch.from_numpy(hexes[0])[None,None,None]
  for j in range(1,79):
   current=torch.from_numpy(hexes[j])[None,None,None];hold=current.expand(1,2,1,721);alpha=torch.tensor([.5,1.]).view(1,2,1,1);linear=last*(1-alpha)+current*alpha;states=net.simulate(torch.cat([hold,linear],dim=0),model.dt,initial_state=state,as_states=True);act=torch.stack([s.nodes.activity for s in states],dim=1);decoded=dec(act);field=torch.stack([decoded.mean(1),decoded[:,-1]],dim=1);flow[:,:,j]=field.cpu().numpy().transpose(0,1,3,2)*np.array([512/169,-512/169]);state=states[-1];maps[:,j]=np.stack([state.nodes.activity[:,i].cpu().numpy() for i in idx],axis=1);last=current
 central=maps[:,:,:,mask].mean(axis=-1);saved=ref['activity'][:,:,1:];fd=float(np.max(np.abs(flow-ref['flow'])));ad=float(np.max(np.abs(central-saved)));same=parameter_sha(net,dec)==frozen;finite=bool(np.isfinite(maps).all());passed=bool(finite and same and fd<=1e-5 and ad<=1e-5);np.savez_compressed(OUT/'spatial_maps.npz',maps=maps,flow=flow,centers=np.load(SRC/'geometry.npz')['centers']);save(OUT/'report.json',dict(passed=passed,case=p['case'],protocol_sha256=sha(OUT/'protocol.json'),reference_output_sha256=sha(q),spatial_maps_sha256=sha(OUT/'spatial_maps.npz'),maps_shape=list(maps.shape),flow_max_abs_diff=fd,central_activity_max_abs_diff=ad,parameter_sha_before_after=frozen,parameters_unchanged=same,cuda_initialized=torch.cuda.is_initialized(),elapsed_seconds=time.perf_counter()-start,bytes=(OUT/'spatial_maps.npz').stat().st_size,scope=p['scope']));print(json.dumps(dict(passed=passed,flow_diff=fd,activity_diff=ad,seconds=time.perf_counter()-start)));assert passed
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['register','run']);v=a.parse_args();register() if v.stage=='register' else run()
