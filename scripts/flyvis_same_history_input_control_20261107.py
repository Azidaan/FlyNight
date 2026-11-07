"""Prespecified same-state control: change only the final input, not prior retinal history."""
import os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from scripts.flyvis_input_separation_20261107 import read,sha,write,GUARD
BASE=Path("runs/flyvis_separate_inputs_20261107");OUT=BASE/"matched_history"
def prepare():
 assert not OUT.exists();OUT.mkdir()
 p=read(BASE/"protocol.json")
 p.update(control_scope="Additional prespecified matched-state control; does not replace or rescue the original primary. All original past RETINAL inputs and the full penultimate state are IDENTICAL in every condition. Only current input is common/union/separate; same original interpolation into the current frame. This control can introduce a masking-onset edge; whole-prefix intervention has the alternative risk of removing the target from earlier inputs.",
          source_parent_protocol_sha256=sha(BASE/"protocol.json"),secondary_only=True)
 p["source_sha256"]={**p["source_sha256"],str(Path(__file__).relative_to(ROOT).as_posix()):sha(Path(__file__))}
 write(OUT/"protocol.json",p);(OUT/"inputs").mkdir();(OUT/"raw").mkdir()
 print("Matched retinal-history control registered before GPU outcomes")
def run(obs):
 import numpy as np,torch,shutil
 from scripts.z13_target_model import build_arm
 from tracking.gpu_lock import GpuLock
 p=read(OUT/"protocol.json");ins=read(BASE/"inputs_completed.json")
 assert read(BASE/"recovery1/controller.json")["passed"]
 for fn,h in p["source_sha256"].items():assert sha(fn)==h,fn
 # Reuse exactly the completed inputs via hardlinks; never edit them.
 for c in p["cases"]:
  src=BASE/"inputs"/f"{c['uid']}.npz";dst=OUT/"inputs"/src.name
  assert sha(src)==ins["sha256"][src.name]
  if not dst.exists():os.link(src,dst)
 write(OUT/"inputs_completed.json",{**ins,"protocol_sha256":sha(OUT/"protocol.json")})
 norm=read("runs/z13_target_center_dev_20261018/C/normalization.json");mu=np.asarray(norm["mean"])[4:12];sd=np.asarray(norm["sd"])[4:12]
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);finished={};start=time.perf_counter()
 with GpuLock("flyvis_fovea_gpu"),torch.no_grad():
  _,_,bb=build_arm("C",torch,None);params=bb.net._param_api()
  for i,c in enumerate(p["cases"]):
   assert shutil.disk_usage(ROOT).free>=5*1024**3
   uid=c["uid"];dest=OUT/"raw"/f"{uid}.npz";before=time.perf_counter()
   assert not dest.exists()
   with np.load(OUT/"inputs"/dest.name,allow_pickle=False) as z:
    seq=z["inputs"].astype("float32");m=z["masks"];ref=z["reference"].astype("float64");w=z["weights"]
   hx=torch.from_numpy(seq[0])[None].to(bb.dev);state=bb.init(hx[:,:1])
   for j in range(1,hx.shape[1]-1):state,_=bb.step(state,hx[:,j-1],hx[:,j])
   original=state.nodes.activity.clone();cl=bb.net._initial_state(params,5);cl.nodes.activity=original.expand(5,-1).clone();cl=bb.net._state_api(cl)
   assert torch.equal(cl.nodes.activity,original.expand(5,-1))
   cur=torch.from_numpy(np.stack([seq[0,-1],seq[0,-1].copy(),seq[1,-1],seq[2,-1],seq[3,-1]])).to(bb.dev)
   _,f=bb.step(cl,hx[:,-2].expand(5,-1),cur)
   assert torch.equal(state.nodes.activity,original)
   maps=((f[:,4:12].cpu().numpy().astype("float64")-mu[None,:,None])/sd[None,:,None]).astype("float32")
   scores=np.stack([maps[:,:,mask].astype("float64").mean(2)@w for mask in m],1)
   refs=np.array([ref[:,mask].mean(1)@w for mask in m]);am=float(np.abs(maps[0]-ref).max());im=float(np.abs(maps[1]-maps[0]).max())
   assert np.isfinite(maps).all() and am<=p["gates"]["archive_normalized_map_max"] and np.abs(scores[0]-refs).max()<=p["gates"]["archive_score_max"]
   rank=lambda x:int(x>1e-10)-int(x< -1e-10)
   assert rank(scores[0,0]-scores[0,1])==rank(refs[0]-refs[1]) and im<=p["gates"]["identity_map_max"]
   assert np.max(np.abs(scores[1]-scores[0]))<=p["gates"]["identity_score_max"]
   with dest.open("xb") as f:np.savez_compressed(f,uid=uid,maps=maps,masks=m,weights=w,reference=ref,archive_map_max=am,identity_map_max=im,worker_scores=scores)
   finished[dest.name]=sha(dest);torch.cuda.synchronize()
   write(OUT/"status.json",dict(stage="matched_state_gpu",complete=i+1,total=172,seconds=time.perf_counter()-start,guard=obs.read()))
   if i==0:write(OUT/"first_case.json",dict(seconds=time.perf_counter()-before,remaining_estimate_s=(time.perf_counter()-before)*171,archive_max=am,identity_max=im))
 for fn,h in p["source_sha256"].items():assert sha(fn)==h,fn
 write(OUT/"inference_completed.json",dict(complete=True,count=172,raw_sha256=finished,seconds=time.perf_counter()-start,guard=obs.read(),protocol_sha256=sha(OUT/"protocol.json")))
if __name__=="__main__":
 if "--prepare" in sys.argv:prepare()
 elif "--audit" in sys.argv:
  from scripts import audit_flyvis_separate_inputs_20261107 as evaluator
  evaluator.OUT=OUT;evaluator.audit()
 else:
  os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]="2"
  sys.path.insert(0,str(GUARD));from system_memory_guard import guard_entrypoint
  run(guard_entrypoint(timeout=2400))
