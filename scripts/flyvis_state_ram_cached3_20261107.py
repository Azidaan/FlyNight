"""Same frozen experiment, with frame and independent-state batching only."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/"runs/flyvis_state_subgroups_ram_cached3_20261107"
OLD=ROOT/"runs/flyvis_state_subgroups_20261107"
from tracking.windows_memory_job import MemoryJob
JOB=MemoryJob(4*1024**3)
import psutil
assert psutil.virtual_memory().available>=12*1024**3,"No system memory headroom; scientific run not started"
STAGE=sys.argv[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".next")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");os.replace(temp,path)
save(OUT/("memory_limit_"+STAGE+".json"),dict(pid=os.getpid(),birth=psutil.Process().create_time(),job=JOB.read()))
manifest=json.loads((OUT/"recovery_manifest.json").read_text(encoding="utf-8"))
for path,digest in manifest["new_sources"].items():assert sha(ROOT/path)==digest,path
assert sha(OUT/"protocol.json")==manifest["scientific_protocol_sha256"]

class FrameChunkEye:
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def __call__(self,x,*args,**kwargs):
        import torch
        return torch.cat([self.base(x[:,i:i+1],*args,**kwargs) for i in range(x.shape[1])],dim=1)

def bounded_states(net):
    import torch
    original=net.simulate
    def simulate(movie_input,dt,initial_state="auto",as_states=False,**kwargs):
        if movie_input.shape[0]<=2:return original(movie_input,dt,initial_state=initial_state,as_states=as_states,**kwargs)
        assert as_states and not isinstance(initial_state,str) and not kwargs
        assert list(initial_state.nodes)==["activity"] and not list(initial_state.edges)
        blocks=[]
        for start in range(0,movie_input.shape[0],2):
            values=initial_state.nodes.activity[start:start+2]
            part=net._initial_state(net._param_api(),len(values));part.nodes.activity=values.clone();part=net._state_api(part)
            seq=original(movie_input[start:start+2],dt,initial_state=part,as_states=True)
            assert all(list(s.nodes)==["activity"] and not list(s.edges) for s in seq)
            blocks.append([s.nodes.activity.clone() for s in seq])
        combined=[]
        for step in range(movie_input.shape[1]):
            state=net._initial_state(net._param_api(),movie_input.shape[0])
            state.nodes.activity=torch.cat([block[step] for block in blocks],dim=0)
            combined.append(net._state_api(state))
        return combined
    net.simulate=simulate

def patch_factory():
    from tracking.flyvis_motion import FlyVisMotionEstimator
    original=FlyVisMotionEstimator.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs);self.eye=FrameChunkEye(self.eye);bounded_states(self.network)
    FlyVisMotionEstimator.__init__=init

def probe():
    import cv2,numpy as np,torch,importlib
    from tracking.flyvis_adapter import activate_flyvis
    fv=activate_flyvis()
    for m in (fv,importlib.import_module("flyvis.network.initialization"),importlib.import_module("flyvis.task.decoder")):m.device=torch.device("cpu")
    torch.set_default_device("cpu");torch.set_default_dtype(torch.float64);torch.set_num_threads(2);cv2.setNumThreads(2)
    assert torch.empty(0).device.type=="cpu" and not torch.cuda.is_initialized()
    from tracking.flyvis_motion import FlyVisMotionEstimator
    from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha
    from scripts.flyvis_offset_recovery2_20261104 import frames_for
    patch_factory()
    model=FlyVisMotionEstimator(mode="full",source_fps=30,flow_gain=1,temporal_resampling="hold",subtract_initial_baseline=False,geometry_version="legacy_v1")
    model.network.double();model.decoder.double();model.eye.conv.double()
    net,dec=model.network,model.decoder;net.eval().requires_grad_(False);dec.eval().requires_grad_(False);net.clamp()
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"));assert parameter_sha(net,dec)==p["parameter_sha"]
    c=p["cases"][0];ref=np.load(OLD/"cases"/(c["id"]+".npz"));gray=np.load(OLD/"textures.npz")[str(c["scene"])]
    frames=frames_for(gray,c);h,w=gray.shape;top=h-512 if c["axis"]==1 and c["speed"]==4 else (h-512)//2;left=(w-512)//2
    extra=[]
    for j in range(25,49):
        d=j*c["speed"];y=top-(d if c["axis"]==1 else 0);x=left-(d if c["axis"]==0 else 0)
        extra.append(np.rint(np.clip(127.5+c["contrast"]*(gray[y:y+512,x:x+512].astype(np.float32)-127.5),0,255)).astype(np.uint8))
    cart=np.stack([cv2.resize(f,(391,391),interpolation=cv2.INTER_AREA) for f in np.concatenate([frames,np.stack(extra)])])
    with torch.inference_mode():
        eye=model.eye(torch.from_numpy(cart.astype(np.float64)/255)[None])[0,:,0].cpu().numpy()
        eye_diff=float(np.max(np.abs(eye-ref["hexes"])));assert eye_diff<=1e-12
        initial=torch.from_numpy(ref["initial_states"]);result=np.empty_like(ref["flow"])
        for si,suffix in enumerate([eye[55:79],eye[79:]]):
            state=net._initial_state(net._param_api(),len(initial));state.nodes.activity=initial.clone();state=net._state_api(state)
            for j in range(24):
                seq=net.simulate(torch.from_numpy(suffix[j])[None,None,None].expand(len(initial),2,1,721),model.dt,initial_state=state,as_states=True)
                state=seq[-1]
                result[si,:,j]=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1)*np.array([512/169,-512/169])
                save(OUT/"probe_status.json",dict(scenario=si,frame=j,job=JOB.read()))
    difference=float(np.max(np.abs(result-ref["flow"])));assert difference<=1e-10
    mask=np.load(ROOT/"runs/flyvis_offset_recovery2_20261104/geometry.npz")["mask"]
    # Discrete signs at every evaluated frame/arm must also agree.
    sign_exact=np.array_equal(np.sign(result[:,:,:,mask,:].mean(axis=3)),np.sign(ref["flow"][:,:,:,mask,:].mean(axis=3)))
    assert sign_exact and parameter_sha(net,dec)==p["parameter_sha"] and not torch.cuda.is_initialized()
    proof=dict(passed=True,case=c["id"],source_sha256=sha(OLD/"cases"/(c["id"]+".npz")),eye_max_diff=eye_diff,
               full_field_max_diff=difference,fields_byte_exact=np.array_equal(result,ref["flow"]),evaluation_signs_exact=sign_exact,
               job=JOB.read(),memory=psutil.Process().memory_info()._asdict(),no_scientific_results_replaced=True)
    save(OUT/"batch_equivalence.json",proof);print(json.dumps(proof))

if STAGE=="probe":probe()
elif STAGE=="run":
    proof=json.loads((OUT/"batch_equivalence.json").read_text());assert proof["passed"]
    patch_factory()
    from scripts import flyvis_state_subgroups_20261107 as base
    base.OUT=OUT
    # Reuse exact previously decoded grayscale pixels, avoiding repeated codec
    # initialization. The predictor only ever consumes the resulting gray frame.
    import cv2,numpy as np
    source_info=json.loads((OLD/"source_images.json").read_text())["sources"]
    cached=np.load(OLD/"textures.npz")
    table={v["video"]:(cached[str(i)],v) for i,v in enumerate(source_info)}
    for video,(gray,info) in table.items():
        assert hashlib.sha256(gray.tobytes()).hexdigest()==info["frame_sha256"]
        assert np.array_equal(cv2.cvtColor(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),cv2.COLOR_BGR2GRAY),gray)
    class CachedFirstFrame:
        def __init__(self,path):
            self.gray,self.info=table[Path(path).stem]
        def read(self):return True,cv2.cvtColor(self.gray,cv2.COLOR_GRAY2BGR)
        def get(self,key):
            assert key==cv2.CAP_PROP_FPS
            return self.info["fps"]
        def release(self):pass
    cv2.VideoCapture=CachedFirstFrame
    original_save=base.save
    first_new=[False]
    def checked_save(path,data):
        original_save(path,data)
        if path.parent.name=="cases" and not first_new[0]:
            import shutil
            done=len(list((OUT/"cases").glob("*.json")));remaining=64-done
            required=2*path.with_suffix(".npz").stat().st_size*remaining+5_000_000_000
            assert shutil.disk_usage(OUT).free>=required
            original_save(OUT/"recovery_first_case_cost.json",dict(seconds=data["seconds"],remaining=remaining,forecast_seconds=remaining*data["seconds"],disk_required=required,disk_free=shutil.disk_usage(OUT).free,job=JOB.read()))
            first_new[0]=True
    base.save=checked_save
    base.run()
    save(OUT/"final_memory_usage.json",dict(job=JOB.read(),memory=psutil.Process().memory_info()._asdict()))
elif STAGE=="audit":
    from scripts import audit_flyvis_state_subgroups_20261107 as audit
    audit.OUT=OUT;audit.main()
else:raise ValueError(STAGE)
