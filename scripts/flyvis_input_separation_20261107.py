"""Matched input-window intervention; no training and no labels in inference."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
OUT=Path("runs/flyvis_separate_inputs_20261107")
OLD=Path("runs/flyvis_atlas_core_20261107")
PAIR=Path("runs/z13_temporal_pair_readiness_20261025")
GUARD=ROOT/"runs/flyvis_repairs_20261106/unification_20261107"
IDENT="flyvis_separate_inputs_v1"
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(".pending");tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8");os.replace(tmp,p)
def status(stage,**kw):write(OUT/"status.json",dict(stage=stage,updated=datetime.now(timezone(timedelta(hours=3))).isoformat(),**kw))
def archive_meta(c):return next(r for r in read(PAIR/"inputs"/(c["video"]+".json"))["rows"] if r["uid"]==c["uid"])
def verify(p):
 for fn,h in p["source_sha256"].items():assert sha(fn)==h,fn
def prepare():
 from scripts.flyvis_atlas_core_20261107 import selected
 assert not OUT.exists(),"Preserve previous execution"
 old=read(OLD/"protocol.json");assert read(OLD/"independent_audit.json")["passed"]
 rows=selected();byuid={int(r["uid"]):r for r in rows}
 cases=[dict(c,choices=[int(byuid[c["uid"]]["j"]),int(byuid[c["uid"]]["k"])]) for c in old["cases"]]
 assert len(cases)==172 and {c["uid"] for c in cases}==set(byuid)
 sources=dict(old["source_sha256"])
 for fn in [Path(__file__).relative_to(ROOT),"scripts/audit_flyvis_separate_inputs_20261107.py","scripts/chain_flyvis_separate_inputs_20261107.py",OLD/"protocol.json",OLD/"independent_audit.json","scripts/z13_data.py","tracking/retina.py","tools/flyvis-runtime/flyvis/datasets/rendering/eye.py"]:
  sources[Path(fn).as_posix()]=sha(fn)
 for v in sorted({c["video"] for c in cases}):
  for fn in [Path("data/bdd100k/videos/train")/(v+".mov"),Path("runs/train_cache")/v/"detections/raw_video.npz"]:
   sources[fn.as_posix()]=sha(fn)
 now=datetime.now(timezone(timedelta(hours=3)))
 p=dict(id=IDENT,registered=now.isoformat(),git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
  author="Давай проверим на тех же эпизодах: отдельные вырезки каждой машины с небольшим окружением против нынешней общей карты. Сохраним одинаковый масштаб и те же прошлые кадры",
  hypothesis="Separate candidate windows before the recurrent core improve frozen T4/T5 candidate choices; not established.",
  scope="Same172 historical evaluate26 pairs,23 contributing videos; full original episode prefixes. No selection of images/results, training, new data or HOTA.",
  input="Replay each full video from frame0 through original lamina preprocessing,blur_downscale4. At archived fr_frame extract exactly archived fr_crop. Original BoxEye grid and source-pixels per receptor. Quantize all inputs to original float16 then float32.",
  intervention="Inflate CURRENT DETECTION width/height by1.25 (12.5% each side), express in current common crop normalized coordinates, keep this mask in the SAME canonical coordinates for ALL original prefix frames. Outside=127.5 AFTER shared retinal front-end and BEFORE BoxEye. No recentering or object zoom. Canvas/render scale exactly old crop at each time. Not a known candidate trajectory or perfect instance segmentation; overlapping objects remain possible.",
  controls="Archived common-map baseline reused scientifically. Re-render common input and cloned sequence validate archive/identity. Union control retains BOTH inflated windows on one canvas, same neutral outside: tests object separation versus just removing background.",
  constants=dict(context=1.25,neutral=127.5,bootstrap=20000,bootstrap_seed=42,tie=1e-10),
  primary="Separate-common AND separate-union paired correct-choice differences. Both lower Bonferroni97.5% bounds>0 for main support; ordinary95% too. Resample all26 original videos including3 without eligible pairs. Ties0.5; publish unfavorable effects. Reused exploratory data.",
  scoring="Same observed-exemplar target coefficients from old w chunks; original normalization; direct T4/T5 channels4:12 only; mean over original FULL candidate masks. Labels only in audit.",
  gates=dict(common_retinal_float16_max=0.0,archive_normalized_map_max=.002,archive_score_max=.00002,identity_map_max=.00001,identity_score_max=.0000001),
  budget="Sequential guardedCPU inputs,GPUworker with shared mutex,independentCPU audit. FreeRAM reserve max(6GiB,20%),+2GiB to start; no per-process cap. Disk>=5GiB. Finite172; first full video/case timing.4800s each compute stage,no unbounded recovery; Saturday full day.",
  source_sha256=sources,cases=cases,cluster_videos=read("runs/flyvis_exclusive_t4t5_20261107/protocol.json")["videos"]["evaluate26"])
 OUT.mkdir();(OUT/"inputs").mkdir();(OUT/"raw").mkdir();write(OUT/"protocol.json",p)
 cat=read("experiments/project_history.json");assert not any(c["id"]==IDENT for c in cat["experiments"])
 cat["experiments"].append(dict(id=IDENT,title="Раздельные входы FlyVis для двух рамок при прежнем масштабе",area="flyvis",state="ready",started=now.strftime("%Y-%m-%d"),tags=["fly-problem-P8"],
  question="Помогают ли раздельные входы двух рамок выбрать продолжение при прежнем масштабе и тех же прошлых кадрах?",
  hypothesis=p["hypothesis"],expected=p["intervention"]+" "+p["controls"],
  acceptance=[dict(criterion="Побитно восстановить прежний float16 вход, проверить масштаб/кадры, архивное решение и тождественную копию; независимый аудит172 пар.",met=None),dict(criterion=p["primary"],met=None)],
  obtained="",interpretation="",decision="",cannot_claim="Диагностический выбор на прежних исторических видео, не качество полных треков. Окна не сегментируют перекрывающиеся машины. Маски, заданные текущими детекциями, не знают прошлую истинную траекторию.",
  evidence=[dict(kind="file",ref=(OUT/"protocol.json").as_posix(),claim="Условия и контроли до расчёта")]))
 write("experiments/project_history.json",cat)
 plan=Path("dissertation/ЕДИНЫЙ_ПЛАН.md");text=plan.read_text(encoding="utf8")
 entry=f"\n**{now:%d.%m.%Y, %H:%M} MSK** — Зарегистрирована проверка {IDENT}: {OUT}/protocol.json. Те же172 пары и полные префиксы; контекст1,25 и нейтральное заполнение при прежнем масштабе. Контроль оставляет обе вырезки на одном поле. Приёмка — две положительные нижние границы97,5%; разметка только в оценщике. Это входное вмешательство до ядра; итог пока не получен.\n"
 plan.write_text(text.replace("## Журнал изменений","## Журнал изменений\n"+entry,1),encoding="utf8")
 status("registered",complete=0,total=172);print("Registered172",flush=True)

def prepare_inputs(obs):
 import numpy as np,cv2,torch
 from tracking.flyvis_adapter import activate_flyvis
 from tracking.retina import make_retina
 torch.set_num_threads(2);torch.set_default_device("cpu");torch.cuda.is_available=lambda:False
 fly=activate_flyvis(device="cpu");from flyvis.datasets.rendering import BoxEye
 cfg=fly.NetworkView("flow/0000/000").dir.config
 eye=BoxEye(extent=int(cfg.network.connectome.extent),kernel_size=int(cfg.task.dataset.boxfilter.kernel_size))
 eye.conv=eye.conv.cpu();S=int(eye.min_frame_size[0]);assert S==int(eye.min_frame_size[1])
 p=read(OUT/"protocol.json");verify(p);by={}
 for c in p["cases"]:by.setdefault(c["video"],[]).append(c)
 manifest={};start=time.perf_counter()
 def render(patches):
  ar=np.stack([cv2.resize(x,(S,S),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0 for x in patches])[:,None]
  with torch.inference_mode():return eye(torch.from_numpy(ar))[:,0].cpu().numpy().astype(np.float16)
 for vi,(video,cases) in enumerate(sorted(by.items())):
  begun=time.perf_counter();todo=[];need={};rawz=np.load("runs/z13_data/"+video+".npz",allow_pickle=False)
  with np.load(Path("runs/train_cache")/video/"detections/raw_video.npz",allow_pickle=False) as d:
   for c in cases:
    dest=OUT/"inputs"/f"{c['uid']}.npz"
    if dest.exists():manifest[dest.name]=sha(dest);continue
    meta=archive_meta(c);a,b=meta["state_prefix_start"],meta["current_array_index"]
    frames=rawz["fr_frame"][a:b+1].copy();crops=rawz["fr_crop"][a:b+1].copy();original=rawz["fr_hex"][a:b+1].copy()
    assert frames[-1]==meta["current_frame"] and np.array_equal(crops[-1],meta["current_crop"])
    sel=np.flatnonzero((d["frame_idx"]==int(frames[-1]))&(d["conf"]>=.25))
    boxes=d["xyxy"][sel[c["choices"]]].astype(np.float64)
    cur=crops[-1].astype(float);side=cur[2:]-cur[:2];assert side[0]==side[1]
    norm=(boxes.reshape(2,2,2)-cur[:2])/side;center=norm.mean(1);half=(norm[:,1]-norm[:,0])*p["constants"]["context"]/2
    item=dict(case=c,frames=frames,crops=crops,old=original,windows=np.stack([center-half,center+half],1),seq=np.empty((4,len(frames),721),np.float16),boxes=boxes)
    todo.append(item)
    for ti,(f,cr) in enumerate(zip(frames,crops)):need.setdefault(int(f),[]).append((item,ti,cr))
  if need:
   cap=cv2.VideoCapture(str(Path("data/bdd100k/videos/train")/(video+".mov")));assert cap.isOpened()
   fps=float(read(Path("runs/train_cache")/video/"motion/flyvis_tiled.json")["fps"]);retina=make_retina("lamina",fps,blur_downscale=4)
   for frame_no in range(max(need)+1):
    ok,rgb=cap.read();assert ok,(video,frame_no)
    pre=retina.process(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY))
    for item,ti,cr in need.get(frame_no,[]):
     x1,y1,x2,y2=map(int,cr);patch=pre[y1:y2,x1:x2];H,W=patch.shape;xx=(np.arange(W)+.5)/W;yy=(np.arange(H)+.5)/H
     keep=[(yy[:,None]>=w[0,1])&(yy[:,None]<=w[1,1])&(xx[None,:]>=w[0,0])&(xx[None,:]<=w[1,0]) for w in item["windows"]]
     variants=[patch,np.where(keep[0]|keep[1],patch,127.5),np.where(keep[0],patch,127.5),np.where(keep[1],patch,127.5)]
     hx=render(variants);err=float(np.max(np.abs(hx[0].astype(float)-item["old"][ti].astype(float))))
     assert err<=p["gates"]["common_retinal_float16_max"],("common_retinal_reconstruction",item["case"]["uid"],frame_no,err)
     item["seq"][:,ti]=hx
    if frame_no%100==0:status("reconstructing_inputs",video=video,video_index=vi,video_frame=frame_no,complete=len(manifest),total=172,guard=obs.read())
   cap.release()
   for item in todo:
    c=item["case"];dest=OUT/"inputs"/f"{c['uid']}.npz";tmp=dest.with_suffix(".pending")
    with np.load(c["full"][0],allow_pickle=False) as z:
     assert int(z["uid"][c["full"][1]])==c["uid"]
     reference=z["full"][c["full"][1],:,:34][:,4:12].T.copy();masks=z["retinal_mask"][c["full"][1],1:3].copy()
    with np.load(c["w"][0],allow_pickle=False) as z:weights=z["seed"][c["w"][1],4:12].astype("float64")
    with tmp.open("xb") as f:np.savez_compressed(f,uid=c["uid"],inputs=item["seq"],frames=item["frames"],crops=item["crops"],windows=item["windows"],boxes=item["boxes"],masks=masks,reference=reference,weights=weights,eye_size=S)
    os.replace(tmp,dest);manifest[dest.name]=sha(dest)
  rawz.close();secs=time.perf_counter()-begun;status("inputs_video_completed",complete=len(manifest),total=172,video=video,seconds=secs)
  if vi==0:write(OUT/"first_input_video.json",dict(video=video,seconds=secs,remaining_estimate_s=secs*(len(by)-1),cases=len(cases),guard=obs.read()))
 verify(p);assert len(manifest)==172 and not torch.cuda.is_initialized()
 write(OUT/"inputs_completed.json",dict(complete=True,count=len(manifest),sha256=manifest,seconds=time.perf_counter()-start,guard=obs.read(),protocol_sha256=sha(OUT/"protocol.json")))

def infer(obs):
 import numpy as np,torch
 from scripts.z13_target_model import build_arm
 from tracking.gpu_lock import GpuLock
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 p=read(OUT/"protocol.json");verify(p);inputs=read(OUT/"inputs_completed.json")
 norm=read("runs/z13_target_center_dev_20261018/C/normalization.json");mu=np.asarray(norm["mean"])[4:12];sd=np.asarray(norm["sd"])[4:12];g=p["gates"]
 finished={};begun=time.perf_counter()
 with GpuLock("flyvis_fovea_gpu"),torch.no_grad():
  _,_,bb=build_arm("C",torch,None);assert str(bb.dev).startswith("cuda") and not bb.net.training
  for i,c in enumerate(p["cases"]):
   uid=c["uid"];dest=OUT/"raw"/f"{uid}.npz"
   if dest.exists():finished[dest.name]=sha(dest);continue
   before=time.perf_counter();ip=OUT/"inputs"/f"{uid}.npz";assert sha(ip)==inputs["sha256"][ip.name]
   with np.load(ip,allow_pickle=False) as z:x=z["inputs"].astype("float32");m=z["masks"];ref=z["reference"].astype("float64");w=z["weights"]
   seq=np.stack([x[0],x[0].copy(),x[1],x[2],x[3]])
   hx=torch.from_numpy(seq).to(bb.dev);st=bb.init(hx[:,:1])
   for t in range(1,hx.shape[1]):st,f=bb.step(st,hx[:,t-1],hx[:,t])
   maps=((f[:,4:12].cpu().numpy().astype("float64")-mu[None,:,None])/sd[None,:,None]).astype("float32")
   assert np.isfinite(maps).all();am=float(np.abs(maps[0]-ref).max());im=float(np.abs(maps[1]-maps[0]).max())
   scores=np.stack([maps[:,:,mask].astype("float64").mean(2)@w for mask in m],1)
   refs=np.array([ref[:,mask].mean(1)@w for mask in m]);err=float(np.max(np.abs(scores[0]-refs)))
   rank=lambda v:int(v>1e-10)-int(v< -1e-10)
   assert am<=g["archive_normalized_map_max"] and err<=g["archive_score_max"],("archive",uid,am,err)
   assert rank(scores[0,0]-scores[0,1])==rank(refs[0]-refs[1]),("archive_rank",uid)
   assert im<=g["identity_map_max"] and np.max(np.abs(scores[1]-scores[0]))<=g["identity_score_max"],("identity",uid,im)
   with dest.with_suffix(".pending").open("xb") as f:np.savez_compressed(f,uid=uid,maps=maps,masks=m,weights=w,reference=ref,archive_map_max=am,identity_map_max=im,worker_scores=scores)
   os.replace(dest.with_suffix(".pending"),dest);finished[dest.name]=sha(dest);torch.cuda.synchronize()
   secs=time.perf_counter()-before;status("gpu",complete=len(finished),total=172,uid=uid,case_seconds=secs,guard=obs.read())
   if i==0:write(OUT/"first_gpu_case.json",dict(uid=uid,seconds=secs,remaining_estimate_s=secs*171,guard=obs.read(),archive_max=am,identity_max=im))
   if i%10==0:print("GPU",i+1,uid,round(secs,2),flush=True)
 assert len(finished)==172;verify(p)
 write(OUT/"inference_completed.json",dict(complete=True,count=172,raw_sha256=finished,seconds=time.perf_counter()-begun,guard=obs.read(),protocol_sha256=sha(OUT/"protocol.json")))

if __name__=="__main__":
 if "--prepare" in sys.argv:prepare()
 else:
  if "--inputs" in sys.argv:os.environ["CUDA_VISIBLE_DEVICES"]=""
  os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8";os.environ["OMP_NUM_THREADS"]=os.environ["MKL_NUM_THREADS"]="2"
  sys.path.insert(0,str(GUARD));from system_memory_guard import guard_entrypoint
  obs=guard_entrypoint(timeout=4800)
  if "--inputs" in sys.argv:prepare_inputs(obs)
  elif "--gpu" in sys.argv:infer(obs)
