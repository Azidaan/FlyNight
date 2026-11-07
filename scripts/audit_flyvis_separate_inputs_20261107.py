"""Evaluate stored maps without importing the inference worker."""
import csv,gzip,json,hashlib
from pathlib import Path
import numpy as np
OUT=Path("runs/flyvis_separate_inputs_20261107")
def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def audit():
 p=read(OUT/"protocol.json");done=read(OUT/"inference_completed.json");ins=read(OUT/"inputs_completed.json")
 assert done["protocol_sha256"]==sha(OUT/"protocol.json") and done["count"]==172
 with gzip.open("runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz","rt") as f:
  truth={int(r["uid"]):(r["video"],int(r["target_side"])) for r in csv.DictReader(f) if r["split"]=="evaluate26" and r["exactly2"]=="1" and r["label_reason"]=="ok" and r["exclusive_choice"]}
 correct=[];ranks=[];videos=[];scores_saved=[];maxrec=0.;maxarchive=0.;maxidentity=0.
 for c in p["cases"]:
  uid=c["uid"];raw=OUT/"raw"/f"{uid}.npz";ip=OUT/"inputs"/raw.name
  assert sha(raw)==done["raw_sha256"][raw.name] and sha(ip)==ins["sha256"][ip.name]
  with np.load(raw,allow_pickle=False) as z:
   m=z["masks"];x=z["maps"].astype("float64");w=z["weights"];ref=z["reference"]
   assert x.shape==(5,8,721) and m.shape==(2,721) and all(m.any(1)) and np.isfinite(x).all()
   sc=np.array([[sum(float(x[k,ch,mask].sum())/int(mask.sum())*float(w[ch]) for ch in range(8)) for mask in m] for k in range(5)])
   maxrec=max(maxrec,float(np.abs(sc-z["worker_scores"]).max()))
   archived=np.array([sum(float(ref[ch,mask].sum())/mask.sum()*w[ch] for ch in range(8)) for mask in m])
   maxarchive=max(maxarchive,float(z["archive_map_max"]));maxidentity=max(maxidentity,float(z["identity_map_max"]))
  v,side=truth[uid];assert v==c["video"]
  # The separated candidate A is read from row3, B from row4; union uses row2 for both.
  scores=np.stack([archived,sc[2],np.array([sc[3,0],sc[4,1]])])
  margin=(scores[:,0]-scores[:,1])*side;vote=np.where(np.abs(margin)<=1e-10,.5,(margin>0).astype(float))
  orig=(sc[0,0]-sc[0,1])*side;identity=(sc[1,0]-sc[1,1])*side
  assert np.sign(orig)==np.sign(identity) and float(abs(orig))>1e-10 and vote[0]==float(orig>0)
  with np.load(ip,allow_pickle=False) as z:
   assert len(z["frames"])==z["inputs"].shape[1] and (np.diff(z["frames"])>=0).all()
   cur=z["crops"][-1];coords=(z["boxes"].reshape(2,2,2)-cur[:2])/(cur[2:]-cur[:2])
   for j in range(2):
    center=coords[j].mean(0);half=(coords[j,1]-coords[j,0])*.625
    assert np.allclose(z["windows"][j],[center-half,center+half],rtol=0,atol=1e-12)
  correct.append(vote);ranks.append(margin);videos.append(v);scores_saved.append(dict(uid=uid,video=v,scores=scores.tolist(),correct=vote.tolist()))
 assert maxrec<1e-12
 votes=np.array(correct);vid=np.array(videos);cluster=p["cluster_videos"]
 counts=np.array([(vid==v).sum() for v in cluster])
 draws=np.random.default_rng(42).multinomial(len(cluster),np.full(len(cluster),1/len(cluster)),size=20000)
 den=draws@counts;draws=draws[den>0];den=den[den>0]
 comparisons=[]
 for a,b,primary in [(2,0,True),(2,1,True),(1,0,False)]:
  delta=votes[:,a]-votes[:,b];by=np.array([delta[vid==v].sum() for v in cluster]);boot=100*(draws@by)/den
  comparisons.append(dict(name=["common","union","separate"][a]+"-"+["common","union","separate"][b],primary=primary,
   delta_pp=float(delta.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),ci97_5_pp=np.quantile(boot,[.0125,.9875]).tolist(),
   gains=int((delta>0).sum()),losses=int((delta<0).sum()),changed=int((votes[:,a]!=votes[:,b]).sum())))
 result=dict(passed=True,pairs=172,contributing_videos=len(set(videos)),cluster_videos=len(cluster),conditions=[dict(name=n,correct=float(votes[:,i].sum()),accuracy=float(votes[:,i].mean()),ties=int((votes[:,i]==.5).sum())) for i,n in enumerate(["common","union","separate"])],
  comparisons=comparisons,main=all(x["ci97_5_pp"][0]>0 for x in comparisons if x["primary"]),score_recomputation_max=maxrec,archive_map_max=maxarchive,identity_map_max=maxidentity,
  protocol_sha256=sha(OUT/"protocol.json"),inference_manifest_sha256=sha(OUT/"inference_completed.json"),cases=scores_saved,
  limitation="Historical diagnostic only; neutral padded windows after shared retina. Fixed query-box coordinates across history; no true instance masks or full tracker/HOTA.")
 with (OUT/"independent_audit.json").open("x",encoding="utf8") as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
 print(json.dumps({k:result[k] for k in ["pairs","conditions","comparisons","main"]}),flush=True)
if __name__=="__main__":audit()
