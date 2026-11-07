"""Rebuild inputs and independently score all saved partial-state outputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_state_subgroups_20261107"
sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    from scripts.flyvis_y1_latency_fp64_20261104 import cpu_eye
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    done=json.loads((OUT/"inference_completed.json").read_text())
    assert done["complete"] and done["cases"]==len(p["cases"])==64 and not done["cuda_initialized"]
    assert done["protocol_sha256"]==sha(OUT/"protocol.json")
    for f,h in p["frozen_sources"].items():assert sha(ROOT/f)==h,f
    for filename,key in [("indices.npz","indices_sha256"),("textures.npz","textures_sha256"),("source_images.json","source_images_sha256"),("node_types.npy","node_types_sha256")]:
        assert sha(OUT/filename)==done[key]
    geo_path=ROOT/"runs/flyvis_offset_recovery2_20261104/geometry.npz";assert sha(geo_path)==p["geometry_sha256"]
    geo=np.load(geo_path);mask=geo["mask"];indices=np.load(OUT/"indices.npz");textures=np.load(OUT/"textures.npz")
    source=json.loads((OUT/"source_images.json").read_text())["sources"]
    assert len(source)==8 and [q["video"] for q in source]==p["videos"]
    assert not set(p["videos"])&set(p["selection"]["excluded_ids"])
    current=sorted(q.stem for q in (ROOT/"data/bdd100k/videos/train").glob("*.mov"))
    expected=sorted([v for v in current if v not in set(p["selection"]["excluded_ids"])],
                    key=lambda v:hashlib.sha256(("flyvis-state-subgroups-20261107:"+v).encode()).hexdigest())[:8]
    assert expected==p["videos"]
    for s,v in enumerate(source):
        assert hashlib.sha256(textures[str(s)].tobytes()).hexdigest()==v["frame_sha256"]
        assert v["source_sha256"]==p["source_video_sha256"][v["video"]]
    # Derive membership from the full node-name array, not the runner's layer_index.
    names=np.load(OUT/"node_types.npy")
    assert names.shape==(45669,)
    for g,types in p["groups"].items():
        assert np.array_equal(np.sort(indices[g]),np.flatnonzero(np.isin(names,types)))
    parent=np.concatenate([indices[g] for g in p["primary_groups"]])
    assert len(np.unique(parent))==len(parent) and np.array_equal(np.sort(parent),np.sort(indices["parent"]))
    rows=[];maximum_eye_error=0.
    for ci,c in enumerate(p["cases"]):
        path=OUT/"cases"/(c["id"]+".npz");m=json.loads(path.with_suffix(".json").read_text())
        assert sha(path)==m["sha256"] and m["protocol_sha256"]==sha(OUT/"protocol.json")
        z=np.load(path);f=z["flow"];initial=z["initial_states"];a,b=z["boundary_activity"]
        assert f.shape==(2,len(p["arms"]),24,721,2) and np.isfinite(f).all() and np.isfinite(initial).all()
        assert initial.shape==(len(p["arms"]),45669)
        for i,v in enumerate([a,b,a,b,b,a]):assert np.array_equal(initial[i],v)
        for gi,g in enumerate(p["groups"]):
            ix=indices[g];assert len(np.unique(ix))==len(ix) and ix.min()>=0 and ix.max()<len(a)
            keep=np.ones(len(a),bool);keep[ix]=False;erase,insert=initial[6+2*gi:8+2*gi]
            assert np.array_equal(erase[ix],b[ix]) and np.array_equal(erase[keep],a[keep])
            assert np.array_equal(insert[ix],a[ix]) and np.array_equal(insert[keep],b[keep])
        for u,v in [(0,2),(1,3),(0,5),(1,4)]:assert np.array_equal(f[:,u],f[:,v])
        assert all(m[k] for k in ["identity_and_full_states_exact","donors_unchanged","parameters_and_buffers_unchanged"])
        image=textures[str(c["scene"])];height,width=image.shape
        top=height-512 if c["axis"]==1 and c["speed"]==4 else (height-512)//2;left=(width-512)//2
        frames=[]
        for j in range(103):
            steps=0 if j<=30 else (j-30 if j<=54 else (78-j if j<=78 else j-54))
            offset=c["speed"]*steps;y=top-(offset if c["axis"]==1 else 0);x=left-(offset if c["axis"]==0 else 0)
            assert min(x,y)>=0 and x+512<=width and y+512<=height
            raw=image[y:y+512,x:x+512].astype(np.float32)
            frame=np.rint(np.clip(127.5+c["contrast"]*(raw-127.5),0,255)).astype(np.uint8)
            assert hashlib.sha256(frame.tobytes()).hexdigest()==str(z["frame_sha256"][j]);frames.append(frame)
        eye_error=float(np.max(np.abs(cpu_eye(np.stack(frames),geo["centers"])-z["hexes"])))
        assert eye_error<=1e-5 and m["eye_max_diff"]<=1e-5
        maximum_eye_error=max(maximum_eye_error,eye_error)
        vectors=f[:,:,:,mask,:].mean(axis=3);metrics={}
        for si,scenario in enumerate(p["scenarios"]):
            truth=np.zeros((24,2));truth[:,c["axis"]]=c["speed"]*(-1 if scenario=="reverse" else 1);arms={}
            for ai,name in enumerate(p["arms"]):
                y=vectors[si,ai];errors=np.sqrt(np.sum((y-truth)**2,axis=-1));sign=y[:,c["axis"]]*truth[:,c["axis"]]>0
                hit=next((j for j in range(22) if np.all(sign[j:j+3])),None)
                arms[name]=dict(epe=float(errors.mean()),early_epe=float(errors[:6].mean()),late_epe=float(errors[-6:].mean()),
                                latency=hit,capped_latency=24 if hit is None else hit,correct_sign_fraction=float(sign.mean()))
            metrics[scenario]=arms
        rows.append(dict(case=c,metrics=metrics,sha256=m["sha256"]))
        (OUT/"audit_status.json").write_text(json.dumps(dict(done=ci+1,total=len(p["cases"])))+"\n",encoding="utf-8")
    groups={};pairs={};scenes=sorted({c["scene"] for c in p["cases"]})
    draw=np.random.default_rng(42).integers(0,len(scenes),(20000,len(scenes)))
    for contrast in (1.,.25):
        cohort=[r for r in rows if r["case"]["contrast"]==contrast];gc={}
        for scenario in p["scenarios"]:
            gc[scenario]={}
            for arm in p["arms"]:
                ms=[r["metrics"][scenario][arm] for r in cohort]
                gc[scenario][arm]={k:float(np.mean([m[k] for m in ms])) for k in ("epe","early_epe","late_epe","capped_latency","correct_sign_fraction")}
                gc[scenario][arm]["not_found"]=sum(m["latency"] is None for m in ms)
            for g in p["groups"]:
                for direction,reference in [("erase","moving"),("insert","stationary")]:
                    for metric in ("capped_latency","epe","early_epe"):
                        ds=np.array([np.mean([r["metrics"][scenario][direction+"_"+g][metric]-r["metrics"][scenario][reference][metric]
                                              for r in cohort if r["case"]["scene"]==s]) for s in scenes])
                        samples=ds[draw].mean(axis=1)
                        pairs[f"{contrast}:{scenario}:{direction}_{g}:{metric}"]=dict(mean=float(ds.mean()),
                            ci95=np.quantile(samples,[.025,.975]).tolist(),ci_family8=np.quantile(samples,[.05/16,1-.05/16]).tolist(),scene_differences=ds.tolist())
        groups[str(contrast)]=gc
    candidates={}
    for g in p["primary_groups"]:
        checks=dict(corrected_latency_reduction=pairs[f"1.0:reverse:erase_{g}:capped_latency"]["ci_family8"][1]<0)
        for ct in ("1.0","0.25"):
            for scenario in p["scenarios"]:
                cur=groups[ct][scenario]["erase_"+g];base=groups[ct][scenario]["moving"]
                for metric in ("epe","early_epe"):checks[f"{ct}:{scenario}:{metric}_not_worse"]=cur[metric]<=base[metric]+1e-12
                if scenario=="reverse":checks[f"{ct}:missing_not_more"]=cur["not_found"]<=base["not_found"]
                else:checks[f"{ct}:correct_sign_not_less"]=cur["correct_sign_fraction"]>=base["correct_sign_fraction"]-1e-12
        candidates[g]=dict(passed=all(checks.values()),checks=checks)
    result=dict(passed=True,cases=len(rows),independent_scenes=len(scenes),protocol_sha256=sha(OUT/"protocol.json"),
        inference_sha256=sha(OUT/"inference_completed.json"),maximum_eye_error=maximum_eye_error,groups=groups,paired=pairs,candidates=candidates,
        primary_met=any(v["passed"] for v in candidates.values()),rows=rows,limits=p["limitations"],
        audit_scope="Independent metric calculation and exact selected/unselected/full-copy checks on saved tensors; all103 frames/case independently reconstructed. OpenCV eye function shared with runner reference check, official BoxEye separate. No second network replay. Eight scenes, not64 independent samples.")
    (OUT/"independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(passed=True,cases=len(rows),primary_met=result["primary_met"],candidates={g:v["passed"] for g,v in candidates.items()})))

if __name__=="__main__":main()
