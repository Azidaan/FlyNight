"""Paired mask-before-cut association replay; never update tracker state or model weights."""
from lab import OUT, ROOT, OLD, GAPS, read, write, fingerprint, sha, now
import hashlib
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(OLD))
from official import labels, masks, bbox, iou, M
from assignment import edges, overlap_matrix
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

prior = read(GAPS / "protocol.json")
old_cases = read(GAPS / "cases.json")["cases"]
former = read(ROOT / "research_ownership_conflict_2179_20261114/twenty_gap_outcomes.json")
detector = ROOT / "weights/baseline_seed42/best.pt"
sam_weights = ROOT / "weights/sam2_official_20261027/sam2.1_hiera_tiny.pt"
label_zip, image_zip = [ROOT / prior[k] for k in ("labelzip","imagezip")]
prefix = "bdd100k/images/seg_track_20/val/"


def eligible(obj):
    return (obj["category"]=="car" and not any((obj.get("attributes") or {}).get(k,False)
            for k in ("ignore","ignored","crowd","iscrowd")) and M.area(obj["rle"])>0)


def first_gap(db, after):
    valid = {idx:{o["id"]:o for o in row["labels"] if eligible(o)} for idx,row in db.items()}
    frames = defaultdict(list)
    for idx, objects in valid.items():
        for identity in objects:
            frames[identity].append(idx)
    events = []
    for identity, indices in frames.items():
        indices.sort()
        for a,b in zip(indices,indices[1:]):
            if a<=after or not 2<=b-a<=8 or not all(i in db for i in range(a,b+1)):
                continue
            target = valid[a][identity]
            others = [o for oid,o in valid[a].items() if oid!=identity]
            if not others or M.area(target["rle"])<512:
                continue
            rival = min(others,key=lambda o:(-iou(bbox(target),bbox(o)), -int(M.area(o["rle"])),o["id"]))
            events.append((b,a,identity,rival["id"]))
    if events:
        b,a,target,rival = min(events)
        return dict(prior=a,return_index=b,target=target,rival=rival,
                    past_boxes=[bbox(valid[a][k]).tolist() for k in (target,rival)])


def prepare():
    if (OUT/"gap_selection.json").exists():
        return read(OUT/"gap_selection.json")
    old = {c["video"]:c for c in old_cases}
    known = {r["case"] for r in former["episodes"] if r["reason"]=="zero_overlap_all_candidates"}
    selected = [dict(c,cohort="known") for c in old_cases if c["id"] in known]
    scanned = []
    with zipfile.ZipFile(label_zip) as z, zipfile.ZipFile(image_zip) as imgs:
        vs = sorted(Path(n).stem for n in z.namelist() if "/rles/val/" in n and n.endswith(".json"))
        split = [(v,"new_video",-1) for v in vs if v not in prior["videos"]]
        split += [(v,"new_episode",c["return_index"]) for v,c in sorted(old.items())]
        for video,cohort,after in split:
            db = labels(z,video)
            event = first_gap(db,after)
            scanned.append(dict(video=video,cohort=cohort,found=bool(event)))
            if not event:
                continue
            a,b=event["prior"],event["return_index"]
            key=f"{video}_{a}_{b}"
            folder=OUT/"frames"/key
            folder.mkdir(parents=True,exist_ok=True)
            fs=[]
            for idx in range(a,b+1):
                name=db[idx]["name"]
                content=imgs.read(prefix+video+"/"+name)
                f=folder/f"{idx-a:05d}.jpg"
                f.write_bytes(content)
                fs.append(dict(index=idx,name=name,image=f.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(content).hexdigest()))
            selected.append(dict(**event,video=video,cohort=cohort,id=key,frames=fs))
    result=dict(cases=selected,scanned=scanned)
    write(OUT/"gap_selection.json",result)
    return result


def run():
    import torch
    import torch.nn.functional as F
    sys.path[:0] = [str(ROOT/"tools/sam2-runtime-20261027"),str(ROOT/"tools/sam2-deps-20261027")]
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tracking.gpu_lock import GpuLock
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cases=prepare()["cases"]
    inputs=[OUT/"protocol.json",OUT/"lab.py", Path(__file__),OUT/"gap_selection.json",detector,sam_weights,
            OLD/"assignment.py",OLD/"official.py",OLD/"actions.json",GAPS/"cases.json",
            ROOT/"tools/sam2-runtime-20261027/sam2/modeling/sam2_base.py",
            ROOT/"tools/sam2-runtime-20261027/sam2/sam2_video_predictor.py",label_zip]
    stamp=fingerprint(inputs)
    receipt=OUT/"gap_inputs_v2.json"
    if receipt.exists():
        assert read(receipt)["sha256"]==stamp["sha256"]
    else:
        write(receipt,stamp)
    expected={(r["case"],r["frame"]):r for r in read(OLD/"actions.json")["traces"]}
    cfg="configs/sam2.1/sam2.1_hiera_t.yaml"
    with zipfile.ZipFile(label_zip) as z, GpuLock(r"Local\DissertationSingleGpu"),torch.inference_mode():
        video_model=build_sam2_video_predictor(cfg,str(sam_weights),device="cuda",apply_postprocessing=False)
        image_model=SAM2ImagePredictor(build_sam2(cfg,str(sam_weights),device="cuda",apply_postprocessing=False))
        yolo=YOLO(str(detector))
        def detect(im):
            out=yolo.predict(im,imgsz=960,conf=.01,iou=.7,classes=[2],max_det=300,rect=True,
                             half=False,batch=1,device=0,verbose=False,save=False)[0].boxes
            return [(i,bb,float(conf)) for i,(bb,conf) in enumerate(zip(out.xyxy.cpu().numpy(),out.conf.cpu().numpy()))]
        decoder={}
        def capture(module,args,kwargs,outputs):
            m,quality,_,score=outputs
            best=int(quality[0].argmax()) if m.shape[1]>1 else 0
            decoder.update(raw=m[0,best].detach().clone(),score=float(score.item()))
        hook=video_model.sam_mask_decoder.register_forward_hook(capture,with_kwargs=True)
        try:
            for idx,c in enumerate(cases):
                assert now()<read(OUT/"protocol.json")["stop_msk"]
                path=OUT/"gap_rows"/(c["id"]+".json")
                if path.exists():
                    continue
                fs=c["frames"]
                assert all(sha(ROOT/f["image"])==f["sha256"] for f in fs)
                imgs=[cv2.imread(str(ROOT/f["image"])) for f in fs]
                db=labels(z,c["video"])
                source=detect(imgs[0])
                overlap=np.asarray([[iou(np.asarray(bb),dd[1]) for dd in source] for bb in c["past_boxes"]])
                utility=np.where(overlap>=.5,1+overlap,0.)
                matched=linear_sum_assignment(-np.c_[utility,np.zeros((2,2))])
                boxes=[None,None]
                for i,j in zip(*matched):
                    if j<len(source) and utility[i,j]>0:
                        boxes[i]=source[j][1]
                forecasts=[{},{}]
                zero=np.zeros(imgs[0].shape[:2],bool)
                for slot,bb in enumerate(boxes):
                    if bb is None:
                        continue
                    st=video_model.init_state(str((ROOT/fs[0]["image"]).parent),offload_video_to_cpu=True,
                                              offload_state_to_cpu=False,async_loading_frames=False)
                    video_model.add_new_points_or_box(st,0,1,box=bb)
                    for k,_,logits in video_model.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=len(fs)-1):
                        if k==0:
                            continue
                        ordinary=(logits[0,0]>0).cpu().numpy()
                        score=float(st["output_dict_per_obj"][0]["non_cond_frame_outputs"][k]["object_score_logits"].item())
                        raw=ordinary
                        if not ordinary.any() and score<=0:
                            assert score==decoder["score"]
                            up=F.interpolate(decoder["raw"][None,None].float(),size=(video_model.image_size,)*2,
                                             mode="bilinear",align_corners=False)
                            raw=(F.interpolate(up,size=ordinary.shape,mode="bilinear",align_corners=False)[0,0]>0).cpu().numpy()
                        forecasts[slot][k]=(ordinary,raw)
                    del st
                rows=[]
                for k,im in enumerate(imgs[1:],1):
                    pool=[r for r in detect(im) if r[2]>.1]
                    ids=[d[0] for d in pool]
                    current=[]
                    if pool:
                        image_model.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                        for _,bb,_ in pool:
                            current.append(image_model.predict(box=bb,multimask_output=False)[0][0].astype(bool))
                    row=db[c["prior"]+k]
                    truth=[masks(z,c["video"],row,oid) for oid in (c["target"],c["rival"])]
                    oracle=overlap_matrix([g[0] for g in truth],[m&truth[0][1] for m in current])
                    ignored=[any(o["id"]==oid and any((o.get("attributes") or {}).get(flag,False)
                             for flag in ("ignore","ignored","crowd","iscrowd")) for o in row["labels"])
                             for oid in (c["target"],c["rival"])]
                    methods={}
                    for col,mode in enumerate(("ordinary","precut")):
                        scores=overlap_matrix([f.get(k,(zero,zero))[col] for f in forecasts],current)
                        pairs=edges(scores,[0,1],ids).pairs
                        if c["cohort"]=="known" and mode=="ordinary":
                            old=expected[c["id"],k]
                            assert old["rids"]==ids and np.allclose(scores,old["scores"]["ordinary"],rtol=0,atol=1e-7)
                            assert [list(pair) for pair in pairs]==old["actions"]["ordinary"]
                        methods[mode]=dict(scores=scores.tolist(),pairs=pairs)
                    rows.append(dict(frame=k,last=k==len(fs)-1,ids=ids,oracle=oracle.tolist(),ignored=ignored,methods=methods))
                write(path,dict(id=c["id"],video=c["video"],cohort=c["cohort"],source=[b is not None for b in boxes],rows=rows))
                print("GAP",idx+1,len(cases),c["cohort"],c["id"],flush=True)
        finally:
            hook.remove()
    assert fingerprint(inputs)["sha256"]==stamp["sha256"]
    write(OUT/"gap_complete.json",{"ended":now(),"cases":len(cases),"inputs_unchanged":True})


if __name__=="__main__":
    run()
