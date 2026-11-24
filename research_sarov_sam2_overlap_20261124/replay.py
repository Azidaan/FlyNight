from pathlib import Path
import sys
import itertools
from collections import defaultdict

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
PILOTS=ROOT/"research_sarov_pilots_20261124"
sys.path.insert(0,str(PILOTS))
from lab import read,write,sha,fingerprint,now
from register import journal
from geometry_data import prepare_scenes
import numpy as np


def catalog(result=None):
    pr=read(OUT/"protocol.json")
    path=ROOT/"experiments/project_history.json"
    data=read(path)
    prefix=OUT.relative_to(ROOT).as_posix()+"/"
    rows=data["experiments"]
    own=next((c for c in rows if c["id"]==pr["id"]),None)
    if result is None:
        assert own is None
        rows.append(dict(id=pr["id"],title="Две перекрывающиеся машины после совместной перестановки номеров",
             area="own_tracker",state="running",started="2026-11-24",question=pr["question"],hypothesis=pr["action"],
             expected=pr["criterion"],acceptance=[dict(criterion=pr["criterion"],met=None)],
             evidence=[dict(kind="file",ref=prefix+"protocol.json")]))
        message="По решению автора облачная реконструкция SAM 3D остановлена до отправки кадра и оплаты. Локальная SAM 2 проверяется на единственной общей перестановке двух наблюдаемых машин в двух сериях из 40 видео."
    else:
        assert own and own["state"]=="running"
        n=result["correct"]
        message=(f"На переходе {result['video']}, кадры {result['source_file_frame']} и {result['target_file_frame']}, "
                 f"локальная SAM 2 правильно назначила {n} из 2 машин, ожиданий {result['wait']}, ошибочных назначений {result['wrong']}. "
                 f"Учитывались все {result['candidates']} текущих рамок. Источник "+prefix+"result.json.")
        own.update(state="completed_positive" if n==2 else "completed_negative",done="2026-11-24",obtained=message,
             interpretation="Это проверка двух старых номеров на одном заранее найденном конфликте рамок.",
             decision="Судить о переносе только после независимых примеров перекрытия.",
             cannot_claim="Точная пиксельная разметка этих двух масок и результат полного трекера не оценивались.")
        own["acceptance"][0].update(met=n==2,evidence=prefix+"result.json")
        own["evidence"].extend([dict(kind="file",ref=prefix+"result.json"),dict(kind="file",ref=prefix+"masks.jpg")])
    write(path,data)
    journal(message+" Протокол — `"+prefix+"protocol.json`.")


def unique_swap():
    pairs=[]
    for name in ("transformer_rows.json","confirmation_rows.json"):
        frames=defaultdict(list)
        for r in read(PILOTS/name):
            frames[r["video"],r["frame"]].append(r)
        for (video,frame),owners in frames.items():
            for a,b in itertools.combinations(owners,2):
                if min(a["truth"],b["truth"])<0:
                    continue
                if all(a[m]==b["truth"] and b[m]==a["truth"] for m in ("geometry","attention")):
                    pairs.append((video,frame,[a,b]))
    assert len(pairs)==1
    return pairs[0]


def picture(images,boxes,panels,notes):
    from PIL import Image, ImageDraw, ImageFont
    colors=np.array([[35,205,243],[244,137,61],[234,219,91]])
    x0,y0=np.floor(np.asarray(boxes)[:,:2].min(0)-[49,50]).astype(int)
    x1,y1=np.ceil(np.asarray(boxes)[:,2:].max(0)+[49,42]).astype(int)
    x0,y0=max(0,x0),max(0,y0);x1,y1=min(images[0].shape[1],x1),min(images[0].shape[0],y1)
    width=402
    height=round(width*(y1-y0)/(x1-x0))
    result=Image.new("RGB",(width*3+32,height+110),"#152028")
    draw=ImageDraw.Draw(result)
    font_path="C:/Windows/Fonts/arial.ttf"
    title=ImageFont.truetype(font_path,18)
    sub=ImageFont.truetype(font_path,14)
    font_bold=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",18)
    import cv2
    for i,((m1,m2),im) in enumerate(zip(panels,images)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        labels=m1.astype(np.int8)+2*m2.astype(np.int8)
        render=im.copy()
        for label,color in enumerate(colors,1):
            mask=labels==label
            render[mask]=np.round(render[mask]*.52+color*.48).astype("uint8")
            contours,_=cv2.findContours(mask.astype("uint8"),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(render,contours,-1,tuple(map(int,color)),1,lineType=cv2.LINE_AA)
        panel=Image.fromarray(render[y0:y1,x0:x1]).resize((width,height),Image.Resampling.LANCZOS)
        x=i*(width+8)+8
        draw.text((x,8),notes[i][0],font=font_bold,fill="#F4F6F7")
        draw.text((x,32),notes[i][1],font=sub,fill="#CEDAE0")
        result.paste(panel,(x,56))
    x=12
    for s,col in zip(("Машина А","Машина Б","Обе маски"),colors):
        draw.rounded_rectangle((x,height+73,x+17,height+90),3,fill=tuple(col))
        draw.text((x+25,height+71),s,font=title,fill="#F4F6F7")
        x+=180
    result.save(OUT/"masks.jpg",quality=94)


def run():
    import torch,cv2
    import shutil
    sys.path[:0]=[str(ROOT/"tools/sam2-runtime-20261027"),str(ROOT/"tools/sam2-deps-20261027"),
                  str(ROOT/"research_memory_gap_transfer_20261114")]
    from assignment import edges,overlap_matrix
    from sam2.build_sam import build_sam2,build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tracking.gpu_lock import GpuLock
    video,index,owners=unique_swap()
    scene=next(s for s in prepare_scenes(video,False) if s["frame"]==index)
    truth=[r["truth"] for r in owners]
    source=[scene["past"][r["owner"],-1] for r in owners]
    db=read(ROOT/"data/bdd100k/labels/box_track_20/val"/(video+".json"))
    originals=[ROOT/"data/bdd100k/images/track/val"/video/db[f]["name"] for f in (index-1,index)]
    folder=OUT/"frames"
    folder.mkdir(exist_ok=True)
    for i,f in enumerate(originals):
        shutil.copyfile(f,folder/f"{i:05d}.jpg")
    ims=[cv2.imread(str(f)) for f in originals]
    weight=ROOT/"weights/sam2_official_20261027/sam2.1_hiera_tiny.pt"
    inputs=[OUT/"replay.py",OUT/"protocol.json",PILOTS/"transformer_rows.json",PILOTS/"confirmation_rows.json",
            PILOTS/"geometry_data.py",ROOT/"runs/det_cache_ctl_960/val"/(video+".npz"),
            ROOT/"research_memory_gap_transfer_20261114/assignment.py",weight,*originals,
            ROOT/"data/bdd100k/labels/box_track_20/val"/(video+".json"),
            ROOT/"tools/sam2-runtime-20261027/sam2/sam2_video_predictor.py",ROOT/"tools/sam2-runtime-20261027/sam2/modeling/sam2_base.py"]
    provenance=fingerprint(inputs)
    write(OUT/"inputs.json",provenance)
    cfg="configs/sam2.1/sam2.1_hiera_t.yaml"
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with GpuLock(r"Local\DissertationSingleGpu"),torch.inference_mode():
        vp=build_sam2_video_predictor(cfg,str(weight),device="cuda",apply_postprocessing=False)
        image=SAM2ImagePredictor(build_sam2(cfg,str(weight),device="cuda",apply_postprocessing=False))
        past=[]
        predicted=[]
        for bb in source:
            st=vp.init_state(str(folder),offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
            vp.add_new_points_or_box(st,0,1,box=bb)
            ms={int(k):(logits[0,0]>0).cpu().numpy() for k,_,logits in vp.propagate_in_video(st,start_frame_idx=0,max_frame_num_to_track=1)}
            past.append(ms[0]);predicted.append(ms[1])
            del st
        image.set_image(cv2.cvtColor(ims[1],cv2.COLOR_BGR2RGB))
        current=[image.predict(box=bb,multimask_output=False)[0][0].astype(bool) for bb in scene["boxes"]]
    scores=overlap_matrix(predicted,current)
    decisions=dict(edges(scores,[0,1],list(range(len(current)))).pairs)
    two=dict(edges(scores[:,truth],[0,1],truth).pairs)
    outcomes=[decisions.get(i)==j for i,j in enumerate(truth)]
    target=[current[j] for j in truth]
    def pair_overlap(ms):
        return float(overlap_matrix(ms[:1],ms[1:])[0,0])
    assert fingerprint(inputs)["sha256"]==provenance["sha256"]
    for i,line in enumerate(scores):
        for j,v in enumerate(line):
            mutual=(v>.5 and (v>np.delete(line,j)).all() and (v>np.delete(scores[:,j],i)).all())
            assert mutual==(decisions.get(i)==j)
    picture([ims[0],ims[1],ims[1]],source+[scene["boxes"][j] for j in truth],[past,predicted,target],
            [(db[index-1]["name"][-11:-4].lstrip("0")+" · Исходные маски","Два номера до перестановки"),
             (db[index]["name"][-11:-4].lstrip("0")+" · Прогноз SAM 2","Маски из памяти каждой машины"),
             (db[index]["name"][-11:-4].lstrip("0")+" · Текущие маски","Цвет задан по размеченным номерам рамок")])
    np.savez_compressed(OUT/"mask_pixels.npz",past=past,propagated=predicted,current_targets=target)
    result=dict(video=video,source_file_frame=index,target_file_frame=index+1,frame_index=index,
                source_boxes=[v.tolist() for v in source],target_boxes=[scene["boxes"][j].tolist() for j in truth],
                candidate_truth=truth,candidates=len(current),assignment=decisions,only_two=two,
                correct=sum(outcomes),wait=2-len(decisions),wrong=len(decisions)-sum(outcomes),
                scores=scores.tolist(),two_target_columns=scores[:,truth].tolist(),
                owner_to_owner_mask_iou={"past":pair_overlap(past),"propagated":pair_overlap(predicted),"current":pair_overlap(target)},
                masks_sha256=sha(OUT/"mask_pixels.npz"),image_sha256=sha(OUT/"masks.jpg"),complete=now(),inputs_unchanged=True)
    write(OUT/"result.json",result)
    catalog(result)
    print(result["assignment"],result["two_target_columns"],result["owner_to_owner_mask_iou"])


if __name__=="__main__":
    catalog() if sys.argv[1]=="register" else run()
