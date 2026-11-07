"""Fresh-image partial-state probe; all choices registered before inference."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse, hashlib, importlib, json, re, shutil, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"runs/flyvis_state_subgroups_20261107"
PREVIOUS=ROOT/"runs/flyvis_partial_state_swap_20261106"
GEOMETRY=ROOT/"runs/flyvis_offset_recovery2_20261104/geometry.npz"
sys.path.insert(0,str(ROOT))
from tracking.flyvis_state_subgroups import GROUPS,LABELS,validate_partition
from scripts.flyvis_history_state_swap_20261106 import sha,save

def journal(text):
    path=ROOT/"dissertation/ЕДИНЫЙ_ПЛАН.md";raw=path.read_bytes();marker="## Журнал изменений".encode()
    nl=b"\r\n" if marker+b"\r\n" in raw else b"\n";assert raw.count(marker+nl)==1
    stamp=datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y, %H:%M")
    path.write_bytes(raw.replace(marker+nl,marker+nl+f"\n**{stamp} MSK** — Узкие группы состояния FlyVis на новых изображениях\n\n{text}\n\n".encode(),1))

def register():
    assert not OUT.exists()
    prior=json.loads((PREVIOUS/"protocol.json").read_text(encoding="utf-8"))
    audit=json.loads((PREVIOUS/"independent_audit.json").read_text(encoding="utf-8"));assert audit["passed"] and audit["cases"]==32
    groups=validate_partition(prior["groups"]["intermediate"])
    manifests=sorted({p for folder in (ROOT/"runs").glob("flyvis*") if folder.is_dir()
                      for pattern in ("**/*protocol*.json","**/*source*image*.json")
                      for p in folder.glob(pattern) if p.is_file()})
    excluded=set()
    for path in manifests:
        excluded.update(re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{8}\b",path.read_text(encoding="utf-8-sig")))
    available=sorted(p.stem for p in (ROOT/"data/bdd100k/videos/train").glob("*.mov"))
    eligible=[v for v in available if v not in excluded]
    videos=sorted(eligible,key=lambda v:hashlib.sha256(("flyvis-state-subgroups-20261107:"+v).encode()).hexdigest())[:8]
    assert len(videos)==8 and not set(videos)&excluded
    cases=[dict(id=f"s{s}_c{ct}_v{v}_a{a}",scene=s,video=video,contrast=ct,speed=v,axis=a,start="still",split="evaluation")
           for s,video in enumerate(videos) for ct in (1.0,.25) for v in (1,4) for a in (0,1)]
    sources=["tracking/flyvis_state_subgroups.py","tracking/flyvis_state_transplant.py",
             "scripts/flyvis_state_subgroups_20261107.py","scripts/audit_flyvis_state_subgroups_20261107.py",
             "scripts/control_flyvis_state_subgroups_20261107.py","scripts/flyvis_history_state_swap_20261106.py",
             "scripts/flyvis_offset_recovery2_20261104.py","scripts/flyvis_y1_latency_fp64_20261104.py",
             "tracking/flyvis_motion.py","tracking/flyvis_adapter.py",
             "tools/flyvis-runtime/flyvis/network/network.py","tools/flyvis-runtime/flyvis/network/dynamics.py",
             "tools/flyvis-runtime/flyvis/task/decoder.py","tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json",
             "runs/flyvis/results/flow/0000/000/chkpts/chkpt_00000"]
    criterion=("Eight primary subgroup comparisons: ordinary-contrast erase-minus-moving latency upper99.375% "
               "scene-cluster bootstrap bound<0 (Bonferroni family8). At both contrasts, reverse and continuing-motion "
               "mean EPE first6/all24 must not increase (numeric tolerance1e-12); reverse missing counts must not "
               "increase, continue correct-sign fraction must not decrease. All required. Parent23 is descriptive "
               "replication outside the primary family. Identity/full-copy states/outputs and selected/unselected "
               "coordinates exact; finite; no weight/buffer mutation; preprocessing<=1e-5; full independent audit.")
    p=dict(id="flyvis_state_subgroups_v1",registered_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        author_request="Давай разделим эти 23 типа клеток на более узкие группы и проверим на новых изображениях. Контроль движения без разворота оставляем, чтобы не потерять полезную память.",
        deadline_msk="2026-11-07T02:30:00+03:00",
        deadline_basis="Saturday: no07:40 workday restriction; separate task cutoff02:30MSK before Monday07:40. Prior completed-run01:00 cutoff unchanged.",
        videos=videos,cases=cases,groups=groups,primary_groups=list(GROUPS),labels=LABELS,
        arms=["moving","stationary","identity_moving","identity_stationary","full_to_stationary","full_to_moving"]+[d+"_"+g for g in groups for d in ("erase","insert")],
        scenarios=["reverse","continue"],criterion=criterion,
        selection=dict(rule="Lowest SHA256('flyvis-state-subgroups-20261107:'+video_id), first frame of8 existing videos; no image-quality/model-response selection",
                       excluded_ids=sorted(excluded),available_count=len(available),eligible_count=len(eligible),
                       scanned_manifest_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in manifests}),
        source_video_sha256={v:sha(ROOT/"data/bdd100k/videos/train"/(v+".mov")) for v in videos},
        prior_protocol_sha256=sha(PREVIOUS/"protocol.json"),prior_audit_sha256=sha(PREVIOUS/"independent_audit.json"),
        parameter_sha=prior["parameter_sha"],geometry_sha256=sha(GEOMETRY),frozen_sources={f:sha(ROOT/f) for f in sources},
        method="Stock flow/0000/000 CPU FP64, hold2 steps/frame, last-step decoder, legacy_v1 geometry, unchanged weights. A:30 still steps then24 forward motion frames. B:boundary frame stationary for equal duration. Fade-in1s both. Swap activity once afterframe54; rebuild aliases. Same24-frame reverse or continued suffix for all24 arms. No interpolation/averaging/calibration/training.",
        grouping_basis="Eight disjoint groups cover23 code entries; CT1(Lo1)/CT1(M10) are two model compartments, not independently identified biological types. Mi1/Mi4/Mi9 separate based on distinct T4 input locations; other groups anatomy/name families, not inferred causal modules. Parent23 replication; choices not based on new outputs.",
        metrics="Central-mask mean vector EPE nominal pixels/frame first6/all24/last6; first3 correct-sign frames, missing explicit/capped24. Paired bootstrap20k seed42 over8 scene means. Primary8 latencyCI99.375%; others95% descriptive. Mean no-harm gate not a statistical equivalence proof.",
        hypotheses="A smaller subgroup may carry transferable delay with less harm. Alternatives: distributed state, normal integration, or off-manifold hybrid disruption. Reciprocal swaps/parent replication constrain but do not eliminate alternatives.",
        decision="One bounded localization round. No passing subgroup: do not tune groups/strength on these images, keep core unchanged and record refusal to implement reset. Passing subgroup: diagnostic candidate for a later realizable causal intervention, no automatic deployment.",
        limitations="Eight images new to this FlyVis diagnostic sequence, not a final tracking test; existing BDD TRAIN, artificial translations, one checkpoint. No detector inference/generalization or HOTA. Unequal cell counts/doses, effects not additive. Artificial stationary donors are not a ready causal implementation. Continued motion safeguards one memory function only.",
        literature=[dict(url="https://elifesciences.org/articles/24394",title="The comprehensive connectome of a neural substrate for ON motion detection in Drosophila",authors="Takemura et al.",year=2017,doi="10.7554/eLife.24394",use="Separate Mi1/Mi4/Mi9 because their inputs target different T4 regions; CT1/C3 also connected. Does not identify our delay cause."),
                    dict(url="https://doi.org/10.1038/s41586-024-07939-3",title="Connectome-constrained networks predict neural activity across the fly visual system",authors="Lappalainen et al.",year=2024,use="Recurrent model and separate decoder; installed official JSON defines exact coordinates.")],
        watchdog_seconds=600,minimum_free_bytes=5_000_000_000)
    OUT.mkdir();save(OUT/"protocol.json",p);save(OUT/"status.json",dict(stage="registered",done=0,total=len(cases)))
    path=ROOT/"experiments/project_history.json";cat=json.loads(path.read_text(encoding="utf-8"));assert not any(c["id"]==p["id"] for c in cat["experiments"])
    cat["experiments"].append(dict(id=p["id"],title="FlyVis — узкие группы памяти на новых изображениях",area="flyvis",state="planned",started="2026-11-07",tags=["Y1","motion-diagnostic"],
        question="Какая подгруппа23 промежуточных обозначений переносит задержку без потери продолжающегося движения?",
        hypothesis=p["hypotheses"],expected="Восемь фиксированных подгрупп, родительская группа и двусторонние контроли на восьми новых изображениях.",
        acceptance=[dict(criterion=criterion,met=None)],done="",obtained="",interpretation="",decision="",cannot_claim="",metrics=[],
        evidence=[dict(kind="json",ref=(OUT/"protocol.json").relative_to(ROOT).as_posix(),claim="Условия и изображения до новых выходов")]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    journal("По прямому поручению автора зарегистрирован отдельный опыт: восемь непересекающихся подгрупп23 промежуточных обозначений модели, обратный перенос, родительская группа и контроль продолжающегося движения. Восемь новых для серии изображений выбраны по фиксированному хешу без оценки качества;64условия. Веса неизменны. Протокол: runs/flyvis_state_subgroups_20261107/protocol.json. До создания файлов была отклонена команда записи из-за синтаксиса строкового литерала инструмента; она ничего не исполнила, научных результатов до исправления не было.")
    print(json.dumps(dict(registered=True,videos=videos,groups=groups,cases=len(cases)),ensure_ascii=False))

def run():
    import cv2,torch
    from tracking.flyvis_adapter import activate_flyvis
    from tracking.flyvis_state_transplant import transplant_nodes
    from scripts.flyvis_y1_latency_fp64_20261104 import parameter_sha,cpu_eye
    from scripts.flyvis_offset_recovery2_20261104 import frames_for
    p=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    for f,h in p["frozen_sources"].items():assert sha(ROOT/f)==h,f
    assert sha(GEOMETRY)==p["geometry_sha256"]
    fv=activate_flyvis()
    for m in (fv,importlib.import_module("flyvis.network.initialization"),importlib.import_module("flyvis.task.decoder")):m.device=torch.device("cpu")
    torch.set_default_device("cpu");torch.set_default_dtype(torch.float64);torch.set_num_threads(2);cv2.setNumThreads(2)
    from tracking.flyvis_motion import FlyVisMotionEstimator
    model=FlyVisMotionEstimator(mode="full",source_fps=30,flow_gain=1,temporal_resampling="hold",subtract_initial_baseline=False,geometry_version="legacy_v1")
    model.network.double();model.decoder.double();model.eye.conv.double()
    net,dec,eye=model.network,model.decoder,model.eye;net.eval().requires_grad_(False);dec.eval().requires_grad_(False);net.clamp()
    assert not net._state_hooks;frozen=parameter_sha(net,dec);assert frozen==p["parameter_sha"]
    def buffers():return {k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in dec.named_buffers()}
    frozen_buffers=buffers()
    ix={g:np.concatenate([np.asarray(net.stimulus.layer_index[n]) for n in names]) for g,names in p["groups"].items()}
    assert np.array_equal(np.sort(np.concatenate([ix[g] for g in GROUPS])),np.sort(ix["parent"]))
    assert len(np.unique(ix["parent"]))==len(ix["parent"]);np.savez_compressed(OUT/"indices.npz",**ix)
    node_types=np.asarray(net.connectome.nodes.type[:]).astype("U")
    assert node_types.shape==(net.n_nodes,)
    np.save(OUT/"node_types.npy",node_types)
    save(OUT/"model_loaded.json",dict(parameter_sha=frozen,n_nodes=net.n_nodes,group_nodes={g:len(v) for g,v in ix.items()},decoder_buffer_sha=frozen_buffers))
    geo=np.load(GEOMETRY);assert np.array_equal(eye.receptor_centers.cpu().numpy(),geo["centers"])
    textures={};sources=[]
    for scene,video in enumerate(p["videos"]):
        source=ROOT/"data/bdd100k/videos/train"/(video+".mov");assert sha(source)==p["source_video_sha256"][video]
        cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert ok
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);assert gray.shape[0]>=704 and gray.shape[1]>=896;textures[scene]=gray
        sources.append(dict(video=video,frame=0,shape=list(gray.shape),source_sha256=sha(source),frame_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),fps=fps))
    save(OUT/"source_images.json",dict(sources=sources));np.savez_compressed(OUT/"textures.npz",**{str(k):v for k,v in textures.items()})
    started=time.monotonic();total=len(p["cases"]);arms=p["arms"]
    for ci,c in enumerate(p["cases"]):
        assert datetime.now(ZoneInfo("Europe/Moscow"))<datetime.fromisoformat(p["deadline_msk"])
        assert shutil.disk_usage(OUT).free>=p["minimum_free_bytes"]
        dest=OUT/"cases"/(c["id"]+".npz");marker=dest.with_suffix(".json")
        if marker.exists():
            m=json.loads(marker.read_text());assert sha(dest)==m["sha256"] and m["protocol_sha256"]==sha(OUT/"protocol.json");continue
        begin=time.monotonic();gray=textures[c["scene"]];frames=frames_for(gray,c)
        h,w=gray.shape;top=h-512 if c["axis"]==1 and c["speed"]==4 else (h-512)//2;left=(w-512)//2;continuing=[]
        for t in range(25,49):
            offset=c["speed"]*t;y=top-(offset if c["axis"]==1 else 0);x=left-(offset if c["axis"]==0 else 0)
            assert min(x,y)>=0 and x+512<=w and y+512<=h
            continuing.append(np.rint(np.clip(127.5+c["contrast"]*(gray[y:y+512,x:x+512].astype(np.float32)-127.5),0,255)).astype(np.uint8))
        all_frames=np.concatenate([frames,np.stack(continuing)])
        with torch.inference_mode():
            cart=np.stack([cv2.resize(f,(391,391),interpolation=cv2.INTER_AREA) for f in all_frames])
            hexes=eye(torch.from_numpy(cart.astype(np.float64)/255)[None])[0,:,0].cpu().numpy()
        eye_error=float(np.max(np.abs(hexes-cpu_eye(all_frames,geo["centers"]))));assert eye_error<=1e-5
        with torch.inference_mode():
            history=net.fade_in_state(1.,model.dt,torch.from_numpy(np.stack([hexes[0],hexes[54]]))[:,None])
            for j in range(1,55):
                inp=torch.from_numpy(np.stack([hexes[j],hexes[54]]))[:,None,None].expand(2,2,1,721)
                history=net.simulate(inp,model.dt,initial_state=history,as_states=True)[-1]
                if j%15==0:save(OUT/"status.json",dict(stage="history",done=ci,total=total,frame=j,case=c["id"],elapsed_seconds=time.monotonic()-started))
            assert list(history.nodes)==["activity"] and not list(history.edges)
            boundary=history.nodes.activity.detach().clone();a,b=boundary
            initial=[a.clone(),b.clone(),transplant_nodes(a,a,ix["parent"]),transplant_nodes(b,b,ix["parent"]),
                     transplant_nodes(a,b,np.arange(net.n_nodes)),transplant_nodes(b,a,np.arange(net.n_nodes))]
            for g in p["groups"]:initial+=[transplant_nodes(a,b,ix[g]),transplant_nodes(b,a,ix[g])]
            initial=torch.stack(initial);before=initial.cpu().numpy().copy()
            flows=np.empty((2,len(arms),24,721,2),np.float64);exact=True
            for si,suffix in enumerate([hexes[55:79],hexes[79:]]):
                state=net._initial_state(net._param_api(),len(arms));state.nodes.activity=initial.clone();state=net._state_api(state)
                for j in range(24):
                    seq=net.simulate(torch.from_numpy(suffix[j])[None,None,None].expand(len(arms),2,1,721),model.dt,initial_state=state,as_states=True)
                    for s in seq:
                        assert torch.isfinite(s.nodes.activity).all()
                        exact &= all(torch.equal(s.nodes.activity[u],s.nodes.activity[v]) for u,v in [(0,2),(1,3),(0,5),(1,4)])
                    state=seq[-1];flows[si,:,j]=dec(state.nodes.activity[:,None])[:,0].cpu().numpy().transpose(0,2,1)*np.array([512/169,-512/169])
                    if j%6==0:save(OUT/"status.json",dict(stage="suffix",done=ci,total=total,scenario=p["scenarios"][si],frame=j,case=c["id"],elapsed_seconds=time.monotonic()-started))
        assert exact and np.isfinite(flows).all()
        assert torch.equal(boundary,history.nodes.activity) and np.array_equal(before,initial.cpu().numpy())
        assert parameter_sha(net,dec)==frozen and buffers()==frozen_buffers and not torch.cuda.is_initialized()
        dest.parent.mkdir(exist_ok=True)
        np.savez_compressed(dest,flow=flows,boundary_activity=boundary.cpu().numpy(),initial_states=before,hexes=hexes,
                            frame_sha256=np.asarray([hashlib.sha256(f.tobytes()).hexdigest() for f in all_frames]))
        save(marker,dict(sha256=sha(dest),protocol_sha256=sha(OUT/"protocol.json"),eye_max_diff=eye_error,identity_and_full_states_exact=exact,
                         donors_unchanged=True,parameters_and_buffers_unchanged=True,seconds=time.monotonic()-begin))
        if ci==0:
            required=2*dest.stat().st_size*(total-1)+p["minimum_free_bytes"];free=shutil.disk_usage(OUT).free;assert free>=required
            save(OUT/"first_case_cost.json",dict(seconds=time.monotonic()-begin,remaining=total-1,forecast_seconds=(total-1)*(time.monotonic()-begin),disk_required=required,disk_free=free,disk_passed=True))
        save(OUT/"status.json",dict(stage="case_complete",done=ci+1,total=total,elapsed_seconds=time.monotonic()-started))
    save(OUT/"inference_completed.json",dict(complete=True,cases=total,protocol_sha256=sha(OUT/"protocol.json"),parameter_sha=frozen,
         indices_sha256=sha(OUT/"indices.npz"),textures_sha256=sha(OUT/"textures.npz"),source_images_sha256=sha(OUT/"source_images.json"),
         node_types_sha256=sha(OUT/"node_types.npy"),
         seconds=time.monotonic()-started,cuda_initialized=torch.cuda.is_initialized()))

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["register","run"]);arg=ap.parse_args()
    register() if arg.stage=="register" else run()
