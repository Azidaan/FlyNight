from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import shutil
import sys

import numpy as np
from scipy.stats import binomtest


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PILOTS = ROOT / "research_sarov_pilots_20261124"
sys.path.insert(0, str(PILOTS))
from geometry_data import prepare_scenes, ious, sources, LABEL, VAL
from lab import fingerprint, now, read, sha, write
from register import journal


PROTOCOL = OUT / "protocol.json"
REF = OUT.relative_to(ROOT).as_posix()
IMAGE_ROOT = ROOT / "data/bdd100k/images/track/val"
KNOWN = "b242f6b2-f5da110f"
SAM = ROOT / "tools/sam2-runtime-20261027"
ASSIGN = ROOT / "research_memory_gap_transfer_20261114"
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"
WEIGHT = ROOT / "weights/sam2_official_20261027/sam2.1_hiera_tiny.pt"
EXCLUSIONS = [PILOTS / name for name in ("transformer_rows.json", "confirmation_rows.json")]


def record(result=None):
    spec = read(PROTOCOL)
    path = ROOT / "experiments/project_history.json"
    data = read(path)
    cards = data["experiments"]
    card = next((c for c in cards if c["id"] == spec["id"]), None)
    if result is None:
        assert card is None
        cards.append(dict(
            id=spec["id"], area="own_tracker", state="running", started=now()[:10],
            title=spec["title"], question=spec["question"], hypothesis=spec["method"],
            expected="Ранний старт сохранит больше пар без роста ошибочных выдач.",
            acceptance=[dict(criterion=spec["criterion"], met=None)],
            evidence=[dict(kind="file", ref=REF + "/protocol.json")],
        ))
        message = "Зарегистрировано сравнение раннего и обычного старта SAM 2 на отдельных видео с перекрытием рамок."
        source = "/protocol.json"
    else:
        assert card and card["state"] == "running"
        a, b = [result["summary"][k] for k in ("late", "early")]
        met = result["summary"]["met"]
        n = result["summary"]["videos"]
        message = (f"SAM 2 на {n} отдельных видео с перекрытием. Полностью правильных пар "
                   f"{a['complete_pairs']} → {b['complete_pairs']}, ошибочных назначений "
                   f"{a['wrong']} → {b['wrong']}, ожиданий {a['wait']} → {b['wait']}. "
                   f"Точный парный p = {result['summary']['paired_p']:.4g}.")
        source = "/result.json"
        card.update(state="completed_positive" if met else "completed_negative", done=now()[:10],
                    obtained=message + " Источник — " + REF + source,
                    interpretation="Эффект раннего старта измерен на одном переходе каждого видео при известной истории.",
                    decision="Изучить разницу в наблюдаемости и ошибки до выбора режима инициализации.",
                    cannot_claim=spec["scope"])
        card["evidence"].append(dict(kind="file", ref=REF + source))
        card["acceptance"][0].update(met=met, evidence=REF + source)
    write(path, data)
    journal(message + " Источник — `" + REF + source + "`.")


def pack(scene, owners, growth=None):
    return dict(video=scene["video"], frame_index=scene["frame"], owners=list(owners),
                past=scene["past"][list(owners), -2:].tolist(), boxes=scene["boxes"].tolist(),
                truth=scene["y"][list(owners)].tolist(), growth=growth)


def select():
    excluded = {r["video"] for f in EXCLUSIONS for r in read(f)}
    assert len(excluded) == 40 and KNOWN in excluded
    videos = sorted((p.stem for p in (VAL / "val").glob("*.npz") if p.stem not in excluded),
                    key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    cases = []
    scanned = []
    declarations = [PROTOCOL, Path(__file__), PILOTS / "geometry_data.py", *EXCLUSIONS]
    for video in videos:
        scanned.append(video)
        declarations.extend(sources(video, False))
        best = None
        for scene in prepare_scenes(video, False):
            past = scene["past"]
            early = ious(past[:, -2], past[:, -2])
            late = ious(past[:, -1], past[:, -1])
            for a, b in combinations(range(len(past)), 2):
                gain = float(late[a, b] - early[a, b])
                if gain <= 0:
                    continue
                score = (gain, float(late[a, b]), -scene["frame"], -a, -b)
                if best is None or score > best[0]:
                    best = (score, pack(scene, [a, b], [float(early[a, b]), float(late[a, b])]))
        if best:
            cases.append(best[1])
        if len(cases) == 20:
            break
    scene = next(s for s in prepare_scenes(KNOWN, False) if s["frame"] == 171)
    owners = [int(np.flatnonzero(scene["y"] == j).item()) for j in (0, 4, 2)]
    three = pack(scene, owners)
    declarations.extend(sources(KNOWN, False))
    write(OUT / "selection.json", dict(excluded=sorted(excluded), scanned=scanned,
                                     cases=cases, three_car_diagnostic=three))
    write(OUT / "selection_inputs.json", fingerprint(declarations))
    print("Selected", len(cases), "from", len(scanned), "videos; reference candidates", three["truth"], flush=True)


def score_case(ctx, case):
    cv2, predictor, segmenter, edges, overlap_matrix = ctx
    video, index = case["video"], case["frame_index"]
    label = read(LABEL / "val" / (video + ".json"))
    paths = [IMAGE_ROOT / video / label[i]["name"] for i in range(index - 2, index + 1)]
    clips = {}
    for arm, start in (("late", 1), ("early", 0)):
        folder = OUT / "clips" / video / arm
        folder.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(paths[start:]):
            dst = folder / f"{i:05d}.jpg"
            if not dst.exists():
                shutil.copyfile(img, dst)
            assert sha(img) == sha(dst)
        assert len(list(folder.glob("*.jpg"))) == 3-start
        clips[arm] = folder
    boxes = np.array(case["boxes"], dtype=np.float32).reshape(-1, 4)
    frame = cv2.imread(str(paths[-1]))
    segmenter.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current = [segmenter.predict(box=b, multimask_output=False)[0][0].astype(bool) for b in boxes]
    if not current:
        current = np.empty((0, *frame.shape[:2]), dtype=bool)
    arms = {}
    pixels = {"current": np.asarray(current)}
    n = len(case["owners"])
    for arm, start in (("late", 1), ("early", 0)):
        histories = []
        for box in np.array(case["past"], dtype=np.float32)[:, start]:
            state = predictor.init_state(str(clips[arm]), offload_video_to_cpu=True, offload_state_to_cpu=False,
                                         async_loading_frames=False)
            predictor.add_new_points_or_box(state, 0, 1, box=box)
            masks = [logits[0, 0].gt(0).cpu().numpy() for _, _, logits in predictor.propagate_in_video(
                state, start_frame_idx=0, max_frame_num_to_track=2-start)]
            assert len(masks) == 3-start
            histories.append(masks)
        stacks = np.asarray(histories)
        pixels[arm] = stacks
        scores = overlap_matrix(stacks[:, -1], current)
        assigned = dict(edges(scores, list(range(n)), list(range(len(current)))).pairs)
        summary = Counter()
        outcomes = []
        for i, target in enumerate(case["truth"]):
            selected = assigned.get(i)
            outcome = "wait" if selected is None else "correct" if selected == target else "wrong"
            outcomes.append(outcome)
            summary[outcome] += 1
            summary["wrong_without_candidate"] += int(target < 0 and outcome == "wrong")
        arms[arm] = dict(scores=scores.tolist(), assigned=assigned, outcomes=outcomes,
                         chosen=[assigned.get(i) for i in range(n)],
                         counts={k:summary[k] for k in ("correct", "wait", "wrong", "wrong_without_candidate")},
                         foreground_sizes=stacks.sum(axis=(-1,-2)).tolist(),
                         overlap_between_owners=overlap_matrix(stacks[:, -1], stacks[:, -1]).tolist())
    directory = OUT / "masks"
    directory.mkdir(exist_ok=True)
    archive = directory / (video + ".npz")
    np.savez_compressed(archive, **pixels)
    return dict(**case, images=[p.relative_to(ROOT).as_posix() for p in paths], arms=arms,
                mask_archive=archive.relative_to(ROOT).as_posix(), pixel_sha256=sha(archive))


def summarize(results):
    totals = {k: Counter() for k in ("late", "early")}
    wins = losses = evaluable = absent = 0
    for row in results:
        evaluable += int(all(t >= 0 for t in row["truth"]))
        absent += sum(t < 0 for t in row["truth"])
        ok = {}
        for arm, total in totals.items():
            counts = row["arms"][arm]["counts"]
            total.update(counts)
            ok[arm] = counts["correct"] == 2
            total["complete_pairs"] += int(ok[arm])
        wins += int(ok["early"] and not ok["late"])
        losses += int(ok["late"] and not ok["early"])
    p = float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.
    met = wins > losses and p < .05 and totals["early"]["wrong"] <= totals["late"]["wrong"]
    return dict(videos=len(results), both_have_candidates=evaluable, owners_missing_candidate=absent,
                wins=wins, regressions=losses, paired_p=p, met=met, **totals)


def run():
    import cv2
    import torch

    sys.path[:0] = [str(SAM), str(ROOT / "tools/sam2-deps-20261027"), str(ASSIGN)]
    from assignment import edges, overlap_matrix
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tracking.gpu_lock import GpuLock

    selection = read(OUT / "selection.json")
    cases = [*selection["cases"], selection["three_car_diagnostic"]]
    selection_inputs = read(OUT / "selection_inputs.json")["sha256"]
    inputs = [ROOT/path for path in selection_inputs]
    inputs += [OUT/"selection.json", SAM/"sam2"/CFG, WEIGHT, ASSIGN/"assignment.py", PILOTS/"lab.py",
               SAM/"sam2/sam2_video_predictor.py", SAM/"sam2/modeling/sam2_base.py", ROOT/"tracking/gpu_lock.py"]
    for c in cases:
        clip_labels = read(LABEL / "val" / (c["video"] + ".json"))
        inputs.extend(IMAGE_ROOT / c["video"] / clip_labels[i]["name"]
                      for i in range(c["frame_index"]-2, c["frame_index"]+1))
    fp = fingerprint(inputs)
    assert all(fp["sha256"][name] == digest for name, digest in selection_inputs.items())
    write(OUT/"inputs.json", fp)
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    with GpuLock(r"Local\DissertationSingleGpu"), torch.inference_mode():
        predictor = build_sam2_video_predictor(CFG, str(WEIGHT), device="cuda", apply_postprocessing=False)
        image = SAM2ImagePredictor(build_sam2(CFG, str(WEIGHT), device="cuda", apply_postprocessing=False))
        for i, case in enumerate(cases, 1):
            results.append(score_case((cv2, predictor, image, edges, overlap_matrix), case))
            print(i, case["video"], [v["counts"] for v in results[-1]["arms"].values()], flush=True)
    assert fingerprint(inputs)["sha256"] == fp["sha256"]
    summary = summarize(results[:-1])
    result = dict(summary=summary, cases=results[:-1], three_car_diagnostic=results[-1],
                  inputs_unchanged=True, finished=now())
    write(OUT/"result.json", result)
    print(summary)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "register":
        record()
    elif command == "select":
        select()
    elif command == "run":
        run()
    elif command == "record":
        record(read(OUT / "result.json"))
