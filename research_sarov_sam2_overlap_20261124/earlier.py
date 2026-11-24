from pathlib import Path
import shutil
import sys

import numpy as np

from replay import OUT, PILOTS, ROOT, fingerprint, journal, now, read, sha, unique_swap, write
from geometry_data import prepare_scenes


PROTOCOL = OUT / "earlier_protocol.json"
PREFIX = OUT.relative_to(ROOT).as_posix() + "/"


def record(result=None):
    spec = read(PROTOCOL)
    catalog_path = ROOT / "experiments/project_history.json"
    catalog = read(catalog_path)
    cards = catalog["experiments"]
    card = next((c for c in cards if c["id"] == spec["id"]), None)
    if result is None:
        assert card is None
        cards.append(dict(
            id=spec["id"], title=spec["title"], area="own_tracker", state="running",
            started=now()[:10], question=spec["question"], hypothesis=spec["change"],
            expected="Обе машины сохранят номера при более раннем начале памяти.",
            acceptance=[dict(criterion=spec["criterion"], met=None)],
            evidence=[dict(kind="file", ref=PREFIX + PROTOCOL.name)],
        ))
        message = "Для двух машин зарегистрировано сравнение начала памяти SAM 2 на кадрах 170 и 171 при общем кадре оценки 172."
    else:
        assert card and card["state"] == "running"
        counts = result["earlier"]["outcomes"]
        complete = counts["correct"] == 2
        message = (f"На кадре 172 контроль с началом на кадре 171 воспроизведён по пикселям и всем 28 оценкам. "
                   f"При начале на кадре 170 правильных назначений {counts['correct']} из 2, "
                   f"ожиданий {counts['wait']}, ошибочных назначений {counts['wrong']}.")
        card.update(state="completed_positive" if complete else "completed_negative", done=now()[:10],
                    obtained=message + " Источник " + PREFIX + "earlier_result.json.",
                    interpretation="Сравнение измеряет эффект выбора предыдущего момента инициализации на одном эпизоде.",
                    decision="Отдельно проверять перенос на других перекрытиях.",
                    cannot_claim="Нет оценки точности пиксельных границ и переноса на новые видео.")
        card["acceptance"][0].update(met=complete, evidence=PREFIX + "earlier_result.json")
        card["evidence"].append(dict(kind="file", ref=PREFIX + "earlier_result.json"))
    write(catalog_path, catalog)
    journal(message + " Протокол — `" + PREFIX + PROTOCOL.name + "`.")


def run():
    import cv2
    import torch

    runtime = ROOT / "tools/sam2-runtime-20261027"
    sys.path[:0] = [str(runtime), str(ROOT / "tools/sam2-deps-20261027"),
                    str(ROOT / "research_memory_gap_transfer_20261114")]
    from assignment import edges, overlap_matrix
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from tracking.gpu_lock import GpuLock

    video, index, owners = unique_swap()
    reference = read(OUT / "result.json")
    original_inputs = read(OUT / "inputs.json")["sha256"]
    assert video == reference["video"] and index == reference["frame_index"] == 171
    sce = next(s for s in prepare_scenes(video, False) if s["frame"] == index)
    truth = [row["truth"] for row in owners]
    assert truth == reference["candidate_truth"]
    history = sce["past"][[o["owner"] for o in owners]]
    assert np.array_equal(history[:, -1], reference["source_boxes"])
    labels = read(ROOT / "data/bdd100k/labels/box_track_20/val" / (video + ".json"))
    images = [ROOT / "data/bdd100k/images/track/val" / video / labels[j]["name"]
              for j in range(index - 2, index + 1)]
    cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    weight = ROOT / "weights/sam2_official_20261027/sam2.1_hiera_tiny.pt"
    files = ([ROOT / name for name in original_inputs] + images +
             [Path(__file__), PROTOCOL, OUT / "inputs.json", OUT / "result.json", OUT / "mask_pixels.npz",
              runtime / "sam2" / cfg, ROOT/"tracking/gpu_lock.py", PILOTS/"lab.py"])
    provenance = fingerprint(files)
    assert all(provenance["sha256"][name] == digest for name, digest in original_inputs.items())
    write(OUT / "earlier_inputs.json", provenance)
    clips = {}
    for arm, start in [("control", 1), ("earlier", 0)]:
        folder = OUT / ("frames" if arm == "control" else "earlier_frames")
        folder.mkdir(exist_ok=True)
        for offset, image in enumerate(images[start:]):
            dest = folder / f"{offset:05d}.jpg"
            if not dest.exists():
                shutil.copyfile(image, dest)
            assert sha(dest) == sha(image)
        assert len(list(folder.glob("*.jpg"))) == len(images) - start
        clips[arm] = folder
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    archive = {}
    arms = {}
    with GpuLock(r"Local\DissertationSingleGpu"), torch.inference_mode():
        predictor = build_sam2_video_predictor(cfg, str(weight), device="cuda", apply_postprocessing=False)
        segmenter = SAM2ImagePredictor(build_sam2(cfg, str(weight), device="cuda", apply_postprocessing=False))
        segmenter.set_image(cv2.cvtColor(cv2.imread(str(images[-1])), cv2.COLOR_BGR2RGB))
        current = np.stack([segmenter.predict(box=box, multimask_output=False)[0][0].astype(bool)
                            for box in sce["boxes"]])
        archive["current"] = current
        for arm, start in [("control", 1), ("earlier", 0)]:
            masks = []
            for box in history[:, start - 2]:
                state = predictor.init_state(str(clips[arm]), offload_video_to_cpu=True, offload_state_to_cpu=False,
                                             async_loading_frames=False)
                predictor.add_new_points_or_box(state, 0, 1, box=box)
                seq = {int(k): (logits[0, 0] > 0).cpu().numpy() for k, _, logits in predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=2-start)}
                assert set(seq) == set(range(3-start))
                masks.append(np.stack([seq[k] for k in range(3-start)]))
                del state
            masks = np.stack(masks)
            archive[arm] = masks
            scores = overlap_matrix(masks[:, -1], current)
            assigned = dict(edges(scores, [0, 1], list(range(len(current)))).pairs)
            correct = sum(assigned.get(i) == j for i, j in enumerate(truth))
            pair_iou = [float(overlap_matrix(masks[:1,k], masks[1:,k])[0,0]) for k in range(masks.shape[1])]
            arms[arm] = dict(
                seed_frame=start + 170, seed_boxes=history[:,start-2].tolist(), owner_iou_each_frame=pair_iou,
                assignment=assigned, scores=scores.tolist(),
                outcomes=dict(correct=correct, wait=2-len(assigned), wrong=len(assigned)-correct))
    with np.load(OUT / "mask_pixels.npz") as previous:
        assert np.array_equal(archive["control"][:,0], previous["past"])
        assert np.array_equal(archive["control"][:,1], previous["propagated"])
        assert np.array_equal(archive["current"][truth], previous["current_targets"])
    assert np.array_equal(arms["control"]["scores"], reference["scores"])
    assert fingerprint(files)["sha256"] == provenance["sha256"]
    pixels = OUT / "earlier_pixels.npz"
    np.savez_compressed(pixels, **archive)
    result = dict(video=video, target_frame=172, correct_candidates=truth, candidates=len(current),
                  **arms, control_pixelwise_exact=True, control_all_scores_exact=True,
                  inputs_unchanged=True, pixels_sha256=sha(pixels), ended=now())
    write(OUT / "earlier_result.json", result)
    record(result)
    print({name: (arm["outcomes"],arm["owner_iou_each_frame"]) for name,arm in arms.items()})


if __name__ == "__main__":
    if sys.argv[1:] == ["register"]:
        record()
    elif sys.argv[1:] == ["run"]:
        run()
