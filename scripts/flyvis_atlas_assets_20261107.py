"""Export the real inputs, display snapshots and saved feature maps of the five reviewed episodes.

Run from the project root. RGB JPEGs in dist/data/media are local previews of BDD100K.
The scientific arrays remain in the archived runs. No model is run by this exporter.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tracking.flyvis_activity import OUTPUT_CELL_TYPES

OUT = ROOT / "ui/flyvis_atlas/dist/data"
BASE = ROOT / "runs/flyvis_exclusive_t4t5_20261107"
INPUT = ROOT / "runs/z13_temporal_pair_readiness_20261025/inputs"
GEOMETRY = ROOT / "runs/z13_local_peak_20261021/geometry.json"
MAP = "runs/z13_spatial_set_readout_20261025/recovery2/features/C/train26/"
SEED = "runs/z13_region_assignment_20261025/recovery2/features/C/train26/"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, wanted):
    if digest(path) != wanted:
        raise ValueError(f"Source hash mismatch: {path.name}")


def output(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_bytes(content)
    temp.replace(path)


def binding_index(inputs, prefix, target_uids):
    refs = {}
    target_uids = set(target_uids)
    for name, checksum in inputs.items():
        if not (name.startswith(prefix) and name.endswith(".npz")):
            continue
        path = ROOT / name
        with np.load(path, allow_pickle=False) as z:
            hits = [(int(uid), i) for i, uid in enumerate(z["uid"]) if uid in target_uids]
        if hits:
            verify(path, checksum)
            for uid, i in hits:
                assert uid not in refs
                refs[uid] = (path, i)
    assert set(refs) == target_uids
    return refs


def read_map(path, i, uid):
    with np.load(path, allow_pickle=False) as z:
        assert int(z["uid"][i]) == uid
        maps = z["full"][i, :, :34].astype(float).T
        masks = z["retinal_mask"][i, 1:3].copy()
        assert maps.shape == (34, 721) and np.isfinite(maps).all()
    return maps, masks


def frames_with_rgb(video_meta, video, frame_numbers):
    source = ROOT / "data/bdd100k/videos/train" / f"{video}.mov"
    assert source.resolve() == Path(video_meta["path"]).resolve()
    width, height = video_meta["width"], video_meta["height"]
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise OSError(source)
    filenames = {}
    try:
        for number in range(max(frame_numbers) + 1):
            assert cap.grab(), (source, number)
            if number not in frame_numbers:
                continue
            ok, frame = cap.retrieve()
            assert ok and frame.shape[:2] == (height, width)
            image = cv2.resize(frame, (800, round(800*height/width)), interpolation=cv2.INTER_AREA)
            ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            assert ok
            filename = f"media/{video}_{number:04d}.jpg"
            output(OUT / filename, data.tobytes())
            filenames[number] = "data/" + filename
    finally:
        cap.release()
    assert set(filenames) == set(frame_numbers)
    return filenames


def make_episode(case, archive, related, grid, sources, map_index, seed_index):
    uid = case["uid"]
    video = case["video"]
    query = next(row for row in related if row["uid"] == uid)
    retinal_file = ROOT / query["retinal_source"]
    verify(retinal_file, archive["source_sha256"])
    first, end = query["state_prefix_start"], query["state_prefix_end"]
    with np.load(retinal_file, allow_pickle=False) as z:
        assert int(z["ep_offset"][query["episode_local"]]) == first
        assert int(z["ep_tid"][query["episode_local"]]) == query["tid"]
        sel = slice(first, end+1)
        numbers = z["fr_frame"][sel].astype(int).tolist()
        crops = z["fr_crop"][sel].astype(int)
        eyes = z["fr_hex"][sel].astype(float)
    assert eyes.shape == (len(numbers), 721) and eyes.min() >= 0 and eyes.max() <= 1
    assert end == query["current_array_index"] and numbers[-1] == case["cached_frame"]
    assert list(query["current_crop"]) == crops[-1].tolist()
    meta = read(ROOT / "runs/train_cache" / video / "alignment.json")["video"]
    filenames = frames_with_rgb(meta, video, numbers)
    frames = [dict(frame=number, image=filenames[number], crop=crop.tolist(), eye=eye.tolist(), maps=None)
              for number, crop, eye in zip(numbers, crops, eyes, strict=True)]
    for row in related:
        u = row["uid"]
        pos = row["current_array_index"] - first
        assert 0 <= pos < len(frames) and numbers[pos] == row["current_frame"]
        assert row["current_crop"] == crops[pos].tolist()
        maps, masks = read_map(*map_index[u], u)
        frames[pos]["maps"] = maps.tolist()
        if u == uid:
            target_maps, target_masks = maps, masks

    path, ix = seed_index[uid]
    with np.load(path, allow_pickle=False) as z:
        assert int(z["uid"][ix]) == uid
        weights = z["seed"][ix, :34].astype(float)
    annotation = ROOT / "runs/z13_temporal_pair_readiness_20261025/supervision" / f"{video}.json"
    verify(annotation, sources[annotation.relative_to(ROOT).as_posix()])
    truth = next(x for x in read(annotation)["rows"] if x["uid"] == uid)
    correct = truth["current_identity"].index(truth["target_identity"])
    assert truth["flags"]["known"] and truth["current_identity"][0] != truth["current_identity"][1]
    det = ROOT / "runs/train_cache" / video / "detections/raw_video.npz"
    with np.load(det, allow_pickle=False) as z:
        ok = (z["frame_idx"] == numbers[-1]) & (z["conf"] >= .25)
        boxes = z["xyxy"][ok][query["current_choices"]].astype(float)
    c = crops[-1]
    extent = c[2:]-c[:2]
    for i, box in enumerate(boxes):
        # Use the same floating-point order as to_eye() to verify mask binding.
        xy = grid * extent + c[:2]
        inside = np.all((xy >= box[:2]) & (xy <= box[2:]), axis=1)
        assert np.array_equal(np.flatnonzero(inside), query["current_masks"][i])
        assert np.array_equal(inside, target_masks[i])
    score = {}
    a, b = target_masks
    for name, masks in [("whole", (a,b)), ("exclusive", (a & ~b,b & ~a))]:
        vals = [float(target_maps[4:12, mask].mean(axis=1) @ weights[4:12]) for mask in masks]
        key = "all" if name == "whole" else "only"
        for j, val in enumerate(vals):
            ref = case["regions"][("right" if j == correct else "wrong")+"_"+key]["score"]
            assert math.isclose(val, ref, abs_tol=1e-12)
        score[name] = vals
    return dict(uid=uid,video=video,frame=numbers[-1],frames=frames, finalIndex=len(frames)-1,
                masks=query["current_masks"], boxes=boxes.tolist(), originalSize=[meta["width"],meta["height"]],
                correct=correct, scores=score, weights=weights.tolist(), fps=meta["fps"],
                availableMapFrames=[f["frame"] for f in frames if f["maps"] is not None])


def main():
    decomposition = read(BASE/"five_flips/decomposition.json")
    sources = read(BASE/"inputs_used.json")
    verify(BASE/"inputs_used.json", read(BASE/"measurement.json")["inputs_sha256"])
    geo = read(GEOMETRY)
    old = read(ROOT/"runs/flyvis_pair_support_overlap_20261107/recovery1/protocol.json")
    verify(GEOMETRY, old["sha256"][GEOMETRY.relative_to(ROOT).as_posix()])
    grid=np.asarray(geo["points"],float)/geo["eye_size"]
    assert grid.shape == (721,2)
    bundles=[]
    snapshot_uids=[]
    for c in decomposition["cases"]:
        path=INPUT/(c["video"]+".json")
        verify(path, sources[path.relative_to(ROOT).as_posix()])
        archive=read(path)
        row=next(r for r in archive["rows"] if r["uid"]==c["uid"])
        subset=[r for r in archive["rows"] if r["episode_local"]==row["episode_local"]
                and row["state_prefix_start"] <= r["current_array_index"] <= row["current_array_index"]]
        bundles.append((c,archive,subset))
        snapshot_uids.extend(r["uid"] for r in subset)
    maps=binding_index(sources,MAP,snapshot_uids)
    seeds=binding_index(sources,SEED,[c["uid"] for c in decomposition["cases"]])
    episodes=[make_episode(c,a,r,grid,sources,maps,seeds) for c,a,r in bundles]
    payload=dict(channels=list(OUTPUT_CELL_TYPES),points=grid.tolist(),episodes=episodes,
                 coverage="five_historical_T4_T5_flips",eye_size=geo["eye_size"])
    js = ("window.FLY_EPISODES="+json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False)
          +";\nfor(const e of window.FLY_EPISODES.episodes)e.maps=e.frames[e.finalIndex].maps;\n")
    output(OUT/"episodes.js",js.encode("utf-8"))
    print([(e["uid"],len(e["frames"]),e["availableMapFrames"]) for e in episodes])


if __name__ == "__main__":
    main()
