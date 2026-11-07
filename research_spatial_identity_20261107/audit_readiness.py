"""Read archived inputs and bindings; never run a model or compute new rankings."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PAIR = ROOT / "runs/z13_temporal_pair_readiness_20261025"
SPATIAL = ROOT / "runs/z13_spatial_set_readout_20261025/recovery2/features"
CHOICES = ROOT / "runs/flyvis_exclusive_t4t5_20261107/choices.csv.gz"
FOLDERS = {"fit84": "fit84", "fit20": "fit20", "evaluate26": "train26"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def save_new(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    start = time.perf_counter()
    assert not (OUT / "readiness_report.json").exists(), "Preserve the completed audit"
    protected = [ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md", ROOT / "dissertation/План целей по Трекеру.md",
                 ROOT / "experiments/project_history.json", ROOT / "experiments/metrics_glossary.json",
                 ROOT / "ObsidianFlyVis/.graph-visible.json"]
    before = {rel(p): digest(p) for p in protected}
    source_hashes = {}

    def capture(path):
        key = rel(path)
        if key not in source_hashes:
            source_hashes[key] = digest(path)
        return source_hashes[key]

    for path in [Path(__file__), CHOICES, PAIR / "protocol.json", SPATIAL / "report.json",
                 ROOT / "runs/z13_local_peak_20261021/geometry.json",
                 ROOT / "runs/z13_target_center_dev_20261018/C/normalization.json",
                 ROOT / "scripts/z13_data.py", ROOT / "scripts/z13_spatial_set_readout_20261025.py",
                 ROOT / "tracking/z13_readout.py", ROOT / "tracking/foveated_eyes.py"]:
        capture(path)
    prior = read(ROOT / "runs/flyvis_exclusive_t4t5_20261107/measurement.json")
    assert capture(CHOICES) == prior["rows_sha256"]
    fp = read(ROOT / "runs/flyvis_exclusive_t4t5_20261107/inputs_used.json")
    spatial_report = read(SPATIAL / "report.json")
    pair_protocol = read(PAIR / "protocol.json")
    with gzip.open(CHOICES, "rt", encoding="ascii", newline="") as stream:
        cohort = [r for r in csv.DictReader(stream)
                  if r["exactly2"] == "1" and r["label_reason"] == "ok"]
    assert len({int(r["uid"]) for r in cohort}) == len(cohort)
    map_locations = {}
    chunk_headers = []
    for split, folder in FOLDERS.items():
        paths = sorted((SPATIAL / "C" / folder).glob("chunk*.npz"))
        assert paths
        for path in paths:
            marker = path.with_suffix(".json")
            assert capture(marker) == spatial_report["manifest"][rel(marker)]
            metadata = read(marker)
            with np.load(path, allow_pickle=False) as arrays:
                uids = arrays["uid"]
                keys = arrays.files
                for index, uid in enumerate(uids):
                    uid = int(uid)
                    assert uid not in map_locations
                    map_locations[uid] = (path, index, split)
            chunk_headers.append({"path": rel(path), "members": keys,
                                  "rows": len(uids), "archived_sha256": metadata["sha256"]})
        print("Indexed archived current maps", split, len(paths), flush=True)

    grouped = {}
    for record in cohort:
        grouped.setdefault(record["video"], []).append(record)
    totals = {split: Counter() for split in FOLDERS}
    rows = []
    verified_maps = set()
    archive_schemas = {}
    metadata_by_video = {}
    for number, (video, records) in enumerate(sorted(grouped.items()), 1):
        path = PAIR / "inputs" / (video + ".json")
        assert capture(path) == fp[rel(path)]
        meta = read(path)["rows"]
        by_uid = {int(r["uid"]): r for r in meta}
        by_index = {(r["episode_local"], r["current_array_index"]): r for r in meta}
        raw = ROOT / "runs/z13_data" / (video + ".npz")
        assert capture(raw) == pair_protocol["sha256"][rel(raw)]
        with np.load(raw, allow_pickle=False) as arrays:
            archive_schemas[video] = arrays.files
            offsets = arrays["ep_offset"]
            frames = arrays["fr_frame"]
            crops = arrays["fr_crop"]
            tids = arrays["ep_tid"]
            perturbed = arrays["ep_pert"]
            exmask = arrays["ex_mask"]
            # Retinal input values only; this is a data integrity check, not inference.
            retina = arrays["fr_hex"]
            assert retina.shape == (len(frames), 721) and retina.dtype == np.float16
            assert np.isfinite(retina).all()
            for record in records:
                uid = int(record["uid"])
                m = by_uid[uid]
                split = record["split"]
                counter = totals[split]
                counter["pairs"] += 1
                assert m["video"] == video and m["split"] == split
                assert m["current_choices"] == [int(record["j"]), int(record["k"])]
                episode = m["episode_local"]
                first, stop = map(int, offsets[episode:episode + 2])
                current = int(m["current_array_index"])
                assert not perturbed[episode] and int(tids[episode]) == m["tid"]
                assert first + 4 <= current < stop and int(frames[current]) == m["current_frame"]
                assert np.array_equal(crops[current], m["current_crop"])
                path_map, map_index, map_split = map_locations[uid]
                assert map_split == split
                if path_map not in verified_maps:
                    assert capture(path_map) == read(path_map.with_suffix(".json"))["sha256"]
                    verified_maps.add(path_map)
                counter["current_spatial_map"] += 1
                counter["four_observed_retinal_inputs"] += 1
                last_observed = first + 3
                assert int(frames[last_observed]) < int(frames[current])
                observed_meta = by_index.get((episode, last_observed))
                observed_map_uid = int(observed_meta["uid"]) if observed_meta else None
                has_observed_map = observed_map_uid in map_locations
                counter["last_observed_spatial_map"] += int(has_observed_map)
                counter["last_observed_support_mask"] += int(exmask[episode].any())
                p = m["previous"]
                previous_uid = None
                if p is not None:
                    counter["previous_candidate_metadata"] += 1
                    previous = int(p["array_index"])
                    assert first <= previous < current and int(frames[previous]) == p["frame"]
                    pm = by_index.get((episode, previous))
                    previous_uid = int(pm["uid"]) if pm else None
                    counter["previous_candidate_spatial_map"] += int(previous_uid in map_locations)
                    counter["previous_in_observed_prefix"] += int(previous < first + 4)
                    counter["previous_in_search_prefix"] += int(previous >= first + 4)
                    counter["previous_has_multiple_candidates"] += int(len(p["choices"]) > 1)
                rows.append({"uid": uid, "video": video, "split": split, "episode": episode,
                             "current_frame": int(frames[current]), "current_array_index": current,
                             "current_map": [rel(path_map), map_index],
                             "last_observed_frame": int(frames[last_observed]),
                             "last_observed_array_index": last_observed,
                             "last_observed_map_uid": observed_map_uid if has_observed_map else None,
                             "observed_mask_cells": int(exmask[episode].sum()),
                             "previous_candidate_map_uid": previous_uid if previous_uid in map_locations else None,
                             "previous_is_reliable_target": "not established by candidate availability",
                             "detector_candidates": m["current_choices"]})
        metadata_by_video[video] = len(meta)
        if number % 10 == 0:
            print("Audited videos", number, "/", len(grouped), flush=True)

    schema_union = sorted({key for schema in archive_schemas.values() for key in schema})
    unavailable_fields = [name for name in ["observed_box_xyxy", "track_mean_xywh",
                          "track_covariance", "native_assignment_detection_id"] if name not in schema_union]
    after = {rel(p): digest(p) for p in protected}
    save_new("readiness_rows.json", rows)
    save_new("source_manifest.json", {"sha256": source_hashes, "chunk_inventory": chunk_headers,
                                     "full_chunks_hashed": len(verified_maps),
                                     "scope": "Complete selected cohort inputs verified; unreferenced chunks indexed by UID and marker only."})
    report = {"complete": True, "created_msk": datetime.now(timezone(timedelta(hours=3))).isoformat(),
              "scope": "Archived exactly-two known-target pairs; readiness only, no new rankings or model execution",
              "totals": {k: dict(v) for k, v in totals.items()}, "videos": len(grouped),
              "rows": len(rows), "archive_schema": schema_union,
              "unavailable_in_episode_archives": unavailable_fields,
              "current_full_maps_verified": len(verified_maps),
              "protected_files_before": before, "protected_files_after": after,
              "protected_files_unchanged": before == after,
              "seconds": time.perf_counter() - start,
              "rows_sha256": digest(OUT / "readiness_rows.json"),
              "manifest_sha256": digest(OUT / "source_manifest.json"),
              "scientific_comparison_ready": False,
              "reason": "Inspect observed-map coverage and missing full tracker state/observed-box fields before designing a faithful spatial identity comparison."}
    save_new("readiness_report.json", report)
    print(json.dumps({"totals": report["totals"], "videos": report["videos"],
                      "seconds": report["seconds"], "protected_unchanged": before == after}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
