"""Explain native edge eligibility from the already recovered first-video state."""
import json
import numpy as np
from audit_readiness import ROOT, OUT, read, digest, save_new


def main():
    probe = read(OUT / "tracker_recovery_probe.json")
    binding = {r["uid"]: r for r in read(OUT / "readiness_rows.json")}
    raw = ROOT / "runs/train_cache" / probe["video"] / "detections/raw_video.npz"
    with np.load(raw, allow_pickle=False) as z:
        boxes, confidence, frames = z["xyxy"], z["conf"], z["frame_idx"]
    results = []
    for r in probe["rows"]:
        cx, cy, w, h = r["mean_xywh_velocity"][:4]
        predicted = np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        high = np.flatnonzero((frames == r["frame"]) & (confidence >= .25))
        candidate_rows = []
        for j in binding[r["uid"]]["detector_candidates"]:
            index = int(high[j])
            b = boxes[index].astype(float)
            intersection = np.maximum(0, np.minimum(predicted[2:], b[2:])-np.maximum(predicted[:2], b[:2])).prod()
            union = w*h + np.maximum(0, b[2:]-b[:2]).prod() - intersection
            iou = float(intersection / union) if union > 0 else 0.0
            cost = 1.0 - iou * float(confidence[index])
            candidate_rows.append({"high_index": j, "raw_index": index, "box": b.tolist(),
                                   "confidence": float(confidence[index]), "iou": iou,
                                   "fused_cost": cost, "below_native_threshold": cost <= .8})
        results.append({"uid": r["uid"], "predicted_box": predicted.tolist(),
                        "native_assignment": r["native_assignment_raw_detection"], "candidates": candidate_rows})
    save_new("candidate_cost_report.json", {"complete": True, "rows": results,
        "probe_sha256": digest(OUT / "tracker_recovery_probe.json"),
        "detector_sha256": digest(raw), "script_sha256": digest(__file__),
        "limitation": "Local eligibility only; other tracks and Hungarian competition were not exported."})
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
