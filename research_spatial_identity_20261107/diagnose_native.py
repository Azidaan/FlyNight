"""Independently explain the two native assignment decisions from exported matrices."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_common import ROOT, RUN, read, sha, write


def optimum(cost, threshold):
    n, m = cost.shape
    full = np.full((n+m, n+m), threshold/2, dtype=np.float64)
    full[:n, :m] = cost
    full[n:, m:] = 0
    i, j = linear_sum_assignment(full)
    return float(full[i, j].sum())


def main():
    path = RUN / "tracker/0001542f-ec815219.json"
    original = read(path)
    write(RUN / "native_diagnosis_protocol.json", {
        "uids": [29017,29057], "source_sha256": {str(path.relative_to(ROOT).as_posix()):sha(path),
        str(path.with_suffix('.npz').relative_to(ROOT).as_posix()):sha(path.with_suffix('.npz')),
        str(Path(__file__).relative_to(ROOT).as_posix()):sha(__file__)},
        "method": "Independent SciPy linear assignment on augmented square matrix: each unmatched row/column costs threshold/2. Compare stored native objective and the optimum when an eligible edge for the lost track is forced.",
        "scope": "Explains the native optimizer; no claim of correct or incorrect physical identity of competitor tracks."}, new=True)
    results = []
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
        for row in original["rows"]:
            frame = str(row["frame"])
            meta = original["snapshots"][frame]
            cost = z["cost_" + frame].astype(np.float64)
            threshold = meta["threshold"]
            n, m = cost.shape
            chosen = meta["matches"]
            stored = sum(cost[i,j] for i,j in chosen) + (n+m-2*len(chosen))*threshold/2
            best = optimum(cost, threshold)
            assert abs(stored-best) < 1e-7, (row['uid'],stored,best)
            own = meta["track_ids"].index(row["tid"])
            candidates = []
            for raw, owner in zip(row["candidate_raw_ids"], row["candidate_assigned_track_ids"]):
                col = meta["detection_raw_ids"].index(raw)
                value = float(cost[own,col])
                item = {"raw_detection":raw,"native_owner_tid":owner,"lost_track_cost":value,
                        "eligible":value <= threshold}
                if owner is not None:
                    item['owner_cost'] = float(cost[meta['track_ids'].index(owner),col])
                if value <= threshold:
                    reduced = np.delete(np.delete(cost,own,axis=0),col,axis=1)
                    forced = value + optimum(reduced,threshold)
                    item.update(forced_objective=forced,objective_increase=forced-best)
                candidates.append(item)
            results.append({"uid":row['uid'],"frame":row['frame'],"lost_tid":row['tid'],
                            "native_assignment":row['native_assignment_raw_id'],"matrix_shape":list(cost.shape),
                            "native_objective":stored,"independent_objective":best,"candidates":candidates})
    write(RUN / "native_diagnosis.json", {"passed":True,"cases":results,
        "conclusion":"In both cases an eligible candidate was assigned to a cheaper competing track; forcing the lost track increases the full assignment objective. The other candidate is above the native cost limit."}, new=True)
    print(json.dumps(results),flush=True)


if __name__ == '__main__':
    main()
