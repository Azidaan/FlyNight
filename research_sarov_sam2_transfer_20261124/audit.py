from pathlib import Path
import hashlib
import json
import math

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
report = json.loads((OUT / "result.json").read_bytes())
selection = json.loads((OUT / "selection.json").read_bytes())
known = report["three_car_diagnostic"]
assert len({r["video"] for r in report["cases"]}) == 20
assert not ({r["video"] for r in report["cases"]} & set(selection["excluded"]))
counts = {arm:{k:0 for k in ("correct", "wait", "wrong", "wrong_without_candidate", "complete_pairs")}
          for arm in ("late", "early")}
wins = regressions = 0
changes = []
for row in [*report["cases"], known]:
    path = ROOT / row["mask_archive"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["pixel_sha256"]
    audit = {}
    with np.load(path, allow_pickle=False) as data:
        obs = data["current"]
        truth = row["truth"]
        for name in ("late", "early"):
            masks = data[name]
            assert masks.shape[0] == len(truth) and len(obs) == len(row["boxes"])
            predictions = masks[:, -1]
            match = {}
            diagnosis = []
            overlap = np.array([[np.count_nonzero(a & b) for b in obs] for a in predictions])
            pa = predictions.sum(axis=(1,2))
            ca = obs.sum(axis=(1,2))
            union = pa[:,None] + ca[None,:] - overlap
            score = np.divide(overlap, union, out=np.zeros(union.shape, dtype=float), where=union>0)
            assert np.array_equal(score, row["arms"][name]["scores"])
            for i, j in enumerate(truth):
                preferred = np.flatnonzero(score[i] == np.max(score[i])) if len(obs) else []
                selected = None
                if len(preferred) == 1:
                    k = int(preferred[0])
                    if 2 * overlap[i,k] > union[i,k] and (score[:,k] >= score[i,k]).sum() == 1:
                        selected = k
                        match[i] = k
                outcome = "wait" if selected is None else "correct" if selected == j else "wrong"
                diagnosis.append(dict(
                    outcome=outcome, preferred=[int(k) for k in preferred], assigned=selected,
                    own_score=float(score[i,j]) if j >= 0 else None,
                    preferred_score=float(np.max(score[i])) if len(obs) else None,
                    prediction_area=int(pa[i]), candidate_area=int(ca[j]) if j >= 0 else None,
                    intersection=int(overlap[i,j]) if j >= 0 else None))
                if row is not known:
                    counts[name][outcome] += 1
                    counts[name]["wrong_without_candidate"] += int(j < 0 and outcome == "wrong")
            assert match == {int(i): k for i, k in row["arms"][name]["assigned"].items()}
            audit[name] = diagnosis
        if any(a["outcome"] != b["outcome"] for a,b in zip(*audit.values())):
            changes.append(dict(video=row["video"], frame=row["frame_index"]+1, truth=row["truth"], **audit))
        if row is known:
            continue
        late, early = [all(r["outcome"] == "correct" for r in arm) for arm in audit.values()]
        wins += int(early and not late)
        regressions += int(late and not early)
        counts["early"]["complete_pairs"] += int(early)
        counts["late"]["complete_pairs"] += int(late)

n = wins + regressions
p = min(1, 2 * sum(math.comb(n, i) for i in range(min(wins, regressions)+1)) / (2**n))
summary = report["summary"]
assert counts == {k: summary[k] for k in counts}
assert (wins, regressions, p) == (summary["wins"], summary["regressions"], summary["paired_p"])
assert summary["both_have_candidates"] == sum(all(i >= 0 for i in row["truth"]) for row in report["cases"])
assert summary["owners_missing_candidate"] == sum(i < 0 for row in report["cases"] for i in row["truth"])
(OUT / "audit.json").write_text(json.dumps(dict(
    checked_masks=len(report["cases"])+1, counts=counts, wins=wins, regressions=regressions,
    paired_p=p, changed_outcomes=changes,
), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(counts, wins, regressions, p)
