import sys, json, numpy as np
sys.path.insert(0, "E:/Dissertation")
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.evaluate_motion_reliability import collect_split, GROUPS, NAMES, GEOMETRY, DIS, FLY
from scripts.evaluate_flyvis_qmatch_gate import TRAIN_VIDEO_IDS, TEST_VIDEO_IDS
R = Path("E:/Dissertation/runs")
sel = json.load(open(R/"train_cache/selection.json", encoding="utf-8"))["videos"]
lum = json.load(open(R/"motion_reliability_none_strata/luminance.json"))
labels_tr = Path("E:/Dissertation/data/bdd100k/labels/box_track_20/train"); labels_val = Path("E:/Dissertation/data/bdd100k/labels/box_track_20/val")
out = {}
for mode, troot, vroot in (("none", R/"train_cache", R/"flyvis_fovea_flow"), ("lamina", R/"train_cache_retina_lamina4", R/"flyvis_fovea_flow_retina_lamina4")):
    ready_tr = [v for v in sel["train"] if (R/"train_cache_retina_lamina4"/v/"fovea_flow.npz").exists()]
    ready_te = [v for v in sel["test"] if (R/"train_cache_retina_lamina4"/v/"fovea_flow.npz").exists()]
    val = list(TRAIN_VIDEO_IDS + TEST_VIDEO_IDS)
    D = {"train": collect_split(troot, R/"train_cache", labels_tr, ready_tr) + (ready_tr,),
         "test": collect_split(troot, R/"train_cache", labels_tr, ready_te) + (ready_te,),
         "val7": collect_split(vroot, R/"video_rate", labels_val, val) + (val,)}
    tau = 0.0903  # τ_e из motion_reliability_v1 (порог на полном train режима none)
    rng = np.random.default_rng(0); F = {}
    for s, d in D.items():
        X = d[0].copy(); a = len(GEOMETRY) + len(DIS); sh = X[:, a:a+len(FLY)].copy()
        for v in np.unique(d[2]):
            idx = np.flatnonzero(d[2] == v); sh[idx] = sh[rng.permutation(idx)]
        F[s] = np.concatenate([X, sh], axis=1)
    y = {s: (d[1] > tau).astype(int) for s, d in D.items()}
    res = {}
    for name, cols in GROUPS.items():
        idx = [NAMES.index(c) for c in cols]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=42)).fit(F["train"][:, idx], y["train"])
        res[name] = {}
        for s in ("test", "val7"):
            p = m.predict_proba(F[s][:, idx])[:, 1]; res[name][s] = float(average_precision_score(y[s], p))
            per = {}
            for k, v in enumerate(D[s][3]):
                mk = D[s][2] == k
                if mk.sum() > 20 and 0 < y[s][mk].sum() < mk.sum(): per[v] = float(average_precision_score(y[s][mk], p[mk]))
            res[name][s + "_by_video"] = per
        dark = [res[name]["test_by_video"][v] for v in res[name]["test_by_video"] if lum[v] < 60]
        bright = [res[name]["test_by_video"][v] for v in res[name]["test_by_video"] if lum[v] >= 60]
        res[name]["test_dark"] = float(np.mean(dark)); res[name]["test_bright"] = float(np.mean(bright)); res[name]["n_dark"] = len(dark); res[name]["n_bright"] = len(bright)
    out[mode] = res
    print(f"== {mode}: train {len(ready_tr)} роликов, test {len(ready_te)}")
    for name in GROUPS:
        r = res[name]; print(f"  {name:22s} test {r['test']:.4f} val7 {r['val7']:.4f} | тёмные ({r['n_dark']}) {r['test_dark']:.4f} светлые ({r['n_bright']}) {r['test_bright']:.4f}")
json.dump(out, open(R/"retina_frontend/stageB_partial.json", "w", encoding="utf-8"), indent=1)
print("OK runs/retina_frontend/stageB_partial.json")
