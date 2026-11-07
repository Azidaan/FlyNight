import json
import pandas as pd

ROOT = "E:/Dissertation/data/flywire/783/"
OUT34 = ["T1", "T2", "T2a", "T3", "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
         "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9", "Tm16", "Tm20", "Tm28", "Tm30",
         "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18"]
OTHER_FLYVIS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2",
                "Am", "C2", "C3", "CT1", "Mi1", "Mi2", "Mi3", "Mi4", "Mi9", "Mi10", "Mi11", "Mi12", "Mi13",
                "Mi14", "Mi15"]
TARGETS = ["LC10a", "LC10b", "LC10c", "LC10d", "LC11", "LC12", "LC15", "LC17", "LC18", "LC21", "LC4", "LC6",
           "LC9", "LPLC1", "LPLC2"]

vt = pd.read_csv(ROOT + "visual_neuron_types.csv.gz")
typ = dict(zip(vt.root_id, vt.type.astype(str)))
fam = dict(zip(vt.root_id, vt.family.astype(str)))
tgt_ids = set(vt.root_id[vt.type.isin(TARGETS)])

acc = []
for ch in pd.read_csv(ROOT + "connections_princeton_no_threshold.csv.gz",
                      usecols=["pre_root_id", "post_root_id", "syn_count"], chunksize=5_000_000):
    acc.append(ch[ch.post_root_id.isin(tgt_ids)])
df = pd.concat(acc)
df = df[df.syn_count >= 5]  # standard FlyWire threshold
df["post_type"] = df.post_root_id.map(typ)
df["pre_type"] = df.pre_root_id.map(typ).fillna("non_optic_or_untyped")
df["pre_fam"] = df.pre_root_id.map(fam).fillna("non_optic_or_untyped")

def cls(t):
    if t in OUT34:
        return "flyvis_output34"
    base = "CT1" if t.startswith("CT1") else t
    if base in OTHER_FLYVIS:
        return "flyvis_other"
    return "not_in_flyvis"

df["cls"] = df.pre_type.map(cls)
res = {}
for t, g in df.groupby("post_type"):
    tot = g.syn_count.sum()
    by = (g.groupby("cls").syn_count.sum() / tot).round(3).to_dict()
    top = (g.groupby("pre_type").syn_count.sum() / tot).sort_values(ascending=False).head(8).round(3).to_dict()
    t4t5 = g[g.pre_type.str.match(r"^T[45][a-d]$")].syn_count.sum() / tot
    res[t] = {"n_cells": int(g.post_root_id.nunique()), "syn_total": int(tot), "fractions": by,
              "T4T5_fraction": round(float(t4t5), 4), "top_inputs": top}
print(json.dumps(res, indent=1))
json.dump(res, open("C:/Users/A946~1/AppData/Local/Temp/claude/E--Dissertation/7acfb76a-f9ac-47a9-a937-ac8540584be9/scratchpad/lc_inputs.json", "w"), indent=1)
