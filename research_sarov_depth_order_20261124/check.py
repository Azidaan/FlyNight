from pathlib import Path
import sys
import hashlib
from collections import Counter
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
sys.path.insert(0,str(ROOT/"research_sarov_pilots_20261124"))
from lab import fingerprint, read, write, now, sha
from register import journal
OLD=ROOT/"research_depth_identity_competition_20261114"
SPEC=read(OUT/"protocol.json")


def catalog(end=None):
    path=ROOT/"experiments/project_history.json"
    obj=read(path)
    key=SPEC["id"]
    existing=next((r for r in obj["experiments"] if r["id"]==key),None)
    prefix=OUT.relative_to(ROOT).as_posix()+"/"
    if not end:
        assert existing is None
        obj["experiments"].append(dict(id=key,title="Порядок ближе–дальше для различения двух машин",
            area="own_tracker",state="running",started="2026-11-24",question=SPEC["question"],
            hypothesis="Относительная глубина пары сохранится чаще, чем её вертикальный порядок в кадре.",
            expected="Сравнить две перестановки тех же наблюдений без переобучения и новых порогов.",
            acceptance=[dict(criterion=SPEC["criterion"],met=None)], evidence=[dict(kind="file",ref=prefix+"protocol.json")]))
        message="Саров 1.6. Зарегистрирована проверка сохранения относительного порядка машин по глубине и вертикальному положению перед решением о пробе SAM 3D."
    else:
        assert existing and existing["state"]=="running"
        q=end["auto"]
        met=q["depth_wins"]>q["height_wins"]
        message=(f"Саров 1.6. Глубина по медиане маски дала {q['ordinal_depth']} правильных соответствий, вертикальный порядок — {q['ordinal_y']} из {q['eligible']}. "
                 f"При расхождении решений выигрыши глубины и высоты — {q['depth_wins']} и {q['height_wins']}. "
                 "Проверена гипотеза о порядке на старых масках. Скорость и точность SAM 3D Objects не измерялись.")
        existing.update(state="completed_positive" if met else "completed_negative",done="2026-11-24",
            obtained=message+" Источник "+prefix+"result.json, groups.auto.",
            interpretation="Оценивалось сохранение номеров по порядку двух объектов на изображении. Физическое расстояние до камеры не размечено.",
            decision="Использовать отбор спорных эпизодов для отдельной проверки 3D-положения при наличии подходящего оборудования.",
            cannot_claim="Данные уже изучались. Результат не измеряет точность SAM 3D, метрическую глубину и полный трекер.")
        existing["acceptance"][0].update(met=met,evidence=prefix+"result.json")
        existing["evidence"].append(dict(kind="file",ref=prefix+"result.json"))
    write(path,obj)
    journal(message+" Источник — `"+prefix+("result.json" if end else "protocol.json")+"`.")


def winner(t):
    return None if t[0]==t[1] else int(np.argmin(t))


def rank_choice(values):
    d0=values[0][2]-values[1][2]
    d1=values[2][2]-values[3][2]
    if d0==0 or d1==0:
        return None
    return int((d0>0)!=(d1>0))


def run():
    files=[OUT/"check.py",OUT/"protocol.json"]+[OLD/(k+ext) for k,ext in (
        ("descriptors",".npz"),("pairs",".json"),("choices",".json"),("result",".json"))]
    fp=fingerprint(files)
    write(OUT/"inputs.json",fp)
    ground={(r["case"],r["offset"],r["branch"]):r for r in read(OLD/"result.json")["records"]}
    cases={c["id"]:c for c in read(OLD/"pairs.json")["cases"]}
    groups={}
    examples=[]
    traces=[]
    with np.load(OLD/"descriptors.npz",allow_pickle=False) as ds:
        for x in read(OLD/"choices.json")["records"]:
            cid,j,branch=x["case"],x["offset"],x["branch"]
            g=ground[cid,j,branch]
            if not g["both_visible"]:
                continue
            q=groups.setdefault(branch,Counter())
            q["eligible"]+=1
            swap=x["shuffled"]
            keys=[f"{branch}__{cid}__{i}__{t}" for i,t in [(0,0),(1,0),(swap,j),(1-swap,j)]]
            if any(k+"__Depth" not in ds for k in keys):
                q["missing_input"]+=1
                continue
            choices={}
            costs={}
            for m in ("Depth","Y","RGB"):
                a,b,c,d=[ds[k+"__"+m] for k in keys]
                dist=lambda x,y:float(np.abs(x-y).mean())
                costs[m]=np.array([dist(a,c)+dist(b,d),dist(a,d)+dist(b,c)])
                assert winner(costs[m])==x["decisions"][m]
                if m!="RGB":
                    choices[m]=rank_choice([a,b,c,d])
            for m in ("Depth","Y"):
                def norm(v):return v/v.sum() if v.sum()>0 else np.full(2,.5)
                assert winner(norm(costs["RGB"])+norm(costs[m]))==x["decisions"]["RGB+"+m]
            answer=g["matched_permutation"]
            if answer is None:
                q["no_unique_pair"]+=1
                continue
            assert g["reason"]=="pairs_evaluable"
            q["evaluable"]+=1
            depth_ok=choices["Depth"]==answer
            y_ok=choices["Y"]==answer
            q["ordinal_depth"]+=int(depth_ok)
            q["ordinal_y"]+=int(y_ok)
            q["depth_wins"]+=int(depth_ok and not y_ok)
            q["height_wins"]+=int(y_ok and not depth_ok)
            q["both_wrong"]+=int(not y_ok and not depth_ok)
            q["ordinal_disagreements"]+=int(choices["Depth"]!=choices["Y"])
            q["distribution_disagreements"]+=int(x["decisions"]["Depth"]!=x["decisions"]["Y"])
            q["simple_control_correct"]+=int(x["decisions"]["RGB+Y"]==answer)
            row=dict(case=cid,video=x["video"],offset=j,branch=branch,answer=answer,
                     depth=choices["Depth"],height=choices["Y"],shuffled=swap)
            traces.append(row)
            if choices["Depth"]!=choices["Y"] or x["decisions"]["Depth"]!=x["decisions"]["Y"] or not depth_ok and not y_ok:
                frame=cases[cid]["frames"][j]; reference=cases[cid]["frames"][0]
                assert sha(ROOT/frame["image"])==frame["sha256"] and sha(ROOT/reference["image"])==reference["sha256"]
                examples.append(dict(row,current=frame["image"],reference=reference["image"],
                    mask_archive=str((OLD/(branch+"_candidates.npz")).relative_to(ROOT).as_posix()),
                    current_mask_keys=[f"{cid}__{i}__{j}" for i in (0,1)]))
    assert fingerprint(files)["sha256"]==fp["sha256"]
    for q in groups.values():
        assert q["eligible"]==q["evaluable"]+q["missing_input"]+q["no_unique_pair"]
        assert q["ordinal_depth"]-q["ordinal_y"]==q["depth_wins"]-q["height_wins"]
    write(OUT/"result.json",dict(ended=now(),groups=groups,pairs_to_examine=examples,queries=traces,
          official_sam3d_inference=False,inputs_unchanged=True))
    print({k:dict(v) for k,v in groups.items()})
    return groups


if __name__=="__main__":
    if sys.argv[1]=="register":
        catalog()
    elif sys.argv[1]=="run":
        scores=run()
        catalog(scores)
