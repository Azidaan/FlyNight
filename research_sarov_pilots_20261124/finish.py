"""Recompute independent confirmation and live receipts; record the two registered outcomes."""
from pathlib import Path
from collections import Counter
import numpy as np
from lab import OUT, ROOT, read, write, sha, now
from register import journal


def verify():
    cf=read(OUT/"confirmation_result.json")
    primary=read(OUT/"transformer_inputs_v2.json")
    rows=read(OUT/"confirmation_rows.json")
    live=read(OUT/"live_result.json")
    clocks=read(OUT/"live_timings.json")
    assert not set(primary["test"])&set(cf["test"]) and not set(primary["train"])&set(cf["test"])
    assert len(set((r["video"],r["frame"],r["owner"]) for r in rows))==len(rows)
    assert {r["video"] for r in rows}==set(cf["test"])
    assert all(r["geometry"]>=0 or r["guard"]<0 for r in rows)
    for mode,numbers in cf["stats"].items():
        competition=[r for r in rows if r["truth"]>=0 and r["competition"]]
        assert len(competition)==numbers["eligible_competing"]
        assert sum(r[mode]==r["truth"] for r in competition)==numbers["correct_competing"]
        assert sum(r[mode]>=0 and r["truth"]<0 for r in rows)==numbers["false_on_missing"]
        assert sum(r[mode]>=0 and r["truth"]>=0 and r["truth"]!=r[mode] for r in rows)==numbers["wrong_owner"]
    assert len(clocks)==80 and len({r["video"] for r in clocks})==20
    assert len({(r["video"],r["frame"]) for r in clocks})==80
    assert Counter(r["first"] for r in clocks)=={"geometry":40,"attention":40}
    deadlines={}
    for mode in ("geometry","attention"):
        for k in ("total","association"):
            arr=[r[mode][k] for r in clocks]
            assert abs(sum(arr)/len(arr)-live["summary"][mode][k]["mean"])<1e-8
            assert abs(np.quantile(arr,.95)-live["summary"][mode][k]["p95"])<1e-8
        deadlines[mode]={str(fps):sum(r[mode]["total"]<=1000/fps for r in clocks) for fps in (5,25,30)}
    d=np.array([r["attention"]["association"]-r["geometry"]["association"] for r in clocks])
    assert abs(d.mean()-live["summary"]["paired_delta"]["association"]["mean"])<1e-8
    base_file=ROOT/"runs/runtime_benchmark/tracker_runtime.json"
    baseline=read(base_file)["pipelines_including_jpeg_decode"]["yolo26_flyvis_tiled_fused"]
    est=baseline["mean_ms"]+float(d.mean())
    files=["gap_result.json","transformer_result.json","confirmation_result.json","live_result.json",
           "confirmation_inputs.json","live_inputs.json","confirmation_rows.json","live_timings.json","attention_runtime.pt"]
    write(OUT/"confirmation_audit.json",dict(passed=True,ended=now(),nonoverlapping_confirmation_videos=len(cf["test"]),
        checked_confirmation_rows=len(rows),live_paired_frames=len(clocks),frames_inside_deadline=deadlines,
        source_for_projection=base_file.relative_to(ROOT).as_posix(),historical_full_mean_ms=baseline["mean_ms"],
        projected_full_mean_ms=est,projected_fps=1000/est,sha256={name:sha(OUT/name) for name in files}))


def record():
    cat=ROOT/"experiments/project_history.json"
    data=read(cat)
    ids=["sarov_precut_return_20261124","sarov_coordinate_transformer_20261124"]
    lookup={a["id"]:a for a in data["experiments"] if a["id"] in ids}
    assert len(lookup)==2 and all(a["state"]=="running" for a in lookup.values())
    prefix=OUT.relative_to(ROOT).as_posix()+"/"
    primary=lookup[ids[0]]
    primary.update(state="inconclusive",done="2026-11-24",
        obtained="Четыре известных эпизода: полностью верно 0 → 1. Четыре других видео: 1 → 1, неверных назначений соседу 0 → 1. Шестнадцать новых эпизодов в ранее исследованных видео: 5 → 7, парный 95%-й интервал разницы от 0 до +31,25 п. п. Источник "+prefix+"gap_result.json, cohorts.",
        interpretation="Проверка текущим наблюдением вернула отдельные хорошие контуры. На других видео устойчивое улучшение сохранения номера не подтверждено.",
        decision="Оставить диагностический оракул. Перед переносом правила требуется отдельно решить, когда можно доверять соперничающей ветви.",
        cannot_claim="Начальная рамка привязана по прошлой разметке. Проверка двух владельцев и четырёх других видео не измеряет качество полного трекера.")
    for entry,met in zip(primary["acceptance"],(True,True,False)):
        entry.update(met=met,evidence=prefix+"gap_result.json, cohorts; "+prefix+"audit.json")
    secondary=lookup[ids[1]]
    secondary.update(state="completed_negative",done="2026-11-24",
        obtained="На первой группе из 20 видео геометрия и трансформер дали 7423 и 8073 правильных выбора из 8521 при конкуренции; ложные выдачи при отсутствии пригодного кандидата 97 и 216 из 678. Источник "+prefix+"transformer_result.json. На следующих 20 видео верно 4870 и 5353 из 5709, локальная нейросеть дала 5354. После сохранения ожиданий геометрии верно 4866, ложных выдач 49, у геометрии 49. Источник "+prefix+"confirmation_result.json. На 80 кадрах JPEG, YOLO26s и назначение заняли в среднем 29,05 и 29,93 мс; средняя добавка участка назначения 1,06 мс. Источник "+prefix+"live_result.json.",
        interpretation="На обеих независимых от обучения группах сеть чаще продолжала наблюдаемый объект и чаще присваивала наблюдение при отсутствии пригодной рамки. Изолированный вклад внимания относительно локальной обученной сети не подтвердился.",
        decision="Сохранить модель для сравнения. Следующая задача — качество решения ждать и защита от чужой рамки при конкуренции. В рабочий трекер вариант не внедрять.",
        cannot_claim="Перенос на сквозные номера, точное изменение скорости полного трекера, эффективность трансформера на десять миллионов параметров и вклад FlyVis здесь не измерялись.")
    for entry,met,source in zip(secondary["acceptance"],(False,True,True),("confirmation_result.json","live_result.json","transformer_result.json")):
        entry.update(met=met,evidence=prefix+source)
    for a,files in ((primary,["gap_result.json","gap_selection.json","audit.json"]),
                    (secondary,["transformer_result.json","confirmation_protocol.json","confirmation_result.json","live_result.json","confirmation_audit.json"])):
        a["evidence"].extend(dict(kind="file",ref=prefix+name) for name in files)
    write(cat,data)
    journal("Саров 1.1 и 1.2. На четырёх новых для проверки видео восстановление контура дало 1 → 1 полностью правильный эпизод, на 16 других эпизодах прежних видео 5 → 7. Малый трансформер повысил число верных назначений при конкуренции, но в обеих группах чаще назначал чужую рамку, когда правильного наблюдения не было. В парном измерении на 80 реальных кадрах среднее время чтения JPEG, YOLO26s и назначения изменилось с 29,05 до 29,93 мс. Решение — рабочий трекер не менять. Первичные измерения и границы вывода — `"+prefix+"РЕЗУЛЬТАТ.md`.")


if __name__=="__main__":
    verify()
    record()
