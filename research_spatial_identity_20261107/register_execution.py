"""Register the user's authorized comparison before its new outputs exist."""
import subprocess
from datetime import datetime
from pathlib import Path
from run_common import ROOT, RUN, MSK, read, sha, write

IDENT = "spatial_identity_maps_v1_20261107"


def main():
    assert not RUN.exists(), "Preserve prior execution"
    original = ROOT / "research_spatial_identity_20261107/comparison_protocol.json"
    p = read(original)
    now = datetime.now(MSK)
    p.update(id=IDENT, status="running_recovery", registered=now.isoformat(),
             authorization="User explicitly requested restoration, comparison under the protocol and diagnosis of the two missing assignments.",
             git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             parent_protocol_sha256=sha(original),
             deadline="2026-11-09T07:40:00+03:00",
             execution_addendum="Re-export first video to capture previously absent full association matrices, retaining the prior successful recovery probe. Future stages get separate immutable source manifests before outputs; original scientific settings remain those of the draft.")
    sources = ["research_spatial_identity_20261107/run_common.py", "research_spatial_identity_20261107/export_tracker.py",
               "research_spatial_identity_20261107/register_execution.py", "research_spatial_identity_20261107/comparison_protocol.json",
               "research_spatial_identity_20261107/readiness_rows.json", "research_spatial_identity_20261107/readiness_report.json",
               "tracking/xywh_bytetrack.py", "tracking/offline_association.py", "tracking/foveated_eyes.py", "tracking/z13_readout.py",
               ".venv/Lib/site-packages/ultralytics/trackers/byte_tracker.py", ".venv/Lib/site-packages/ultralytics/trackers/bot_sort.py",
               ".venv/Lib/site-packages/ultralytics/trackers/utils/matching.py",
               "runs/flyvis/results/flow/0000/000/best_chkpt", "runs/z13_target_center_dev_20261018/C/normalization.json",
               "runs/z13_local_peak_20261021/geometry.json"]
    p["source_sha256"] = {name: sha(ROOT/name) for name in sources}
    RUN.mkdir()
    write(RUN / "execution_protocol.json", p, new=True)
    (RUN / ".gitignore").write_text("tracker/*.npz\nfeatures/\ngeometry/\n*.log\n*.worker.log\n", encoding="utf-8")
    (RUN / ".gitattributes").write_text("* -text whitespace=cr-at-eol\n", encoding="utf-8")
    catalog_path = ROOT / "experiments/project_history.json"
    cat = read(catalog_path)
    assert not any(c["id"] == IDENT for c in cat["experiments"])
    cat["experiments"].append(dict(id=IDENT, title="Пространственный образ прошлой цели — выбор продолжения среди соседних рамок",
        area="flyvis", state="running", started=now.date().isoformat(),
        question=p["question"], hypothesis=p["hypothesis"], expected="Парное преимущество карт прошлого и нынешнего наблюдений над прежним считывателем и контролями без ядра при общей геометрии.",
        acceptance=[{"criterion": p["gates"][key], "met": None} for key in ("data", "technical", "proceed")],
        obtained="", interpretation="", decision="", cannot_claim=p["cohorts"]["independence"],
        evidence=[{"kind":"file", "ref": "runs/spatial_identity_maps_20261107/execution_protocol.json", "claim":"Условия до восстановления и сравнения"}],
        tags=["fly-problem-P8"]))
    write(catalog_path, cat)
    plan = ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md"
    before = plan.read_bytes()
    newline = b"\r\n" if b"\r\n" in before else b"\n"
    anchor = "## Журнал изменений".encode() + newline + newline
    assert before.count(anchor) == 1
    entry = (f"**{now:%d.%m.%Y, %H:%M} MSK** — `{IDENT}` зарегистрирован по прямому поручению автора восстановить данные, провести сравнение и разобрать два отсутствующих назначения. "
             "*Постановка.* Общая геометрия, восемь карт T4/T5, перестановка, изображение и R1; порог по fit20 до evaluate26. "
             "*Условия.* Исходные935пар и682эпизода; отсутствующие карты прошлого наблюдения извлекаются из сохранённых входов, текущие переиспользуются. "
             "Экспорт реальных назначений отделён от ранжирования пары. Первая повторная запись видео нужна для ранее не сохранённой полной матрицы. "
             "*Критерий.* Четыре положительные нижние98,75%границы парных разностей по исходным26видео. Научного результата пока нет. "
             "Источник: `runs/spatial_identity_maps_20261107/execution_protocol.json`.\n\n")
    plan.write_bytes(before.replace(anchor, anchor + entry.replace("\n", newline.decode()).encode(), 1))
    print("Registered", IDENT, flush=True)


if __name__ == "__main__":
    main()
