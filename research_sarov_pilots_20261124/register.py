import sys
from lab import OUT, ROOT, read, write, now

p = read(OUT / "protocol.json")
path = ROOT / "experiments/project_history.json"
plan = ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md"
specs = [p["oracle"], p["transformer"]]
data = read(path)
ids = {s["id"] for s in specs}
refs = OUT.relative_to(ROOT).as_posix()
mark = "## Журнал изменений"
def journal(message):
    original = plan.read_bytes()
    lines = original.decode("utf-8")
    nl = "\r\n" if "\r\n" in lines else "\n"
    anchor = mark+nl
    assert lines.count(anchor)==1
    from datetime import datetime
    stamp = datetime.fromisoformat(now()).strftime("%d.%m.%Y, %H:%M MSK")
    extra = nl+"**"+stamp+"** — "+message+nl
    assert plan.read_bytes()==original
    plan.write_bytes(lines.replace(anchor,anchor+extra,1).encode("utf-8"))

if __name__ == "__main__" and len(sys.argv)==1:
    assert not ids & {x["id"] for x in data["experiments"]}
    for s in specs:
        data["experiments"].append(dict(id=s["id"], title=s["title"], area="own_tracker", state="running",
            started="2026-11-24", question=s["title"], hypothesis=s.get("action",s.get("conditions")),
            expected=s["criteria"][0], acceptance=[{"criterion":v,"met":None} for v in s["criteria"]],
            evidence=[dict(kind="file",ref=refs+"/protocol.json")]))
    write(path, data)
    journal("Проверки Сарова 1.1 и 1.2 зарегистрированы до вычислений. Маска до обнуления проходит прежнее правило совместного назначения. Малый трансформер сравнивается с геометрией на других видео, отдельно измеряется добавочное время. Условия — `"+refs+"/protocol.json`.")
