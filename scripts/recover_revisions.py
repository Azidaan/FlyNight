"""Восстановить прежние версии исправленных текстов карточек из истории git (правило автора 10.10).

Правило: при исправлении текста карточки прежний текст не удаляется — он сохраняется в revisions[] карточки
({date, commit, field, old, reason}) и показывается в графе блоком «Прежние версии», на случай если исправление окажется
ошибкой. Этот скрипт находит в истории experiments/project_history.json тексты полей interpretation, decision,
cannot_claim, obtained, summary, которые после того, как карточка получила итоговое состояние, были ЗАМЕНЕНЫ (прежний
текст не входит целиком в нынешний — простые дополнения не считаются), и добавляет их в revisions[] (без повторов).

Запуск: python scripts/recover_revisions.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = "experiments/project_history.json"
FIELDS = ("summary", "obtained", "interpretation", "decision", "cannot_claim")
FINAL = {"completed_positive", "completed_negative", "inconclusive", "closed", "superseded"}


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout


def collect(snapshots) -> dict:
    """snapshots: [(commit, date, catalog)] по времени. Возвращает (id, field) -> [(commit, date, value)] после итога."""
    seen: dict[tuple, list] = {}
    for h, date, cat in snapshots:
        for e in cat.get("experiments", []):
            if e.get("state") not in FINAL:
                continue
            for f in FIELDS:
                v = e.get(f)
                if not isinstance(v, str) or not v.strip():
                    continue
                lst = seen.setdefault((e["id"], f), [])
                if not lst or lst[-1][2] != v:
                    lst.append((h, date, v))
    return seen


def add_revisions(cur: dict, seen: dict) -> int:
    """Добавить в revisions[] карточек тексты, заменённые после итога (без повторов). Возвращает число добавленных."""
    added = 0
    for e in cur["experiments"]:
        revs = e.setdefault("revisions", [])
        known = {(r.get("field"), r.get("old")) for r in revs}
        for f in FIELDS:
            now = e.get(f) or ""
            hist = seen.get((e["id"], f), [])
            for i, (h, date, v) in enumerate(hist):
                if v == now or v in now or (f, v) in known:
                    continue                      # не заменён (совпадает или вошёл в нынешний текст целиком)
                nxt = hist[i + 1] if i + 1 < len(hist) else None
                revs.append({"date": nxt[1] if nxt else date, "commit": nxt[0] if nxt else h, "field": f, "old": v,
                             "reason": "восстановлено из истории git 10.10: текст заменён после итога; причина правки — в журнале плана за эту дату"})
                known.add((f, v)); added += 1
        if not revs:
            e.pop("revisions")
    return added


def main() -> int:
    commits = [l.split(" ", 1) for l in git("log", "--reverse", "--format=%h %ad", "--date=short", "--", CAT).splitlines()]
    snaps = []
    for h, date in commits:
        try:
            snaps.append((h, date, json.loads(git("show", f"{h}:{CAT}"))))
        except Exception:
            continue
    seen = collect(snaps)
    cur = json.loads((ROOT / CAT).read_text(encoding="utf-8"))
    added = add_revisions(cur, seen)
    (ROOT / CAT).write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] восстановлено прежних версий: {added}; карточек с прежними версиями: {sum(1 for e in cur['experiments'] if e.get('revisions'))}")
    for e in cur["experiments"]:
        for r in e.get("revisions", []):
            print(f"  {e['id']} · {r['field']} · {r['date']} ({r['commit']}): {r['old'][:90]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
