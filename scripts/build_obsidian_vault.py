"""Собрать хранилище Obsidian из документации проекта (ObsidianFlyVis).

Источники (только чтение): dissertation/ЕДИНЫЙ_ПЛАН.md (разделы и журнал), experiments/project_history.json
(карточки, направления), experiments/metrics_glossary.json (метрики, термины), research_*/ (исследования),
прочие документы dissertation/ и корня. Заметки мухи (01 Муха) — обзор поверх карточек, числа в них
ссылаются на карточки-источники.

Безопасность: скрипт пишет только в свои папки хранилища и ведёт список созданных файлов
(.generated.json); при пересборке удаляет только файлы из этого списка. Чужие заметки не трогает.
Настройки графа (.obsidian/graph.json) дополняются группами цветов, остальные ключи сохраняются.

Запуск: python scripts/build_obsidian_vault.py [--vault ObsidianFlyVis]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GEN_DIRS = ["00 Главная.md", "01 Муха", "02 Эксперименты", "03 Направления", "04 Термины и метрики", "05 План", "06 Журнал", "07 Исследования", "08 Документы"]

STATE_RU = {"planned": "запланирован", "ready": "готов к запуску", "running": "идёт", "completed_positive": "положительный",
            "completed_negative": "отрицательный", "inconclusive": "неопределённый", "closed": "закрыт", "infrastructure": "инфраструктура",
            "superseded": "заменён", "unknown": "неизвестно"}
STATE_ICON = {"completed_positive": "🟢", "completed_negative": "🔴", "inconclusive": "🟡", "running": "🔵", "planned": "⚪", "ready": "⚪",
              "closed": "⚫", "infrastructure": "🔧", "superseded": "↪️", "unknown": "❔"}
FIELDS = [("question", "Вопрос"), ("hypothesis", "Гипотеза"), ("expected", "Ожидалось (до прогона)"), ("done", "Что сделано"),
          ("obtained", "Получено"), ("interpretation", "Интерпретация"), ("decision", "Решение"), ("limitations", "Ограничения"),
          ("cannot_claim", "Нельзя заявлять")]
RESEARCH = {"research_project_now": "Модернизация мухи и линия B", "research_fly_detector": "Детектор с мухой", "research_fundamental_problems": "Фундаментальные проблемы"}
FIELD_RU = {"essence": "Суть", "sources": "Источники", "relevance": "Относимость к проекту", "novelty_overlap": "Пересечение с чужими работами",
            "author_results": "Результаты у авторов", "measured_level": "Уровень измерения", "matched_control": "Согласованный контроль",
            "conditions_breakdown": "Разрез по условиям", "cost_here": "Стоимость у нас", "expected_effect": "Ожидаемый эффект", "risks": "Риски",
            "attribution": "Как подписать заимствование", "fly_role": "Место мухи", "loss_mechanism_targeted": "Какую потерю закрывает",
            "strong_baseline_check": "Проверка против сильной базы", "cheap_gate": "Дешёвая проверка", "mechanism": "Механизм",
            "evidence_here": "Где видно у нас", "solutions": "Решения из литературы", "fly_transfer": "Что сделать с мухой",
            "pre_build_diagnostic": "Тест до постройки", "ceiling": "Потолок", "target_subset": "Где обязано помогать",
            "negative_control": "Отрицательный контроль", "branch_contribution": "Вклад ветви", "mde_and_cost": "Минимальный эффект и стоимость",
            "verdict_for_project": "Вердикт для проекта"}

# Темы мухи: правило отбора карточек (по id) → заголовок
FLY_THEMES = [
    ("Глаз и сетчатка-ламина", ["flyvis_retina_frontend_v1", "flyvis_retina_stageB_v1", "dis_retina_control_v1", "fly_eye_frontend_v1"]),
    ("Движение мухи", ["flyvis_stress", "motion_confidence_audit", "flyvis_fovea_motion_v1", "motion_reliability_v1", "e2e_motion_source_v1", "fly_decoder_refit_v1"]),
    ("Считывание активности и события", ["flyvis_internal_readout_v1", "flyvis_foveation_readout_v1", "flyvis_readout_nonlinear_diag", "flyvis_event_features_v1", "qmatch_gate", "qmatch_30hz_pseudo", "qmatch_candidate_shift"]),
    ("Отпечаток мухи и опознание", ["fly_signature_reid_v1", "fly_matched_controls_v1", "fly_connectome_null_v1", "stitch_appearance_v1"]),
    ("Устойчивость выводов: ансамбль", ["fly_ensemble_members_v1"]),
    ("Трекер на мухе", ["flyvis_standalone", "fly_tracker_lifecycle_v1"]),
]


def _trim(name: str, n: int) -> str:
    return name[:n].strip().rstrip(".").strip()


def safe(name: str, n: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name)
    return _trim(re.sub(r"\s+", " ", name).strip(), n)


def tag(s: str) -> str:
    return re.sub(r"[^\w/-]", "-", s).strip("-")


def fm(d: dict) -> str:
    return "---\n" + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=10000).strip() + "\n---\n\n"


class Vault:
    def __init__(self, path: Path):
        self.path = path
        self.written: list[str] = []

    def write(self, rel: str, text: str) -> None:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        self.written.append(rel.replace("\\", "/"))

    def copy(self, src: Path, rel: str) -> None:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)
        self.written.append(rel.replace("\\", "/"))


class Linker:
    """Ссылки на карточки (по id), разделы плана (§7.7, раздел 7.5) и термины — в произвольном тексте."""

    def __init__(self, card_ids, plan_sections):
        self.ids = sorted(card_ids, key=len, reverse=True)
        self.card_re = re.compile(r"`?(?<![\w/\[])(" + "|".join(map(re.escape, self.ids)) + r")(?![\w\]])`?") if self.ids else None
        self.sec = plan_sections  # номер → имя заметки
        self.sec_re = re.compile(r"(§\s?|раздел[а-яё]*\s+|разд\.\s*)(\d+(?:\.\d+)?[аб]?)")

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        out, fence = [], False
        for line in str(text).split("\n"):
            if line.strip().startswith("```"):
                fence = not fence; out.append(line); continue
            if fence:
                out.append(line); continue
            if self.card_re:
                line = self.card_re.sub(lambda m: f"[[{m.group(1)}]]", line)

            def sec(m):
                num = m.group(2)
                note = self.sec.get(num) or self.sec.get(num.split(".")[0])
                return f"[[{note}|{m.group(0)}]]" if note else m.group(0)
            line = self.sec_re.sub(sec, line)
            out.append(line)
        return "\n".join(out)


def split_plan(text: str):
    """→ (преамбула, [(уровень, номер, заголовок, тело)], журнал)."""
    jhead = "## Журнал изменений"
    body, journal = (text.split(jhead, 1) + [""])[:2] if jhead in text else (text, "")
    parts = re.split(r"^(##+) (.+)$", body, flags=re.M)
    pre = parts[0]
    secs = []
    for i in range(1, len(parts), 3):
        level, title, content = len(parts[i]), parts[i + 1].strip(), parts[i + 2]
        m = re.match(r"(\d+[аб]?(?:\.\d+[аб]?)?)\.?\s*(.*)", title)
        num = m.group(1) if m else ""
        secs.append((level, num, title, content))
    return pre, secs, journal


def split_journal(text: str):
    pat = re.compile(r"^\*\*(\d\d)\.(\d\d)\.(\d{4})(?:, (\d\d):(\d\d) MSK)?\*\* — (.+)$", re.M)
    ms = list(pat.finditer(text))
    entries = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
        dd, mm, yy, hh, mi, topic = m.groups()
        entries.append({"date": f"{yy}-{mm}-{dd}", "time": f"{hh or '00'}:{mi or '00'}", "topic": topic.strip(), "body": text[m.end():end].strip()})
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vault", type=Path, default=ROOT / "ObsidianFlyVis")
    args = ap.parse_args()
    V = Vault(args.vault)
    manifest = args.vault / ".generated.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []

    H = json.loads((ROOT / "experiments/project_history.json").read_text(encoding="utf-8"))
    G = json.loads((ROOT / "experiments/metrics_glossary.json").read_text(encoding="utf-8"))
    cards = {e["id"]: e for e in H["experiments"]}
    areas = {a["id"]: a for a in H["areas"]}
    plan_text = (ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md").read_text(encoding="utf-8")
    pre, secs, journal_text = split_plan(plan_text)

    # имена заметок плана
    sec_note, plan_notes = {}, []
    for level, num, title, content in secs:
        name = f"05 План/{safe(title)}"
        plan_notes.append((level, num, title, content, name))
        if num and num not in sec_note:
            sec_note[num] = name
    L = Linker(cards.keys(), sec_note)

    # --- термины и метрики ---
    term_note = {}
    for kind, items in (("метрика", G.get("metrics", {})), ("термин", G.get("terms", {}))):
        for key, d in items.items():
            term_note[key] = f"04 Термины и метрики/{safe(d.get('title', key))}"
    for a, k in G.get("aliases", {}).items():
        if k in term_note:
            term_note.setdefault(a, term_note[k])
    metric_users = defaultdict(set)
    for e in cards.values():
        for m in e.get("metrics") or []:
            if m.get("key") in term_note:
                metric_users[term_note[m["key"]]].add(e["id"])
    for kind, items in (("метрика", G.get("metrics", {})), ("термин", G.get("terms", {}))):
        for key, d in items.items():
            note = term_note[key]
            aliases = [key] + [a for a, k in G.get("aliases", {}).items() if k == key]
            body = f"# {d.get('title', key)}\n\n"
            if d.get("full"):
                body += f"*{d['full']}*\n\n"
            body += f"{d.get('ru', '')}\n\n"
            rows = [(lbl, d.get(f)) for lbl, f in (("Лучше", "better"), ("Единица", "unit"), ("Диапазон", "range"), ("Чем считается", "tool"), ("Оговорка", "caveat"), ("Источник", "source")) if d.get(f)]
            for lbl, v in rows:
                body += f"- **{lbl}:** {v}\n"
            users = sorted(metric_users.get(note, []))
            if users:
                body += "\n## Где используется\n" + "".join(f"- [[{u}]]\n" for u in users)
            V.write(note + ".md", fm({"aliases": sorted(set(aliases)), "tags": [f"тип/{kind}"]}) + body)

    # --- карточки ---
    for e in cards.values():
        a = areas.get(e["area"], {"title": e["area"]})
        area_note = f"03 Направления/{safe(a['title'])}"
        tags = ["эксперимент", f"исход/{e['state']}", f"направление/{tag(e['area'])}"] + [f"метка/{tag(t)}" for t in e.get("tags", [])]
        if e["area"] == "flyvis" or e["id"] in {i for _, ids in FLY_THEMES for i in ids}:
            tags.append("муха")
        props = {"aliases": [e["title"]], "tags": tags, "исход": STATE_RU.get(e["state"], e["state"]), "направление": a["title"],
                 "начат": e.get("started") or None, "завершён": e.get("finished") or None}
        body = f"# {STATE_ICON.get(e['state'], '')} {e['title']}\n\n"
        body += f"**Направление:** [[{area_note}|{a['title']}]] · **Исход:** {STATE_RU.get(e['state'], e['state'])} · `{e['id']}`\n\n"
        if e.get("summary"):
            body += f"{L(e['summary'])}\n\n"
        for f, lbl in FIELDS:
            if e.get(f):
                body += f"## {lbl}\n{L(e[f])}\n\n"
        cond = [(k, e.get(k)) for k in ("baseline", "variant", "dataset", "split", "detector", "tracker", "seeds", "checkpoint") if e.get(k)]
        if cond:
            body += "## Условия\n" + "".join(f"- **{k}:** {L(str(v))}\n" for k, v in cond) + "\n"
        if e.get("acceptance"):
            body += "## Критерии приёмки (записаны до прогона)\n"
            for c in e["acceptance"]:
                mark = {True: "✅", False: "❌", None: "⏳"}.get(c.get("met"), "⏳")
                body += f"- {mark} {L(c.get('criterion', ''))}" + (f"  \n  *{L(c['evidence'])}*" if c.get("evidence") else "") + "\n"
            body += "\n"
        if e.get("metrics"):
            body += "## Метрики\n| метрика | значение | baseline | подмножество | источник |\n|---|---|---|---|---|\n"
            for m in e["metrics"]:
                k = m.get("key", ""); kn = term_note.get(k)
                kl = f"[[{kn}|{k}]]" if kn else k
                body += f"| {kl} | {m.get('value')} | {m.get('baseline')} | {m.get('subset', '')} | `{m.get('source', '')}` · {m.get('locator', '')} |\n"
            body += "\n"
        if e.get("evidence"):
            body += "## Доказательства\n"
            for ev in e["evidence"]:
                ref = ev.get("ref", ""); loc = ev.get("locator", "")
                link = L(f"{loc}") if ev.get("kind") == "plan_section" else f"`{ref}` · {loc}"
                body += f"- {ev.get('claim', ev.get('kind', ''))}: {link}\n"
            body += "\n"
        for f, lbl in (("files", "Код"), ("runs", "Данные"), ("commits", "Коммиты")):
            if e.get(f):
                body += f"## {lbl}\n" + "".join(f"- `{x}`\n" for x in e[f]) + "\n"
        V.write(f"02 Эксперименты/{e['id']}.md", fm(props) + body)

    # --- направления ---
    by_area = defaultdict(list)
    for e in cards.values():
        by_area[e["area"]].append(e)
    for aid, a in areas.items():
        es = sorted(by_area.get(aid, []), key=lambda x: x.get("started") or "")
        body = f"# {a['title']}\n\n{a.get('description', '')}\n\n## Эксперименты ({len(es)})\n"
        body += "".join(f"- {STATE_ICON.get(x['state'], '')} [[{x['id']}|{x['title']}]] — {STATE_RU.get(x['state'], x['state'])}, {x.get('started') or '—'}\n" for x in es)
        V.write(f"03 Направления/{safe(a['title'])}.md", fm({"tags": ["направление"] + (["муха"] if aid == "flyvis" else [])}) + body)

    # --- план ---
    children = defaultdict(list)
    parent = None
    for level, num, title, content, name in plan_notes:
        if level == 2:
            parent = name
        elif parent:
            children[parent].append(name)
    for level, num, title, content, name in plan_notes:
        up = None
        if level > 2:
            for p, ch in children.items():
                if name in ch:
                    up = p
        body = f"# {title}\n\n" + (f"↑ [[{up}]] · [[05 План/Единый план|Единый план]]\n\n" if up else "↑ [[05 План/Единый план|Единый план]]\n\n")
        body += L(content.strip()) + "\n"
        if children.get(name):
            body += "\n## Подразделы\n" + "".join(f"- [[{c}]]\n" for c in children[name])
        V.write(name + ".md", fm({"tags": ["план"] + (["муха"] if re.search(r"мух|FlyVis", title, re.I) else []), "раздел": num or None}) + body)
    idx = "# Единый план\n\nИсточник — `dissertation/ЕДИНЫЙ_ПЛАН.md` (главный документ проекта). Разделы:\n\n"
    idx += "".join(f"- [[{n}]]\n" for lv, num, t, c, n in plan_notes if lv == 2) + "- [[06 Журнал/Журнал изменений|Журнал изменений]]\n\n" + L(pre.strip()) + "\n"
    V.write("05 План/Единый план.md", fm({"tags": ["план"]}) + idx)

    # --- журнал ---
    entries = split_journal(journal_text)
    jnames = []
    for en in entries:
        nm = f"06 Журнал/{en['date']} {en['time'].replace(':', '-')} {safe(en['topic'], 70)}"
        fly = bool(re.search(r"мух|FlyVis|сетчат|коннектом", en["topic"] + en["body"][:400], re.I))
        V.write(nm + ".md", fm({"tags": ["журнал"] + (["муха"] if fly else []), "дата": en["date"], "время": en["time"]}) +
                f"# {en['date']} {en['time']} — {en['topic']}\n\n↑ [[06 Журнал/Журнал изменений|Журнал изменений]]\n\n{L(en['body'])}\n")
        jnames.append((en, nm, fly))
    by_day = defaultdict(list)
    for en, nm, fly in jnames:
        by_day[en["date"]].append((en, nm, fly))
    jidx = "# Журнал изменений\n\nЗаписи из `dissertation/ЕДИНЫЙ_ПЛАН.md`, новые сверху. 🪰 — запись о мухе.\n\n"
    for d in sorted(by_day, reverse=True):
        jidx += f"## {d}\n" + "".join(f"- {'🪰 ' if fly else ''}[[{nm}|{en['time']} — {en['topic']}]]\n" for en, nm, fly in sorted(by_day[d], key=lambda x: x[0]['time'], reverse=True)) + "\n"
    V.write("06 Журнал/Журнал изменений.md", fm({"tags": ["журнал"]}) + jidx)

    # --- исследования ---
    for rdir, rtitle in RESEARCH.items():
        rp = ROOT / rdir
        if not (rp / "outline.yaml").exists():
            continue
        outline = yaml.safe_load((rp / "outline.yaml").read_text(encoding="utf-8"))
        items_md = []
        for it in outline.get("items", []):
            slug = re.sub(r"[^\w]", "", it["name"].replace(" ", "_"))
            res = rp / "results" / f"{slug}.json"
            nm = f"07 Исследования/{rtitle}/{safe((it.get('id', '') + ' ' + it['name']).strip())}"
            body = f"# {it.get('id', '')} {it['name']}\n\n↑ [[07 Исследования/{rtitle}/{rtitle}|{rtitle}]]\n\n"
            if it.get("focus"):
                body += f"> {it['focus']}\n\n"
            if res.exists():
                d = json.loads(res.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if k in ("id", "name", "block", "uncertain"):
                        continue
                    body += f"## {FIELD_RU.get(k, k)}\n{L(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))}\n\n"
                if d.get("uncertain"):
                    body += f"*Неуверенные поля: {', '.join(d['uncertain'])}*\n"
            else:
                body += "*Результата нет (тема не исследовалась).* \n"
            V.write(nm + ".md", fm({"tags": ["исследование"] + (["муха"] if re.search(r"мух|FlyVis|fly", it["name"] + str(it.get("focus", "")), re.I) else [])}) + body)
            items_md.append((it, nm, res.exists()))
        summ = rp / "СВОДКА.md"
        head = f"# {rtitle}\n\nТема: {outline.get('topic', '')}\n\n"
        if summ.exists():
            head += f"## Сводка\n{L(summ.read_text(encoding='utf-8'))}\n\n"
        head += "## Темы\n" + "".join(f"- {'✅' if ok else '·'} [[{nm}|{it.get('id', '')} {it['name']}]]\n" for it, nm, ok in items_md)
        V.write(f"07 Исследования/{rtitle}/{rtitle}.md", fm({"tags": ["исследование"]}) + head)

    # --- документы ---
    docs = [(p, "08 Документы") for p in sorted((ROOT / "dissertation").glob("*.md")) if p.name != "ЕДИНЫЙ_ПЛАН.md"]
    docs += [(p, "08 Документы/Черновики глав") for p in sorted((ROOT / "dissertation/drafts").glob("*.md"))]
    docs += [(p, "08 Документы/Архив 2026-09-07") for p in sorted((ROOT / "dissertation/backup_20260907").rglob("*.md"))]
    docs += [(ROOT / n, "08 Документы/Корень проекта") for n in ("FLYVIS_TRACKER_RESEARCH_ROADMAP.md", "FLYVIS_TRACKER_GATE_REPORT.md", "FLYVIS_STRESS_AND_SECOND_ASSOCIATION_REPORT.md",
                                                             "FLYVIS_VIDEO_RATE_EXPERIMENT_REPORT.md", "TRACKER_1_0.md", "09092026.md", "СВОДНЫЙ_ПЛАН1.md", "ПЛАН_DRB_v3.md", "README.md", "AGENTS.md") if (ROOT / n).exists()]
    doc_list = []
    for p, folder in docs:
        nm = f"{folder}/{safe(p.stem)}"
        txt = p.read_text(encoding="utf-8", errors="replace")
        fly = bool(re.search(r"мух|FlyVis", p.stem + txt[:2000], re.I))
        V.write(nm + ".md", fm({"tags": ["документ"] + (["муха"] if fly else []), "источник": str(p.relative_to(ROOT)).replace("\\", "/")}) + L(txt))
        doc_list.append((nm, folder, fly))
    pubs = sorted((ROOT / "dissertation/публикации").glob("*.pdf"))
    for p in pubs:
        V.copy(p, f"08 Документы/Публикации/{p.name}")
    didx = "# Документы\n\n"
    for folder in dict.fromkeys(f for _, f in docs):
        didx += f"## {folder.split('/')[-1]}\n" + "".join(f"- {'🪰 ' if fly else ''}[[{nm}|{nm.split('/')[-1]}]]\n" for nm, f, fly in doc_list if f == folder) + "\n"
    if pubs:
        didx += "## Публикации\n" + "".join(f"- ![[08 Документы/Публикации/{p.name}]]\n" for p in pubs)
        docx = sorted((ROOT / "dissertation/публикации").glob("*.docx"))
        if docx:
            didx += "".join(f"- `{p.name}` (Word, лежит в `dissertation/публикации/`)\n" for p in docx)
    V.write("08 Документы/Документы.md", fm({"tags": ["документ"]}) + didx)

    # --- муха ---
    build_fly(V, cards, L, jnames)

    # --- главная ---
    n_state = defaultdict(int)
    for e in cards.values():
        n_state[e["state"]] += 1
    home = "# Диссертация: трекинг и муха (FlyVis)\n\nХранилище собрано скриптом `scripts/build_obsidian_vault.py` из документации проекта; пересобирается без потери ваших заметок.\n\n"
    home += "## С чего начать\n- 🪰 [[01 Муха/Муха (FlyVis)|Муха — главная заметка]] и [[01 Муха/Карта мухи.canvas|карта мухи]]\n- [[05 План/Единый план|Единый план]] · [[06 Журнал/Журнал изменений|Журнал]]\n"
    home += "- [[08 Документы/Документы|Документы]] · " + " · ".join(f"[[07 Исследования/{t}/{t}|{t}]]" for t in RESEARCH.values()) + "\n\n"
    home += "## Направления\n" + "".join(f"- [[03 Направления/{safe(a['title'])}|{a['title']}]] ({len(by_area.get(aid, []))})\n" for aid, a in sorted(areas.items(), key=lambda x: x[1].get("order", 0)))
    home += "\n## Эксперименты по исходам\n" + "".join(f"- {STATE_ICON.get(s, '')} {STATE_RU.get(s, s)}: {n}\n" for s, n in sorted(n_state.items(), key=lambda x: -x[1]))
    home += "\n## Как читать граф\nЦвета: оранжевый — муха; зелёный / красный / жёлтый — исход эксперимента; синий — план; серый — журнал; фиолетовый — термины; бирюзовый — исследования.\n"
    V.write("00 Главная.md", fm({"tags": ["главная"]}) + home)

    graph_colors(args.vault)
    # удалить устаревшие собственные файлы
    stale = [r for r in old if r not in V.written]
    for r in stale:
        p = args.vault / r
        if p.exists() and p.is_file():
            p.unlink()
    manifest.write_text(json.dumps(sorted(set(V.written)), ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[OK] заметок/файлов: {len(set(V.written))}, удалено устаревших: {len(stale)} → {args.vault}")
    return 0


def fly_link(cards, cid, label=None):
    if cid in cards:
        return f"[[{cid}|{label or cards[cid]['title']}]]"
    return label or cid


def build_fly(V: Vault, cards: dict, L: Linker, jnames) -> None:
    c = lambda cid, label=None: fly_link(cards, cid, label)
    concepts = {
        "Что такое муха": f"""Муха (FlyVis) — модель зрения плодовой мушки дрозофилы, построенная по реальной схеме связей её нервной системы (коннектому): 45 669 нейронов 65 типов и 1,5 млн связей (Lappalainen et al., Nature 2024). Схему связей взяли из анатомии, а настраиваемыми оставили только 734 числа — скорости нейронов, их уровни покоя и силу связей между типами. Авторы настроили эти числа на одну задачу — оценивать движение в мультфильмах Sintel.

**Место в нашем конвейере.** YOLO26 находит объекты → трекер ByteTrack связывает рамки по геометрии → на каждый трек сажается свой глаз мухи ([[01 Муха/Глаз на трек|глаз на трек]]), он смотрит на кусочек видео вокруг объекта и выдаёт [[01 Муха/Отпечаток мухи|отпечаток]] и оценку движения. Муха — вторая нейросеть, но с готовой «анатомией» из биологии.

**Главный вопрос диссертации:** даёт ли схема связей живой мухи то, чего не даёт обычная сеть того же размера на тех же данных.""",
        "Коннектом и 734 параметра": f"""Схема связей (коннектом) фиксирована: какие типы нейронов с какими соединены, знак и число синапсов. Обучаются 734 числа: 65 постоянных времени, 65 потенциалов покоя, 604 силы связей «тип → тип». Поэтому любое дообучение остаётся «мушиным» по устройству, но может потерять «мушиность» по работе — её проверяют сравнением с биологией (функции `dsi_correlation_to_known`, `fri_correlation_to_known` в flyvis).

Проверка значимости схемы — [[01 Муха/Нулевая модель коннектома|нулевая модель]]. Устойчивость выводов к выбору модели — [[01 Муха/Ансамбль FlyVis|ансамбль]].""",
        "Сетчатка-ламина": f"""Передняя ступень глаза, которой нет в FlyVis: фоторецептор подстраивается под освещённость (закон Вебера), ламина выделяет локальный контраст. Без неё муха ночью проваливалась.

- Сетчатка устраняет ночной провал мухи: {c('flyvis_retina_stageB_v1')}.
- Выигрыш специфичен для мухи — оптическому потоку DIS та же сетчатка не помогает: {c('dis_retina_control_v1')}.
- Перед замороженным детектором YOLO кадры после сетчатки вредят (сдвиг входа): {c('fly_eye_frontend_v1')}.
- Первая разведка: {c('flyvis_retina_frontend_v1')}.""",
        "Нейроны движения T4-T5": f"""T4 и T5 — нейроны мухи, чувствительные к направлению движения (светлые и тёмные края). На них держится оценка движения мухой.

- Движение мухи на наших видео много хуже DIS: {c('flyvis_stress')}, {c('flyvis_fovea_motion_v1')}, {c('motion_reliability_v1')}.
- Линейный декодер на активности мухи хуже штатного, но муха несёт больше движения, чем её вход: {c('fly_decoder_refit_v1')}.
- Собственная уверенность мухи в движении антиинформативна: {c('motion_confidence_audit')}.
- Малые объекты у мухи (нейрон LC11) питают T2/T3, а не T4/T5 — см. [[07 Исследования/Детектор с мухой/Детектор с мухой|исследование детектора]].""",
        "Глаз на трек": f"""На каждый трек сажается свой фовеальный глаз мухи (721 фасетка) — EyeBank. Глаз смотрит на окрестность рамки на всех кадрах 30 Гц, его внутреннее состояние «живёт» вместе с треком. По устройству это уже «один глаз — один трек» — основа для [[01 Муха/Трекер на мухе|трекера, где решения принимает муха]].

- Считывание глаза для событий трека: {c('flyvis_foveation_readout_v1')}, {c('flyvis_event_features_v1')}.""",
        "Отпечаток мухи": f"""Отпечаток (сигнатура) — сводка активности 34 выходных типов глаза мухи за начало и конец жизни трека. По нему решается «тот же объект или другой» при сшивке обрывков треков.

- Отпечаток мухи несёт идентичность: к геометрии +0,14 AP — {c('fly_signature_reid_v1')}.
- Но почти вся информация уже есть во входе глаза: поверх входа муха добавляет +0,006 (ночью +0,010) — {c('fly_matched_controls_v1')}.
- Сшивка треков по одной геометрии даёт +0,79 HOTA, внешность и муха сверх неё незначимы — {c('stitch_appearance_v1')}.""",
        "Нулевая модель коннектома": f"""Проверка «важна ли схема связей»: у той же мухи связи перемешивают с сохранением степеней и знаков.

- {c('fly_connectome_null_v1')}: перемешанная муха хуже настоящей (+0,044 AP у мухи одной), но вклад поверх входа на грани.
- **Поправка 03.10:** перемешивание делалось без переобучения — это проверка расстройки обученной сети, а не топологии; при переобучении перемешанной сети разница исчезает (Dhiman 2026). Заявлять «топология значима» нельзя — [[07 Исследования/Фундаментальные проблемы/Фундаментальные проблемы|исследование фундаментальных проблем]], F17.""",
        "Ансамбль FlyVis": f"""Авторы FlyVis обучили 50 моделей с одной схемой связей (ансамбль); модели приходят к разным решениям при почти равной ошибке. Все наши выводы были получены на одной — `flow/0000/000` (лучшая по Sintel).

- Проверка на моделях 012, 025, 037: {c('fly_ensemble_members_v1')}.
- Обоснование — исследование фундаментальных проблем, тема F13.""",
        "Линия A — трекер на мухе": f"""Собственный трекер, где движение и связывание шли через муху. Обнаружение было даже чуть выше, но связывание провалилось: HOTA 29,1 против 52,4 у ByteTrack, переключений ID 3 384 против 573 (val7) — главная причина в ошибке движения мухи.

- {c('flyvis_standalone')}, {c('flyvis_stress')}.
- План возвращения к мухе как трекеру — [[01 Муха/Трекер на мухе|трекер, где решения принимает муха]].""",
        "Линия B — муха как помощник": f"""Трекер XYWHByteTrack со своими модулями, муха — источник признаков. Итоговый трекер: HOTA 58,47 → 59,27 ({c('line_b_combined_v1')}), но DetA не изменилась — потолок в обнаружении. Сшивка по геометрии, каскад буферов, отложенное рождение: {c('stitch_appearance_v1')}, {c('cbiou_cascade_v1')}, {c('birth_deferred_train_v1')}.""",
        "Трекер на мухе": f"""Решение автора 04.10: сделать трекер, где все решения ByteTrack принимает муха (план 7.8). Этапы: жизнь трека → рождение → сопоставление по отпечатку → движение (после дообучения мухи) → сборка против ByteTrack.

- Мишень первого этапа: у трети реальных потерь на пороге объект уверенно виден в прошлых 5 кадрах — {c('past_view_oracle_v1')}; простое продолжение по Калману вредит — {c('track_continuation_v1')}.
- Первый этап: {c('fly_tracker_lifecycle_v1')}.
- Подробно — [[05 План/7.8. Трекер, где решения принимает муха, начато 04.10|раздел 7.8 плана]].""",
        "Дообучение мухи": f"""Муху на нашей задаче ещё никто не обучал: 734 числа — с Sintel. Рецепт (исследование фундаментальных проблем, F18): сперва вход и декодер, затем постоянные времени и потенциалы покоя с притяжением к решению Sintel, все 734 — только при сохранении «мушиности»; минимум 3 модели ансамбля; контроль — обычная сеть того же размера.

- Почему линейного декодера мало: {c('fly_decoder_refit_v1')}.
- Кандидат К1 в отборе экспериментов — [[05 План/7.6. Отбор следующих экспериментов по схеме RPM, 03.10|раздел 7.6 плана]].""",
    }
    for title, text in concepts.items():
        V.write(f"01 Муха/{title}.md", fm({"tags": ["муха", "понятие"]}) + f"# {title}\n\n↑ [[01 Муха/Муха (FlyVis)|Муха]]\n\n{text}\n")

    fly_cards = sorted({cid for _, ids in FLY_THEMES for cid in ids if cid in cards} | {cid for cid, e in cards.items() if e["area"] == "flyvis"},
                       key=lambda x: cards[x].get("started") or "")
    themed = {cid for _, ids in FLY_THEMES for cid in ids}
    hub = "# 🪰 Муха (FlyVis)\n\n" + concepts["Что такое муха"] + "\n\n## Понятия\n" + "".join(f"- [[01 Муха/{t}|{t}]]\n" for t in concepts if t != "Что такое муха")
    hub += "\n- [[01 Муха/Что можно и нельзя заявлять о мухе|Что можно и нельзя заявлять]]\n- [[01 Муха/Хронология мухи|Хронология]]\n- [[01 Муха/Карта мухи.canvas|Карта мухи (холст)]]\n\n## Эксперименты по темам\n"
    for theme, ids in FLY_THEMES:
        present = [i for i in ids if i in cards]
        if present:
            hub += f"### {theme}\n" + "".join(f"- {STATE_ICON.get(cards[i]['state'], '')} [[{i}|{cards[i]['title']}]]\n" for i in present) + "\n"
    rest = [i for i in fly_cards if i not in themed]
    if rest:
        hub += "### Прочее о мухе\n" + "".join(f"- {STATE_ICON.get(cards[i]['state'], '')} [[{i}|{cards[i]['title']}]]\n" for i in rest) + "\n"
    hub += "## Где муха в плане и исследованиях\n- [[05 План/7.4. FlyVis и раздельная уверенность, начато 13.09|7.4 FlyVis и раздельная уверенность]]\n- [[05 План/7.5. Детектор с мухой, начато 01.10|7.5 Детектор с мухой]]\n- [[05 План/7.7. Фундаментальные проблемы и порядок борьбы, 03.10|7.7 Фундаментальные проблемы]]\n- [[05 План/7.8. Трекер, где решения принимает муха, начато 04.10|7.8 Трекер на мухе]]\n"
    hub += "- " + " · ".join(f"[[07 Исследования/{t}/{t}|{t}]]" for t in RESEARCH.values()) + "\n"
    V.write("01 Муха/Муха (FlyVis).md", fm({"tags": ["муха", "главная"], "aliases": ["FlyVis", "Муха"]}) + hub)

    claims = "# Что можно и нельзя заявлять о мухе\n\nСобрано из полей «Решение» и «Нельзя заявлять» карточек мухи (источник — каталог экспериментов).\n\n"
    for cid in fly_cards:
        e = cards[cid]
        if e.get("decision") or e.get("cannot_claim"):
            claims += f"## {STATE_ICON.get(e['state'], '')} [[{cid}|{e['title']}]]\n"
            if e.get("decision"):
                claims += f"**Решение.** {L(e['decision'])}\n\n"
            if e.get("cannot_claim"):
                claims += f"**Нельзя заявлять.** {L(e['cannot_claim'])}\n\n"
    V.write("01 Муха/Что можно и нельзя заявлять о мухе.md", fm({"tags": ["муха"]}) + claims)

    tl = "# Хронология мухи\n\n## Эксперименты\n| начат | завершён | исход | эксперимент |\n|---|---|---|---|\n"
    tl += "".join(f"| {cards[i].get('started') or '—'} | {cards[i].get('finished') or '—'} | {STATE_ICON.get(cards[i]['state'], '')} {STATE_RU.get(cards[i]['state'], '')} | [[{i}|{cards[i]['title']}]] |\n" for i in fly_cards)
    tl += "\n## Записи журнала о мухе\n" + "".join(f"- {en['date']} {en['time']} — [[{nm}|{en['topic']}]]\n" for en, nm, fly in sorted(jnames, key=lambda x: (x[0]['date'], x[0]['time'])) if fly)
    V.write("01 Муха/Хронология мухи.md", fm({"tags": ["муха"]}) + tl)

    # холст: конвейер сверху, темы экспериментов ниже
    color = {"completed_positive": "4", "completed_negative": "1", "inconclusive": "3", "running": "5", "planned": "6"}
    nodes, edges = [], []
    pipe = [("p1", "Видео 30 Гц"), ("p2", "YOLO26\nдетектор"), ("p3", "ByteTrack\nтрекер"), ("p4", "Сетчатка-ламина"), ("p5", "Глаз мухи на трек\n(коннектом, 734 параметра)"), ("p6", "Отпечаток · движение\n→ решения трекера")]
    for k, (nid, text) in enumerate(pipe):
        nodes.append({"id": nid, "type": "text", "text": f"**{text}**", "x": k * 300, "y": -400, "width": 260, "height": 90, "color": "2" if nid in ("p4", "p5", "p6") else "0"})
    for a, b in (("p1", "p2"), ("p2", "p3"), ("p1", "p4"), ("p4", "p5"), ("p5", "p6"), ("p6", "p3")):
        edges.append({"id": f"e{a}{b}", "fromNode": a, "fromSide": "right" if a != "p6" else "top", "toNode": b, "toSide": "left" if b != "p3" else "top"})
    x = 0
    for g, (theme, ids) in enumerate(FLY_THEMES):
        present = [i for i in ids if i in cards]
        if not present:
            continue
        gid = f"g{g}"
        h = 120 + 110 * len(present)
        nodes.append({"id": gid, "type": "group", "label": theme, "x": x, "y": -200, "width": 340, "height": h})
        for k, i in enumerate(present):
            nodes.append({"id": f"n{g}_{k}", "type": "file", "file": f"02 Эксперименты/{i}.md", "x": x + 20, "y": -150 + 110 * k, "width": 300, "height": 90, **({"color": color[cards[i]['state']]} if cards[i]['state'] in color else {})})
        x += 380
    nodes.append({"id": "legend", "type": "text", "text": "Цвет карточки: зелёный — положительный, красный — отрицательный, жёлтый — неопределённый, голубой — идёт, фиолетовый — запланирован.", "x": 0, "y": -560, "width": 900, "height": 70})
    V.write("01 Муха/Карта мухи.canvas", json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=1))


def graph_colors(vault: Path) -> None:
    p = vault / ".obsidian" / "graph.json"
    g = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rgb = lambda h: int(h.lstrip("#"), 16)
    g["colorGroups"] = [
        {"query": "tag:#муха", "color": {"a": 1, "rgb": rgb("#f28c28")}},
        {"query": "tag:#исход/completed_positive", "color": {"a": 1, "rgb": rgb("#3fb950")}},
        {"query": "tag:#исход/completed_negative", "color": {"a": 1, "rgb": rgb("#e5534b")}},
        {"query": "tag:#исход/inconclusive", "color": {"a": 1, "rgb": rgb("#d4a72c")}},
        {"query": 'path:"05 План"', "color": {"a": 1, "rgb": rgb("#4c8ef7")}},
        {"query": 'path:"06 Журнал"', "color": {"a": 1, "rgb": rgb("#8b949e")}},
        {"query": 'path:"04 Термины и метрики"', "color": {"a": 1, "rgb": rgb("#a371f7")}},
        {"query": 'path:"07 Исследования"', "color": {"a": 1, "rgb": rgb("#39c5bb")}},
    ]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
