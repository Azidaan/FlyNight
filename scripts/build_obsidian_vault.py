"""Собрать хранилище Obsidian из документации проекта (ObsidianFlyVis).

Источники (только чтение): dissertation/ЕДИНЫЙ_ПЛАН.md (разделы и журнал), experiments/project_history.json
(карточки, направления), experiments/metrics_glossary.json (метрики, термины), research_*/ (исследования),
прочие документы dissertation/ и корня. Заметки мухи — обзор поверх карточек, числа в них ссылаются на
карточки-источники.

Устройство графа — семь блоков, каждый своим кластером:
  01 Муха (понятия, темы, ВСЕ эксперименты мухи), 02 Эксперименты (остальные, по направлениям),
  05 План, 06 Журнал, 04 Термины и метрики, 07 Исследования, 08 Документы; 00 Главная связывает центры блоков.
Внутри блока ссылки — обычные [[вики-ссылки]] (рисуются в графе). Между блоками — ссылки obsidian://open
(кликабельны, но рёбер в графе не дают), чтобы блоки не слипались.

Безопасность: скрипт пишет только свои файлы хранилища и ведёт их список (.generated.json); при пересборке
удаляет только файлы из этого списка. Чужие заметки не трогает. Настройки графа (.obsidian/graph.json)
дополняются цветами и силами; остальные ключи сохраняются. Obsidian держит настройки графа в памяти и
перезаписывает файл — пересобирать при закрытом Obsidian (или потом «Перезагрузить без сохранения»).

Запуск: python scripts/build_obsidian_vault.py [--vault ObsidianFlyVis]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import yaml

ROOT = Path(__file__).resolve().parents[1]
VAULT_NAME = "ObsidianFlyVis"

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

FLY_THEMES = [
    ("Глаз и сетчатка-ламина", ["flyvis_retina_frontend_v1", "flyvis_retina_stageB_v1", "dis_retina_control_v1", "fly_eye_frontend_v1"]),
    ("Движение мухи", ["flyvis_stress", "motion_confidence_audit", "flyvis_fovea_motion_v1", "motion_reliability_v1", "e2e_motion_source_v1", "fly_decoder_refit_v1"]),
    ("Считывание активности и события", ["flyvis_internal_readout_v1", "flyvis_foveation_readout_v1", "flyvis_readout_nonlinear_diag", "flyvis_event_features_v1", "qmatch_gate", "qmatch_30hz_pseudo", "qmatch_candidate_shift"]),
    ("Отпечаток мухи и опознание", ["fly_signature_reid_v1", "fly_matched_controls_v1", "fly_connectome_null_v1", "stitch_appearance_v1"]),
    ("Устойчивость выводов — ансамбль", ["fly_ensemble_members_v1"]),
    ("Трекер на мухе", ["flyvis_standalone", "fly_tracker_lifecycle_v1"]),
]
FLY_OTHER = "Прочее о мухе"
FINAL = {"completed_positive", "completed_negative", "inconclusive", "closed", "superseded"}
PLANNED = {"planned", "ready", "running"}
JOURNAL_RESULT = re.compile(r"закрыт|ЗАКРЫТ|отрицательн|положительн|completed_|inconclusive|итог|сведен|не помогает|не возвращает|вредит|провал", re.I)

HUB = {"09": "09 Запланировано/Запланировано", "01": "01 Муха/Муха (FlyVis)", "02": "02 Эксперименты/Эксперименты", "04": "04 Термины и метрики/Термины и метрики",
       "05": "05 План/Единый план", "06": "06 Журнал/Журнал изменений", "07": "07 Исследования/Исследования", "08": "08 Документы/Документы"}
BLOCK_TITLE = {"09": "Запланировано", "01": "Муха (FlyVis)", "02": "Эксперименты", "04": "Термины и метрики", "05": "Единый план", "06": "Журнал изменений", "07": "Исследования", "08": "Документы"}


def _trim(name: str, n: int) -> str:
    return name[:n].strip().rstrip(".").strip()


def safe(name: str, n: int = 120) -> str:
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name)
    return _trim(re.sub(r"\s+", " ", name).strip(), n)


def tag(s: str) -> str:
    return re.sub(r"[^\w/-]", "-", s).strip("-")


def fm(d: dict) -> str:
    return "---\n" + yaml.safe_dump({k: v for k, v in d.items() if v not in (None, [])}, allow_unicode=True, sort_keys=False, width=10000).strip() + "\n---\n\n"


def block(path: str) -> str:
    return path[:2]


def lab(s: str) -> str:
    return str(s).replace("[", "(").replace("]", ")").replace("|", "/")


def ref(src_block: str, path: str, label: str, structural: bool = False) -> str:
    """Структурная ссылка внутри блока — вики-ссылка (ребро графа: дерево «центр → группа → заметка»).
    Все остальные ссылки (по тексту, между заметками одного блока и между блоками) — obsidian://open:
    кликабельны, но рёбер в графе не дают, чтобы каждый блок оставался деревом, а не клубком."""
    label = lab(label)
    if structural and block(path) == src_block:
        return f"[[{path}|{label}]]"
    return f"[{label}↗](obsidian://open?vault={quote(VAULT_NAME)}&file={quote(path, safe='')})"


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
    """Ссылки на карточки (по id) и разделы плана (§7.7, раздел 7.5) в произвольном тексте."""

    def __init__(self, card_path: dict, cards: dict, sec_note: dict):
        self.card_path, self.cards, self.sec = card_path, cards, sec_note
        ids = sorted(card_path, key=len, reverse=True)
        self.card_re = re.compile(r"`?(?<![\w/\[(=])(" + "|".join(map(re.escape, ids)) + r")(?![\w\]])`?")
        self.sec_re = re.compile(r"(§\s?|раздел[а-яё]*\s+|разд\.\s*)(\d+(?:\.\d+)?[аб]?)")

    def __call__(self, text, src: str) -> str:
        if not text:
            return ""
        out, fence = [], False
        for line in str(text).split("\n"):
            if line.strip().startswith("```"):
                fence = not fence; out.append(line); continue
            if fence:
                out.append(line); continue
            line = self.card_re.sub(lambda m: ref(src, self.card_path[m.group(1)], m.group(1)), line)

            def sec(m):
                num = m.group(2)
                note = self.sec.get(num) or self.sec.get(num.split(".")[0])
                return ref(src, note, m.group(0)) if note else m.group(0)
            line = self.sec_re.sub(sec, line)
            out.append(line)
        return "\n".join(out)


def split_plan(text: str):
    jhead = "## Журнал изменений"
    body, journal = (text.split(jhead, 1) + [""])[:2] if jhead in text else (text, "")
    parts = re.split(r"^(##+) (.+)$", body, flags=re.M)
    secs = []
    for i in range(1, len(parts), 3):
        level, title, content = len(parts[i]), parts[i + 1].strip(), parts[i + 2]
        m = re.match(r"(\d+[аб]?(?:\.\d+[аб]?)?)\.?\s*(.*)", title)
        secs.append((level, m.group(1) if m else "", title, content))
    return parts[0], secs, journal


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
    ap.add_argument("--vault", type=Path, default=ROOT / VAULT_NAME)
    args = ap.parse_args()
    V = Vault(args.vault)
    manifest = args.vault / ".generated.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []

    H = json.loads((ROOT / "experiments/project_history.json").read_text(encoding="utf-8"))
    G = json.loads((ROOT / "experiments/metrics_glossary.json").read_text(encoding="utf-8"))
    cards_all = {e["id"]: e for e in H["experiments"]}
    # в граф попадают только завершённые эксперименты и закрытые ветки (решение автора 04.10);
    # запланированные, идущие и инфраструктура остаются только в плане и каталоге
    cards = {i: e for i, e in cards_all.items() if e["state"] in FINAL}
    # запланированное и идущее — отдельный красный блок 09 (решение автора 04.10)
    planned = {i: e for i, e in cards_all.items() if e["state"] in PLANNED}
    areas = {a["id"]: a for a in H["areas"]}

    # где живёт каждая карточка: муха — в блоке 01 по темам, остальные — в 02 по направлениям
    fly_theme = {}
    for theme, ids in FLY_THEMES:
        for i in ids:
            if i in cards:
                fly_theme[i] = theme
    for i, e in cards.items():
        if e["area"] == "flyvis" and i not in fly_theme:
            fly_theme[i] = FLY_OTHER
    card_path = {i: (f"01 Муха/Эксперименты/{i}" if i in fly_theme else f"02 Эксперименты/{i}") for i in cards}
    card_path.update({i: f"09 Запланировано/{i}" for i in planned})
    plan_group = {aid: f"09 Запланировано/{safe(a['title'])} — планы" for aid, a in areas.items()}
    area_path = {aid: f"02 Эксперименты/Направления/{safe(a['title'])}" for aid, a in areas.items()}
    theme_path = {t: f"01 Муха/Темы/{safe(t)}" for t in [t for t, _ in FLY_THEMES] + [FLY_OTHER]}

    pre, secs, journal_text = split_plan((ROOT / "dissertation/ЕДИНЫЙ_ПЛАН.md").read_text(encoding="utf-8"))
    sec_note, plan_notes = {}, []
    for level, num, title, content in secs:
        name = f"05 План/{safe(title)}"
        plan_notes.append((level, num, title, content, name))
        if num and num not in sec_note:
            sec_note[num] = name
    L = Linker(card_path, cards, sec_note)

    # ---------------- 04 Термины и метрики ----------------
    term_note = {}
    for items in (G.get("metrics", {}), G.get("terms", {})):
        for key, d in items.items():
            term_note[key] = f"04 Термины и метрики/{safe(d.get('title', key))}"
    for a, k in G.get("aliases", {}).items():
        if k in term_note:
            term_note.setdefault(a, term_note[k])
    users = defaultdict(set)
    for e in cards.values():
        for m in e.get("metrics") or []:
            if m.get("key") in term_note:
                users[term_note[m["key"]]].add(e["id"])
    tlist = {"метрика": [], "термин": []}
    for kind, items in (("метрика", G.get("metrics", {})), ("термин", G.get("terms", {}))):
        for key, d in items.items():
            note = term_note[key]
            aliases = sorted({key} | {a for a, k in G.get("aliases", {}).items() if k == key})
            body = f"# {d.get('title', key)}\n\n↑ [[{HUB['04']}|{BLOCK_TITLE['04']}]]\n\n" + (f"*{d['full']}*\n\n" if d.get("full") else "") + f"{d.get('ru', '')}\n\n"
            for lbl, f in (("Лучше", "better"), ("Единица", "unit"), ("Диапазон", "range"), ("Чем считается", "tool"), ("Оговорка", "caveat"), ("Источник", "source")):
                if d.get(f):
                    body += f"- **{lbl}:** {d[f]}\n"
            if users.get(note):
                body += "\n## Где используется\n" + "".join(f"- {ref('04', card_path[u], cards[u]['title'])}\n" for u in sorted(users[note]))
            V.write(note + ".md", fm({"aliases": aliases, "tags": [f"тип/{kind}"]}) + body)
            tlist[kind].append((d.get("title", key), note))
    hub = f"# {BLOCK_TITLE['04']}\n\n↑ [[00 Главная|Главная]]\n\n"
    for kind, title in (("метрика", "Метрики"), ("термин", "Термины")):
        hub += f"## {title}\n" + "".join(f"- [[{n}|{t}]]\n" for t, n in sorted(tlist[kind])) + "\n"
    V.write(HUB["04"] + ".md", fm({"tags": ["центр"]}) + hub)

    # ---------------- карточки ----------------
    for e in list(cards.values()) + list(planned.values()):
        p = card_path[e["id"]]; src = block(p)
        a = areas.get(e["area"], {"title": e["area"]})
        if src == "09":
            up = ref(src, plan_group[e["area"]], a["title"], structural=True)
        elif src == "01":
            up = ref(src, theme_path[fly_theme[e["id"]]], fly_theme[e["id"]], structural=True)
        else:
            up = ref(src, area_path[e["area"]], a["title"], structural=True)
        tags = ["эксперимент", f"исход/{e['state']}", f"направление/{tag(e['area'])}"] + [f"метка/{tag(t)}" for t in e.get("tags", [])] + (["муха"] if src == "01" else [])
        props = {"aliases": [e["title"]], "tags": tags, "исход": STATE_RU.get(e["state"], e["state"]), "направление": a["title"],
                 "начат": e.get("started") or None, "завершён": e.get("finished") or None}
        body = f"# {STATE_ICON.get(e['state'], '')} {e['title']}\n\n↑ {up} · **Исход:** {STATE_RU.get(e['state'], e['state'])} · `{e['id']}`\n\n"
        if e.get("summary"):
            body += f"{L(e['summary'], src)}\n\n"
        for f, lbl in FIELDS:
            if e.get(f):
                body += f"## {lbl}\n{L(e[f], src)}\n\n"
        cond = [(k, e.get(k)) for k in ("baseline", "variant", "dataset", "split", "detector", "tracker", "seeds", "checkpoint") if e.get(k)]
        if cond:
            body += "## Условия\n" + "".join(f"- **{k}:** {L(str(v), src)}\n" for k, v in cond) + "\n"
        if e.get("acceptance"):
            body += "## Критерии приёмки (записаны до прогона)\n"
            for c in e["acceptance"]:
                mark = {True: "✅", False: "❌"}.get(c.get("met"), "⏳")
                body += f"- {mark} {L(c.get('criterion', ''), src)}" + (f"  \n  *{L(c['evidence'], src)}*" if c.get("evidence") else "") + "\n"
            body += "\n"
        if e.get("metrics"):
            body += "## Метрики\n| метрика | значение | baseline | подмножество | источник |\n|---|---|---|---|---|\n"
            for m in e["metrics"]:
                k = m.get("key", ""); kn = term_note.get(k)
                body += f"| {ref(src, kn, k) if kn else k} | {m.get('value')} | {m.get('baseline')} | {m.get('subset', '')} | `{m.get('source', '')}` · {m.get('locator', '')} |\n"
            body += "\n"
        if e.get("evidence"):
            body += "## Доказательства\n"
            for ev in e["evidence"]:
                link = L(ev.get("locator", ""), src) if ev.get("kind") == "plan_section" else f"`{ev.get('ref', '')}` · {ev.get('locator', '')}"
                body += f"- {ev.get('claim', ev.get('kind', ''))}: {link}\n"
            body += "\n"
        for f, lbl in (("files", "Код"), ("runs", "Данные"), ("commits", "Коммиты")):
            if e.get(f):
                body += f"## {lbl}\n" + "".join(f"- `{x}`\n" for x in e[f]) + "\n"
        V.write(p + ".md", fm(props) + body)

    # ---------------- 02 Эксперименты: направления ----------------
    by_area = defaultdict(list)
    for e in cards.values():
        by_area[e["area"]].append(e)
    for aid, a in areas.items():
        es = sorted(by_area.get(aid, []), key=lambda x: x.get("started") or "")
        body = f"# {a['title']}\n\n↑ [[{HUB['02']}|{BLOCK_TITLE['02']}]]\n\n{a.get('description', '')}\n\n## Эксперименты ({len(es)})\n"
        body += "".join(f"- {STATE_ICON.get(x['state'], '')} {ref('02', card_path[x['id']], x['title'], structural=True)} — {STATE_RU.get(x['state'], x['state'])}, {x.get('started') or '—'}\n" for x in es)
        V.write(area_path[aid] + ".md", fm({"tags": ["направление"]}) + body)
    n_state = defaultdict(int)
    for e in cards.values():
        n_state[e["state"]] += 1
    hub = f"# {BLOCK_TITLE['02']}\n\n↑ [[00 Главная|Главная]]\n\nЗавершённые эксперименты по направлениям (запланированные и идущие — только в плане и каталоге). Эксперименты мухи живут в блоке {ref('02', HUB['01'], 'Муха')} (ссылки на них отмечены ↗).\n\n"
    hub += "## Направления\n" + "".join(f"- [[{area_path[aid]}|{a['title']}]] ({len(by_area.get(aid, []))})\n" for aid, a in sorted(areas.items(), key=lambda x: x[1].get("order", 0)))
    hub += "\n## По исходам\n" + "".join(f"- {STATE_ICON.get(s, '')} {STATE_RU.get(s, s)}: {n}\n" for s, n in sorted(n_state.items(), key=lambda x: -x[1]))
    V.write(HUB["02"] + ".md", fm({"tags": ["центр"]}) + hub)

    # ---------------- 09 Запланировано ----------------
    p_area = defaultdict(list)
    for e in planned.values():
        p_area[e["area"]].append(e)
    for aid, es in p_area.items():
        body = f"# {areas[aid]['title']} — планы\n\n↑ [[{HUB['09']}|{BLOCK_TITLE['09']}]]\n\n"
        body += "".join(f"- {STATE_ICON.get(x['state'], '')} {ref('09', card_path[x['id']], x['title'], structural=True)} — {STATE_RU.get(x['state'], x['state'])}\n" for x in es)
        V.write(plan_group[aid] + ".md", fm({"tags": ["запланировано"]}) + body)
    hub = f"# {BLOCK_TITLE['09']}\n\n↑ [[00 Главная|Главная]]\n\nИдеи и эксперименты, которые ещё не завершены: запланированные, готовые к запуску и идущие. После завершения карточка переезжает в блок экспериментов (или мухи).\n\n"
    hub += "".join(f"- [[{plan_group[aid]}|{areas[aid]['title']}]] ({len(es)})\n" for aid, es in sorted(p_area.items(), key=lambda x: areas[x[0]].get("order", 0)))
    # идеи из текста плана (7.6 — кандидаты К, 7.7 — проверки Д, 7.8 — этапы трекера на мухе), ещё не ставшие карточками
    S = {num: c for lv, num, t, c in secs}
    done_tags = {t for e in cards_all.values() for t in e.get("tags", [])}
    done = {"Д" + t.split("-D")[1][:1] for t in done_tags if t.startswith("fundamental-D")} | {"К" + t[1:] for t in done_tags if re.fullmatch(r"K\d+", t)}
    ideas = []
    for code, title, basis in re.findall(r"^\| (К\d+) \| (.+?) \| (.+?) \|", S.get("7.6", ""), re.M):
        ideas.append((code, re.sub(r"\*\*", "", title), basis, "7.6"))
    for code, title in re.findall(r"(Д\d) ([^;]+?)(?:;|\. Затем)", S.get("7.7", "")):
        ideas.append((code, title, "", "7.7"))
    for n, title, rest in re.findall(r"^(\d)\. \*([^*]+)\*(.*)$", S.get("7.8", ""), re.M):
        if n != "1":  # этап 1 — карточка fly_tracker_lifecycle_v1
            ideas.append((f"Т{n}", f"Трекер на мухе, этап {n}: {title}", rest.strip(" —"), "7.8"))
    ideas = [x for x in ideas if x[0] not in done]
    igroup = "09 Запланировано/Идеи из плана"
    for code, title, basis, sec in ideas:
        nm = f"09 Запланировано/Идеи из плана/{safe(code + ' ' + title, 90)}"
        body = f"# {code} — {title}\n\n↑ [[{igroup}|Идеи из плана]] · источник: {ref('09', sec_note.get(sec, HUB['05']), 'раздел ' + sec + ' плана')}\n\n"
        if basis:
            body += f"**Опора:** {L(basis, '09')}\n"
        V.write(nm + ".md", fm({"tags": ["запланировано", "идея"] + (["идея/муха"] if re.search(r"мух|FlyVis", title, re.I) else [])}) + body)
    if ideas:
        V.write(igroup + ".md", fm({"tags": ["запланировано"]}) + f"# Идеи из плана\n\n↑ [[{HUB['09']}|{BLOCK_TITLE['09']}]]\n\nЕщё не ставшие карточками идеи из разделов 7.6–7.8 плана; сделанные исключены автоматически по меткам карточек.\n\n" +
                "".join(f"- [[09 Запланировано/Идеи из плана/{safe(c + ' ' + t, 90)}|{c} — {lab(t)[:90]}]]\n" for c, t, b, sc in ideas))
        hub += f"\n- [[{igroup}|Идеи из плана]] ({len(ideas)})\n"
    V.write(HUB["09"] + ".md", fm({"tags": ["центр", "запланировано"]}) + hub)

    # ---------------- 05 План ----------------
    children, parent = defaultdict(list), None
    for level, num, title, content, name in plan_notes:
        if level == 2:
            parent = name
        elif parent:
            children[parent].append(name)
    up_of = {c: p for p, cs in children.items() for c in cs}
    for level, num, title, content, name in plan_notes:
        up = up_of.get(name, HUB["05"])
        body = f"# {title}\n\n↑ [[{up}|{up.split('/')[-1]}]]\n\n" + L(content.strip(), "05") + "\n"
        if children.get(name):
            body += "\n## Подразделы\n" + "".join(f"- [[{c}|{c.split('/')[-1]}]]\n" for c in children[name])
        V.write(name + ".md", fm({"tags": ["план"], "раздел": num or None}) + body)
    idx = f"# {BLOCK_TITLE['05']}\n\n↑ [[00 Главная|Главная]]\n\nИсточник — `dissertation/ЕДИНЫЙ_ПЛАН.md`. Разделы:\n\n"
    idx += "".join(f"- [[{n}|{t}]]\n" for lv, num, t, c, n in plan_notes if lv == 2) + f"- {ref('05', HUB['06'], 'Журнал изменений')}\n\n" + L(pre.strip(), "05") + "\n"
    V.write(HUB["05"] + ".md", fm({"tags": ["центр"]}) + idx)

    # ---------------- 06 Журнал ----------------
    jnames = []
    final_re = re.compile(r"(?<![\w/])(" + "|".join(map(re.escape, sorted(cards, key=len, reverse=True))) + r")(?!\w)")
    entries = [en for en in split_journal(journal_text) if final_re.search(en["topic"] + " " + en["body"]) or JOURNAL_RESULT.search(en["topic"])]
    for en in entries:
        nm = f"06 Журнал/{en['date']} {en['time'].replace(':', '-')} {safe(en['topic'], 70)}"
        fly = bool(re.search(r"мух|FlyVis|сетчат|коннектом", en["topic"] + en["body"][:400], re.I))
        day = f"06 Журнал/Дни/{en['date']}"
        V.write(nm + ".md", fm({"tags": ["журнал"] + (["журнал/муха"] if fly else []), "дата": en["date"], "время": en["time"]}) +
                f"# {en['date']} {en['time']} — {en['topic']}\n\n↑ [[{day}|{en['date']}]]\n\n{L(en['body'], '06')}\n")
        jnames.append((en, nm, fly))
    by_day = defaultdict(list)
    for en, nm, fly in jnames:
        by_day[en["date"]].append((en, nm, fly))
    for d, items in by_day.items():
        V.write(f"06 Журнал/Дни/{d}.md", fm({"tags": ["журнал"]}) + f"# {d}\n\n↑ [[{HUB['06']}|{BLOCK_TITLE['06']}]]\n\n" +
                "".join(f"- {'🪰 ' if fly else ''}[[{nm}|{en['time']} — {lab(en['topic'])}]]\n" for en, nm, fly in sorted(items, key=lambda x: x[0]['time'])))
    jidx = f"# {BLOCK_TITLE['06']}\n\n↑ [[00 Главная|Главная]]\n\nИтоги экспериментов и закрытия веток из журнала `dissertation/ЕДИНЫЙ_ПЛАН.md` (полная хроника — в самом плане), новые сверху. 🪰 — запись о мухе.\n\n"
    for d in sorted(by_day, reverse=True):
        jidx += f"## [[06 Журнал/Дни/{d}|{d}]]\n" + "".join(f"- {'🪰 ' if fly else ''}{en['time']} — {lab(en['topic'])}\n" for en, nm, fly in sorted(by_day[d], key=lambda x: x[0]['time'], reverse=True)) + "\n"
    V.write(HUB["06"] + ".md", fm({"tags": ["центр"]}) + jidx)

    # ---------------- 07 Исследования ----------------
    rlinks = []
    for rdir, rtitle in RESEARCH.items():
        rp = ROOT / rdir
        if not (rp / "outline.yaml").exists():
            continue
        outline = yaml.safe_load((rp / "outline.yaml").read_text(encoding="utf-8"))
        tnote = f"07 Исследования/{rtitle}/{rtitle}"
        items_md = []
        for it in outline.get("items", []):
            slug = re.sub(r"[^\w]", "", it["name"].replace(" ", "_"))
            res = rp / "results" / f"{slug}.json"
            nm = f"07 Исследования/{rtitle}/{safe((it.get('id', '') + ' ' + it['name']).strip())}"
            body = f"# {it.get('id', '')} {it['name']}\n\n↑ [[{tnote}|{rtitle}]]\n\n" + (f"> {it['focus']}\n\n" if it.get("focus") else "")
            if res.exists():
                d = json.loads(res.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if k not in ("id", "name", "block", "uncertain"):
                        body += f"## {FIELD_RU.get(k, k)}\n{L(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), '07')}\n\n"
                if d.get("uncertain"):
                    body += f"*Неуверенные поля: {', '.join(d['uncertain'])}*\n"
            else:
                body += "*Результата нет (тема не исследовалась).*\n"
            V.write(nm + ".md", fm({"tags": ["исследование"]}) + body)
            items_md.append((it, nm, res.exists()))
        head = f"# {rtitle}\n\n↑ [[{HUB['07']}|{BLOCK_TITLE['07']}]]\n\nТема: {outline.get('topic', '')}\n\n"
        if (rp / "СВОДКА.md").exists():
            head += f"## Сводка\n{L((rp / 'СВОДКА.md').read_text(encoding='utf-8'), '07')}\n\n"
        head += "## Темы\n" + "".join(f"- {'✅' if ok else '·'} [[{nm}|{lab(it.get('id', '') + ' ' + it['name'])}]]\n" for it, nm, ok in items_md)
        V.write(tnote + ".md", fm({"tags": ["исследование"]}) + head)
        rlinks.append((rtitle, tnote))
    V.write(HUB["07"] + ".md", fm({"tags": ["центр"]}) + f"# {BLOCK_TITLE['07']}\n\n↑ [[00 Главная|Главная]]\n\n" + "".join(f"- [[{n}|{t}]]\n" for t, n in rlinks))

    # ---------------- 08 Документы ----------------
    docs = [(p, "08 Документы/Диссертация") for p in sorted((ROOT / "dissertation").glob("*.md")) if p.name != "ЕДИНЫЙ_ПЛАН.md"]
    docs += [(p, "08 Документы/Черновики глав") for p in sorted((ROOT / "dissertation/drafts").glob("*.md"))]
    docs += [(p, "08 Документы/Архив 2026-09-07") for p in sorted((ROOT / "dissertation/backup_20260907").rglob("*.md"))]
    docs += [(ROOT / n, "08 Документы/Корень проекта") for n in ("FLYVIS_TRACKER_RESEARCH_ROADMAP.md", "FLYVIS_TRACKER_GATE_REPORT.md", "FLYVIS_STRESS_AND_SECOND_ASSOCIATION_REPORT.md",
                                                             "FLYVIS_VIDEO_RATE_EXPERIMENT_REPORT.md", "TRACKER_1_0.md", "09092026.md", "СВОДНЫЙ_ПЛАН1.md", "ПЛАН_DRB_v3.md", "README.md", "AGENTS.md") if (ROOT / n).exists()]
    folders = list(dict.fromkeys(f for _, f in docs))
    by_folder = defaultdict(list)
    for p, folder in docs:
        nm = f"{folder}/{safe(p.stem)}"
        txt = p.read_text(encoding="utf-8", errors="replace")
        fly = bool(re.search(r"мух|FlyVis", p.stem + txt[:2000], re.I))
        V.write(nm + ".md", fm({"tags": ["документ"] + (["документ/муха"] if fly else []), "источник": str(p.relative_to(ROOT)).replace("\\", "/")}) +
                f"↑ [[{folder}/{folder.split('/')[-1]}|{folder.split('/')[-1]}]]\n\n" + L(txt, "08"))
        by_folder[folder].append((nm, fly))
    for folder in folders:
        V.write(f"{folder}/{folder.split('/')[-1]}.md", fm({"tags": ["документ"]}) + f"# {folder.split('/')[-1]}\n\n↑ [[{HUB['08']}|{BLOCK_TITLE['08']}]]\n\n" +
                "".join(f"- {'🪰 ' if fly else ''}[[{nm}|{nm.split('/')[-1]}]]\n" for nm, fly in by_folder[folder]))
    pubs = sorted((ROOT / "dissertation/публикации").glob("*.pdf"))
    for p in pubs:
        V.copy(p, f"08 Документы/Публикации/{p.name}")
    didx = f"# {BLOCK_TITLE['08']}\n\n↑ [[00 Главная|Главная]]\n\n" + "".join(f"- [[{f}/{f.split('/')[-1]}|{f.split('/')[-1]}]] ({len(by_folder[f])})\n" for f in folders)
    if pubs:
        didx += "\n## Публикации\n" + "".join(f"- [[08 Документы/Публикации/{p.name}|{p.name}]]\n" for p in pubs)
        didx += "".join(f"- `{p.name}` (Word, лежит в `dissertation/публикации/`)\n" for p in sorted((ROOT / "dissertation/публикации").glob("*.docx")))
    V.write(HUB["08"] + ".md", fm({"tags": ["центр"]}) + didx)

    # ---------------- 01 Муха ----------------
    build_fly(V, cards, card_path, fly_theme, theme_path, jnames, L, cards_all)

    # ---------------- 00 Главная ----------------
    home = "# Диссертация: трекинг и муха (FlyVis)\n\nХранилище собрано скриптом `scripts/build_obsidian_vault.py`; пересобирается без потери ваших заметок.\n\n"
    home += "## Блоки\n" + "".join(f"- [[{HUB[b]}|{BLOCK_TITLE[b]}]]\n" for b in ("01", "02", "09", "05", "06", "04", "07", "08"))
    home += f"\n- Карта мухи (холст): [[01 Муха/Карта мухи.canvas|открыть]]\n"
    home += "\n## Как читать граф\nКаждый блок — свой кластер; рёбра — только структура (центр блока → его заметки). Ссылки между блоками помечены ↗: они открывают заметку, но в графе не рисуются.\n\nЦвета: оранжевый — муха; красный — запланированное и идущее (отдельный кластер); зелёный / малиновый / жёлтый — исход завершённого эксперимента (положительный / отрицательный / неопределённый); синий — план; серый — журнал; фиолетовый — термины; бирюзовый — исследования; коричневый — документы.\n"
    V.write("00 Главная.md", fm({"tags": ["главная"]}) + home)

    graph_settings(args.vault)
    stale = [r for r in old if r not in set(V.written)]
    for r in stale:
        p = args.vault / r
        if p.exists() and p.is_file():
            p.unlink()
    for d in sorted({(args.vault / r).parent for r in stale}, key=lambda x: -len(str(x))):
        try:
            if d != args.vault and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    manifest.write_text(json.dumps(sorted(set(V.written)), ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"[OK] файлов: {len(set(V.written))}, удалено устаревших: {len(stale)} → {args.vault}")
    return 0


def build_fly(V, cards, card_path, fly_theme, theme_path, jnames, L, cards_all) -> None:
    def c(cid, label=None):
        if cid in card_path:
            return ref("01", card_path[cid], label or cards_all[cid]["title"])
        e = cards_all.get(cid)
        return f"«{label or (e['title'] if e else cid)}» ({STATE_RU.get(e['state'], '') if e else 'нет в каталоге'}; в граф попадёт после завершения)"
    H01 = HUB["01"]
    concepts = {
        "Коннектом и 734 параметра": f"""Схема связей (коннектом) фиксирована: какие типы нейронов с какими соединены, знак и число синапсов. Обучаются 734 числа: 65 постоянных времени, 65 потенциалов покоя, 604 силы связей «тип → тип». Поэтому любое дообучение остаётся «мушиным» по устройству, но может потерять «мушиность» по работе — её проверяют сравнением с биологией (`dsi_correlation_to_known`, `fri_correlation_to_known` в flyvis).

Значимость схемы — [[01 Муха/Нулевая модель коннектома|нулевая модель]]; устойчивость выводов — [[01 Муха/Ансамбль FlyVis|ансамбль]].""",
        "Сетчатка-ламина": f"""Передняя ступень глаза, которой нет в FlyVis: фоторецептор подстраивается под освещённость (закон Вебера), ламина выделяет локальный контраст. Без неё муха ночью проваливалась.

- Сетчатка устраняет ночной провал мухи: {c('flyvis_retina_stageB_v1')}.
- Выигрыш специфичен для мухи — оптическому потоку DIS та же сетчатка не помогает: {c('dis_retina_control_v1')}.
- Перед замороженным детектором YOLO кадры после сетчатки вредят (сдвиг входа): {c('fly_eye_frontend_v1')}.
- Первая разведка: {c('flyvis_retina_frontend_v1')}.""",
        "Нейроны движения T4-T5": f"""T4 и T5 — нейроны мухи, чувствительные к направлению движения (светлые и тёмные края). На них держится оценка движения мухой.

- Движение мухи на наших видео много хуже DIS: {c('flyvis_stress')}, {c('flyvis_fovea_motion_v1')}, {c('motion_reliability_v1')}.
- Линейный декодер на активности мухи хуже штатного, но муха несёт больше движения, чем её вход: {c('fly_decoder_refit_v1')}.
- Собственная уверенность мухи в движении антиинформативна: {c('motion_confidence_audit')}.""",
        "Глаз на трек": f"""На каждый трек сажается свой фовеальный глаз мухи (721 фасетка) — EyeBank. Глаз смотрит на окрестность рамки на всех кадрах 30 Гц, его внутреннее состояние «живёт» вместе с треком — по устройству это уже «один глаз — один трек», основа для [[01 Муха/Трекер на мухе|трекера на мухе]].

- Считывание глаза для событий трека: {c('flyvis_foveation_readout_v1')}, {c('flyvis_event_features_v1')}.""",
        "Отпечаток мухи": f"""Отпечаток (сигнатура) — сводка активности 34 выходных типов глаза мухи за начало и конец жизни трека; по нему решается «тот же объект или другой».

- Отпечаток несёт идентичность: к геометрии +0,14 AP — {c('fly_signature_reid_v1')}.
- Почти вся информация уже во входе глаза: поверх входа муха добавляет +0,006 (ночью +0,010) — {c('fly_matched_controls_v1')}.
- Сшивка треков по одной геометрии даёт +0,79 HOTA, внешность и муха сверх неё незначимы — {c('stitch_appearance_v1')}.""",
        "Нулевая модель коннектома": f"""Проверка «важна ли схема связей»: у той же мухи связи перемешивают с сохранением степеней и знаков.

- {c('fly_connectome_null_v1')}: перемешанная муха хуже настоящей (+0,044 AP у мухи одной), но вклад поверх входа на грани.
- **Поправка 03.10:** перемешивание делалось без переобучения — это проверка расстройки, а не топологии; при переобучении перемешанной сети разница исчезает (Dhiman 2026). Заявлять «топология значима» нельзя (исследование фундаментальных проблем, F17).""",
        "Ансамбль FlyVis": f"""Авторы FlyVis обучили 50 моделей с одной схемой связей; модели приходят к разным решениям при почти равной ошибке. Все наши выводы — на одной модели `flow/0000/000`.

- Проверка на моделях 012, 025, 037: {c('fly_ensemble_members_v1')}.""",
        "Линия A — трекер на мухе": f"""Собственный трекер, где движение и связывание шли через муху: HOTA 29,1 против 52,4 у ByteTrack, переключений ID 3 384 против 573 (val7); причина — ошибка движения мухи.

- {c('flyvis_standalone')}, {c('flyvis_stress')}.""",
        "Линия B — муха как помощник": f"""Трекер XYWHByteTrack со своими модулями, муха — источник признаков. Итоговый трекер: HOTA 58,47 → 59,27, но DetA не изменилась — потолок в обнаружении. Модули линии B — в блоке {ref('01', HUB['02'], 'Эксперименты')}.""",
        "Трекер на мухе": f"""Решение автора 04.10: трекер, где все решения ByteTrack принимает муха (план 7.8). Этапы: жизнь трека → рождение → сопоставление по отпечатку → движение (после дообучения мухи) → сборка против ByteTrack.

- Первый этап: {c('fly_tracker_lifecycle_v1')}.
- Предшественник: [[01 Муха/Линия A — трекер на мухе|линия A]].
- Мишень: у трети реальных потерь на пороге объект уверенно виден в прошлых 5 кадрах ({ref('01', card_path['past_view_oracle_v1'], 'Д2')}); продолжение по Калману вредит ({ref('01', card_path['track_continuation_v1'], 'гейт продолжения')}).""",
        "Дообучение мухи": f"""Муху на нашей задаче ещё никто не обучал: 734 числа — с Sintel. Рецепт (исследование фундаментальных проблем, F18): сперва вход и декодер, затем постоянные времени и потенциалы покоя с притяжением к решению Sintel, все 734 — только при сохранении «мушиности»; минимум 3 модели ансамбля; контроль — обычная сеть того же размера.

- Почему линейного декодера мало: {c('fly_decoder_refit_v1')}.""",
    }
    for title, text in concepts.items():
        V.write(f"01 Муха/Понятия/{title}.md", fm({"tags": ["муха", "понятие"]}) + f"# {title}\n\n↑ [[{H01}|Муха]]\n\n{text}\n")
    # ссылки понятий между собой написаны как [[01 Муха/<имя>]] — привести к папке «Понятия»
    for title in concepts:
        p = V.path / f"01 Муха/Понятия/{title}.md"
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"\[\[01 Муха/([^|\]]+)\|([^\]]+)\]\]",
                   lambda m: ref("01", f"01 Муха/Понятия/{m.group(1)}" if m.group(1) in concepts else f"01 Муха/{m.group(1)}", m.group(2))
                   if m.group(1) != "Муха (FlyVis)" else m.group(0), t)
        p.write_text(t, encoding="utf-8")

    themes = [t for t, _ in FLY_THEMES] + [FLY_OTHER]
    for theme in themes:
        ids = sorted([i for i, t in fly_theme.items() if t == theme], key=lambda i: cards[i].get("started") or "")
        if not ids:
            continue
        V.write(theme_path[theme] + ".md", fm({"tags": ["муха", "тема"]}) + f"# {theme}\n\n↑ [[{H01}|Муха]]\n\n" +
                "".join(f"- {STATE_ICON.get(cards[i]['state'], '')} [[{card_path[i]}|{lab(cards[i]['title'])}]] — {STATE_RU.get(cards[i]['state'], '')}, {cards[i].get('started') or '—'}\n" for i in ids))

    hub = f"""# 🪰 Муха (FlyVis)

↑ [[00 Главная|Главная]]

Муха (FlyVis) — модель зрения плодовой мушки дрозофилы, построенная по реальной схеме связей её нервной системы (коннектому): 45 669 нейронов 65 типов и 1,5 млн связей (Lappalainen et al., Nature 2024). Схему связей взяли из анатомии, а настраиваемыми оставили только 734 числа — скорости нейронов, их уровни покоя и силу связей между типами. Авторы настроили эти числа на одну задачу — оценивать движение в мультфильмах Sintel.

**Место в нашем конвейере.** YOLO26 находит объекты → трекер ByteTrack связывает рамки по геометрии → на каждый трек сажается свой глаз мухи, он смотрит на кусочек видео вокруг объекта и выдаёт отпечаток и оценку движения. Муха — вторая нейросеть, но с готовой «анатомией» из биологии.

**Главный вопрос диссертации:** даёт ли схема связей живой мухи то, чего не даёт обычная сеть того же размера на тех же данных.

## Понятия
""" + "".join(f"- [[01 Муха/Понятия/{t}|{t}]]\n" for t in concepts)
    hub += "\n## Темы экспериментов\n" + "".join(f"- [[{theme_path[t]}|{t}]] ({sum(1 for v in fly_theme.values() if v == t)})\n" for t in themes if any(v == t for v in fly_theme.values()))
    hub += f"\n## Сводки\n- [[01 Муха/Что можно и нельзя заявлять о мухе|Что можно и нельзя заявлять]]\n- [[01 Муха/Хронология мухи|Хронология]]\n- [[01 Муха/Карта мухи.canvas|Карта мухи (холст)]]\n"
    hub += "\n## Где муха в плане ↗\n" + "".join(f"- {ref('01', n, n.split('/')[-1])}\n" for n in ("05 План/7.4. FlyVis и раздельная уверенность, начато 13.09", "05 План/7.5. Детектор с мухой, начато 01.10",
                                                                                                      "05 План/7.7. Фундаментальные проблемы и порядок борьбы, 03.10", "05 План/7.8. Трекер, где решения принимает муха, начато 04.10"))
    V.write(H01 + ".md", fm({"tags": ["муха", "центр"], "aliases": ["FlyVis", "Муха"]}) + hub)

    fly_ids = sorted(fly_theme, key=lambda i: cards[i].get("started") or "")
    claims = f"# Что можно и нельзя заявлять о мухе\n\n↑ [[{H01}|Муха]]\n\nСобрано из полей «Решение» и «Нельзя заявлять» карточек мухи.\n\n"
    for cid in fly_ids:
        e = cards[cid]
        if e.get("decision") or e.get("cannot_claim"):
            claims += f"## {STATE_ICON.get(e['state'], '')} {e['title']}\n→ {c(cid, cid)}\n\n"
            if e.get("decision"):
                claims += f"**Решение.** {L(e['decision'], '01')}\n\n"
            if e.get("cannot_claim"):
                claims += f"**Нельзя заявлять.** {L(e['cannot_claim'], '01')}\n\n"
    V.write("01 Муха/Что можно и нельзя заявлять о мухе.md", fm({"tags": ["муха"]}) + claims)

    tl = f"# Хронология мухи\n\n↑ [[{H01}|Муха]]\n\n## Эксперименты\n| начат | завершён | исход | эксперимент |\n|---|---|---|---|\n"
    tl += "".join(f"| {cards[i].get('started') or '—'} | {cards[i].get('finished') or '—'} | {STATE_ICON.get(cards[i]['state'], '')} {STATE_RU.get(cards[i]['state'], '')} | {c(i)} |\n" for i in fly_ids)
    tl += "\n## Записи журнала о мухе ↗\n" + "".join(f"- {en['date']} {en['time']} — {ref('01', nm, en['topic'])}\n" for en, nm, fly in sorted(jnames, key=lambda x: (x[0]['date'], x[0]['time'])) if fly)
    V.write("01 Муха/Хронология мухи.md", fm({"tags": ["муха"]}) + tl)

    color = {"completed_positive": "4", "completed_negative": "1", "inconclusive": "3", "running": "5", "planned": "6"}
    nodes, edges = [], []
    pipe = [("p1", "Видео 30 Гц"), ("p2", "YOLO26\nдетектор"), ("p3", "ByteTrack\nтрекер"), ("p4", "Сетчатка-ламина"), ("p5", "Глаз мухи на трек\n(коннектом, 734 параметра)"), ("p6", "Отпечаток · движение\n→ решения трекера")]
    for k, (nid, text) in enumerate(pipe):
        nodes.append({"id": nid, "type": "text", "text": f"**{text}**", "x": k * 300, "y": -400, "width": 260, "height": 90, "color": "2" if nid in ("p4", "p5", "p6") else "0"})
    for a, b in (("p1", "p2"), ("p2", "p3"), ("p1", "p4"), ("p4", "p5"), ("p5", "p6"), ("p6", "p3")):
        edges.append({"id": f"e{a}{b}", "fromNode": a, "fromSide": "right" if a != "p6" else "top", "toNode": b, "toSide": "left" if b != "p3" else "top"})
    x = 0
    for g, theme in enumerate(themes):
        ids = [i for i in fly_ids if fly_theme[i] == theme]
        if not ids:
            continue
        nodes.append({"id": f"g{g}", "type": "group", "label": theme, "x": x, "y": -200, "width": 340, "height": 120 + 110 * len(ids)})
        for k, i in enumerate(ids):
            nodes.append({"id": f"n{g}_{k}", "type": "file", "file": f"{card_path[i]}.md", "x": x + 20, "y": -150 + 110 * k, "width": 300, "height": 90, **({"color": color[cards[i]['state']]} if cards[i]['state'] in color else {})})
        x += 380
    nodes.append({"id": "legend", "type": "text", "text": "Цвет карточки: зелёный — положительный, красный — отрицательный, жёлтый — неопределённый, голубой — идёт, фиолетовый — запланирован.", "x": 0, "y": -560, "width": 900, "height": 70})
    V.write("01 Муха/Карта мухи.canvas", json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=1))


def graph_settings(vault: Path) -> None:
    p = vault / ".obsidian" / "graph.json"
    g = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    rgb = lambda h: int(h.lstrip("#"), 16)
    g["colorGroups"] = [
        {"query": 'path:"09 Запланировано"', "color": {"a": 1, "rgb": rgb("#ff2d2d")}},
        {"query": 'path:"01 Муха"', "color": {"a": 1, "rgb": rgb("#f28c28")}},
        {"query": "tag:#исход/completed_positive", "color": {"a": 1, "rgb": rgb("#3fb950")}},
        {"query": "tag:#исход/completed_negative", "color": {"a": 1, "rgb": rgb("#db61a2")}},
        {"query": "tag:#исход/inconclusive", "color": {"a": 1, "rgb": rgb("#d4a72c")}},
        {"query": 'path:"02 Эксперименты"', "color": {"a": 1, "rgb": rgb("#7ee787")}},
        {"query": 'path:"05 План"', "color": {"a": 1, "rgb": rgb("#4c8ef7")}},
        {"query": 'path:"06 Журнал"', "color": {"a": 1, "rgb": rgb("#8b949e")}},
        {"query": 'path:"04 Термины и метрики"', "color": {"a": 1, "rgb": rgb("#a371f7")}},
        {"query": 'path:"07 Исследования"', "color": {"a": 1, "rgb": rgb("#39c5bb")}},
        {"query": 'path:"08 Документы"', "color": {"a": 1, "rgb": rgb("#b08b5b")}},
    ]
    g.update({"repelStrength": 18, "linkStrength": 1, "linkDistance": 60, "centerStrength": 0.25, "showArrow": False, "showTags": False})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
