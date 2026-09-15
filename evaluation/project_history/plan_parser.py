"""Разбор ЕДИНЫЙ_ПЛАН.md: заголовки, таблицы, хронология, журнал изменений.

Свободный научный текст целиком не «понимается» — вынимается только то,
что имеет надёжную форму:
  * заголовки `#..####` с номерами строк — оглавление и диапазоны разделов;
  * таблицы `| a | b |` — с заголовком раздела, в котором стоят;
  * хронология (раздел «Хронология»): подразделы вида `### 2.2. 18–25.08: …`
    и маркеры `- **19.08.** …` — даты без года получают год из контекста;
  * журнал изменений: блоки `**13.09.2026, 19:19 MSK** — …` и подзаписи
    `*02:17, тема.* …` / `*14.09, 21:35, тема.* …` / `*Тема.* …`.
Всё остальное остаётся текстом раздела. Если дату разобрать нельзя, запись
сохраняется с precision='derived' и исходным фрагментом.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

RE_HEADING = re.compile(r'^(#{1,4})\s+(.+?)\s*$')
RE_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
RE_TABLE_SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$')
RE_HASH = re.compile(r'`([0-9a-f]{7,40})`')
RE_YEAR = re.compile(r'\b(20\d\d)\b')
# **13.09.2026, 19:19 MSK** — тема   |   **08.09.2026** — тема
RE_CHANGELOG = re.compile(
    r'^\*\*(\d{1,2})\.(\d{1,2})\.(\d{4})(?:,\s*(\d{1,2}):(\d{2})\s*(?:MSK|МСК)?)?\*\*'
    r'\s*[—–-]\s*(.*)$')
# *02:17, тема.* текст  |  *14.09, 21:35, тема.* текст  |  *Тема.* текст
RE_SUBENTRY = re.compile(r'^\*([^*]+?)\.\*\s*(.*)$')
RE_SUB_TIME = re.compile(r'^(\d{1,2}):(\d{2})$')
RE_SUB_DATE = re.compile(r'^(\d{1,2})\.(\d{1,2})$')
# - **19.08.** текст
RE_BULLET_DATE = re.compile(r'^\s*[-*]\s+\*\*(\d{1,2})\.(\d{1,2})\.?\*\*\s*(.*)$')
# **09.09. Измерена цена…** текст   |   **10–11.09. Гейт пройден** текст
RE_PARA_DATE = re.compile(
    r'^\*\*(\d{1,2})(?:\s*[–-]\s*(\d{1,2}))?\.(\d{1,2})\.\s*(.*?)\*\*\s*(.*)$')
# 18–25.08: …  |  02–03.09: …  |  31.08: …  |  09–11.09: …
RE_NUMBERING = re.compile(r'^\d+(?:\.\d+)*\.?\s+')
RE_HEAD_RANGE = re.compile(
    r'^(\d{1,2})(?:\s*[–-]\s*(\d{1,2}))?\.(\d{1,2})\s*[:.]\s*(.*)$')
MONTHS = {'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4, 'ма': 5, 'июн': 6,
          'июл': 7, 'август': 8, 'сентябр': 9, 'октябр': 10, 'ноябр': 11,
          'декабр': 12}
RE_HEAD_MONTHS = re.compile(
    r'^([А-Яа-я]+)(?:\s*[—–-]\s*([А-Яа-я]+))?\s*:\s*(.*)$')
_DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass
class Heading:
    level: int
    title: str
    line: int
    line_to: int = 0

    @property
    def line_from(self) -> int:
        return self.line

    @property
    def anchor(self) -> str:
        return f'L{self.line}'


@dataclass
class Table:
    section: str
    line: int
    header: list[str]
    rows: list[list[str]]


@dataclass
class SubEntry:
    title: str
    text: str
    when: str                       # ISO дата или дата-время
    line_from: int
    line_to: int
    hashes: list[str] = field(default_factory=list)
    precision: str = 'day'


@dataclass
class ChangelogEntry:
    date: str                       # YYYY-MM-DD
    time: str                       # HH:MM или ''
    title: str
    line_from: int
    line_to: int
    text: str = ''
    entries: list[SubEntry] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)

    @property
    def when(self) -> str:
        return f'{self.date}T{self.time}' if self.time else self.date

    def all_hashes(self) -> list[str]:
        out = list(self.hashes)
        for s in self.entries:
            out += [h for h in s.hashes if h not in out]
        return out


@dataclass
class ChronoItem:
    title: str
    text: str
    when: str
    when_end: str
    precision: str
    line: int
    line_to: int
    section: str
    hashes: list[str] = field(default_factory=list)
    kind: str = 'section'           # section | bullet


@dataclass
class PlanDocument:
    text: str
    lines: list[str]
    headings: list[Heading]
    tables: list[Table]
    changelog: list[ChangelogEntry]
    chronology: list[ChronoItem]
    default_year: int

    def section_by_title(self, title: str) -> Heading | None:
        return next((h for h in self.headings if h.title == title), None)

    def section_text(self, heading: Heading) -> str:
        return '\n'.join(self.lines[heading.line - 1:heading.line_to])

    def heading_at(self, line: int) -> Heading | None:
        """Ближайший заголовок выше строки."""
        best = None
        for h in self.headings:
            if h.line <= line:
                best = h
            else:
                break
        return best


# ——— вспомогательное ———

def _iso(year: int, month: int, day: int) -> str:
    return f'{year:04d}-{month:02d}-{day:02d}'


def _valid(month: int, day: int) -> bool:
    return 1 <= month <= 12 and 1 <= day <= _DAYS_IN_MONTH[month - 1]


def _month_of(word: str) -> int | None:
    w = word.lower()
    for stem, num in MONTHS.items():
        if w.startswith(stem):
            return num
    return None


def _cut(text: str) -> str:
    return text.strip()


def parse_headings(lines: list[str]) -> list[Heading]:
    heads: list[Heading] = []
    in_code = False
    for i, raw in enumerate(lines, 1):
        if raw.strip().startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = RE_HEADING.match(raw)
        if m:
            heads.append(Heading(level=len(m.group(1)), title=_cut(m.group(2)), line=i))
    # диапазон раздела — до следующего заголовка того же или более высокого уровня
    n = len(lines)
    for k, h in enumerate(heads):
        end = n
        for nxt in heads[k + 1:]:
            if nxt.level <= h.level:
                end = nxt.line - 1
                break
        h.line_to = end
    return heads


def _split_row(row: str) -> list[str]:
    s = row.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    # экранированная черта `\|` внутри клетки не делит её
    parts = re.split(r'(?<!\\)\|', s)
    return [p.replace('\\|', '|').strip() for p in parts]


def parse_tables(lines: list[str], headings: list[Heading]) -> list[Table]:
    tables: list[Table] = []
    i, n = 0, len(lines)

    def section_at(line: int) -> str:
        sec = ''
        for h in headings:
            if h.line <= line:
                sec = h.title
            else:
                break
        return sec

    while i < n:
        if (RE_TABLE_ROW.match(lines[i]) and i + 1 < n
                and RE_TABLE_SEP.match(lines[i + 1])):
            header = _split_row(lines[i])
            start = i + 1
            j = i + 2
            rows: list[list[str]] = []
            while j < n and RE_TABLE_ROW.match(lines[j]):
                rows.append(_split_row(lines[j]))
                j += 1
            tables.append(Table(section=section_at(start), line=start,
                                header=header, rows=rows))
            i = j
        else:
            i += 1
    return tables


def parse_changelog(lines: list[str], headings: list[Heading]) -> list[ChangelogEntry]:
    head = next((h for h in headings if h.title.lower().startswith('журнал изменений')), None)
    if head is None:
        return []
    start, end = head.line, head.line_to           # 1-based включительно
    entries: list[ChangelogEntry] = []
    cur: ChangelogEntry | None = None
    sub: SubEntry | None = None
    para: list[str] = []
    para_from = 0

    def close_para(upto: int) -> None:
        nonlocal para, para_from, sub
        if not para or cur is None:
            para, para_from = [], 0
            return
        text = ' '.join(p.strip() for p in para).strip()
        hashes = [h for h in RE_HASH.findall(text)]
        m = RE_SUBENTRY.match(para[0].strip())
        if m:
            head_txt, rest = m.group(1).strip(), m.group(2)
            parts = [p.strip() for p in head_txt.split(',')]
            when, prec = cur.date, 'day'
            date_set = False
            title_parts: list[str] = []
            for p in parts:
                mt = RE_SUB_TIME.match(p)
                md = RE_SUB_DATE.match(p)
                if mt and not title_parts:
                    when = when.split('T')[0] + f'T{int(mt.group(1)):02d}:{mt.group(2)}'
                    prec = 'exact'
                elif md and not title_parts and not date_set:
                    d, mo = int(md.group(1)), int(md.group(2))
                    year = int(cur.date[:4])
                    if _valid(mo, d):
                        when = _iso(year, mo, d)
                        date_set = True
                else:
                    title_parts.append(p)
            title = ', '.join(title_parts)
            body = ' '.join([rest.strip()] + [p.strip() for p in para[1:]]).strip()
            s = SubEntry(title=title, text=body, when=when, line_from=para_from,
                         line_to=upto, hashes=[h for h in RE_HASH.findall(body)],
                         precision=prec)
            cur.entries.append(s)
        else:
            cur.text = (cur.text + '\n\n' + text).strip() if cur.text else text
            for h in hashes:
                if h not in cur.hashes:
                    cur.hashes.append(h)
        para, para_from = [], 0

    for i in range(start, end + 1):
        raw = lines[i - 1]
        line = raw.rstrip()
        m = RE_CHANGELOG.match(line.strip())
        if m:
            close_para(i - 1)
            if cur is not None:
                cur.line_to = i - 1
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            t = f'{int(m.group(4)):02d}:{m.group(5)}' if m.group(4) else ''
            cur = ChangelogEntry(date=_iso(y, mo, d) if _valid(mo, d) else f'{y}',
                                 time=t, title=_cut(m.group(6)), line_from=i,
                                 line_to=end)
            entries.append(cur)
            continue
        if not line.strip():
            close_para(i - 1)
            continue
        if not para:
            para_from = i
        para.append(line)
    close_para(end)
    # хвост блока — до конца раздела; последний уже стоит
    return entries


def parse_chronology(lines: list[str], headings: list[Heading],
                     default_year: int) -> list[ChronoItem]:
    head = next((h for h in headings
                 if re.match(r'^\d+\.\s*хронология', h.title.lower())), None)
    if head is None:
        return []
    items: list[ChronoItem] = []
    subs = [h for h in headings if head.line < h.line <= head.line_to and h.level > head.level]
    for h in subs:
        title_no_num = RE_NUMBERING.sub('', h.title)
        body = '\n'.join(lines[h.line:h.line_to]).strip()
        hashes = RE_HASH.findall(body)
        m = RE_HEAD_RANGE.match(title_no_num)
        when = when_end = ''
        prec = 'derived'
        if m:
            d1, d2, mo = int(m.group(1)), m.group(2), int(m.group(3))
            if _valid(mo, d1):
                when = _iso(default_year, mo, d1)
                if d2 and _valid(mo, int(d2)):
                    when_end, prec = _iso(default_year, mo, int(d2)), 'range'
                else:
                    prec = 'day'
        else:
            mm = RE_HEAD_MONTHS.match(title_no_num)
            if mm and _month_of(mm.group(1)):
                m1 = _month_of(mm.group(1))
                m2 = _month_of(mm.group(2)) if mm.group(2) else None
                when = _iso(default_year, m1, 1)
                last = m2 or m1
                when_end = _iso(default_year, last, _DAYS_IN_MONTH[last - 1])
                prec = 'month'
        items.append(ChronoItem(title=title_no_num, text=body, when=when,
                                when_end=when_end, precision=prec, line=h.line,
                                line_to=h.line_to, section=h.title,
                                hashes=hashes, kind='section'))
        # датированные маркеры внутри подраздела
        for i in range(h.line + 1, h.line_to + 1):
            raw = lines[i - 1]
            mb = RE_BULLET_DATE.match(raw)
            mp = RE_PARA_DATE.match(raw.strip()) if not mb else None
            if not mb and not mp:
                continue
            j = i + 1
            extra: list[str] = []
            if mb:
                d, d2, mo = int(mb.group(1)), None, int(mb.group(2))
                rest = mb.group(3)
                # продолжение маркера — строки с отступом до пустой/следующего маркера
                while (j <= h.line_to and lines[j - 1].startswith('  ')
                       and lines[j - 1].strip()):
                    extra.append(lines[j - 1].strip())
                    j += 1
                text = ' '.join([rest.strip()] + extra).strip()
                title = re.split(r'(?<=[.!?])\s', text, maxsplit=1)[0]
            else:
                d, d2 = int(mp.group(1)), (int(mp.group(2)) if mp.group(2) else None)
                mo, title, rest = int(mp.group(3)), mp.group(4).strip(), mp.group(5)
                # абзац — до пустой строки
                while j <= h.line_to and lines[j - 1].strip():
                    extra.append(lines[j - 1].strip())
                    j += 1
                text = ' '.join([rest.strip()] + extra).strip()
            ok = _valid(mo, d)
            items.append(ChronoItem(
                title=title, text=text,
                when=_iso(default_year, mo, d) if ok else '',
                when_end=_iso(default_year, mo, d2) if (ok and d2 and _valid(mo, d2)) else '',
                precision=('range' if d2 else 'day') if ok else 'derived',
                line=i, line_to=j - 1, section=h.title,
                hashes=RE_HASH.findall(text), kind='bullet'))
    return items


def detect_year(lines: list[str]) -> int:
    """Год по первой полной дате в тексте (например, «Составлен 07.09.2026»)."""
    for raw in lines[:60]:
        m = re.search(r'\d{1,2}\.\d{1,2}\.(20\d\d)', raw)
        if m:
            return int(m.group(1))
    for raw in lines:
        m = RE_YEAR.search(raw)
        if m:
            return int(m.group(1))
    return 2026


def parse_plan(text: str) -> PlanDocument:
    lines = text.splitlines()
    headings = parse_headings(lines)
    year = detect_year(lines)
    return PlanDocument(
        text=text, lines=lines, headings=headings,
        tables=parse_tables(lines, headings),
        changelog=parse_changelog(lines, headings),
        chronology=parse_chronology(lines, headings, year),
        default_year=year)
