"""Безопасный HTML: экранирование и минимальный рендер плана.

Текст из Git, плана и JSON никогда не вставляется в страницу как разметка.
Здесь — единственный рендер Markdown → HTML для вкладки «Единый план»:
сначала весь текст экранируется, потом распознаются только заголовки,
таблицы, списки, `код` и **жирный**. Всё остальное остаётся текстом.
Каждый блок получает id="L<номер строки>" — для перехода из хронологии.
"""
from __future__ import annotations

import html
import re

from . import plan_parser

RE_CODE = re.compile(r'`([^`]+)`')
RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
RE_ITALIC = re.compile(r'(?<![*\w])\*([^*\n]+?)\*(?![*\w])')
RE_STRIKE = re.compile(r'~~(.+?)~~')
RE_BULLET = re.compile(r'^(\s*)[-*]\s+(.*)$')
RE_NUMBERED = re.compile(r'^(\s*)\d+\.\s+(.*)$')


def escape(text: str) -> str:
    return html.escape(str(text), quote=True)


def _inline(escaped: str) -> str:
    """Разметка внутри строки — по уже экранированному тексту."""
    s = RE_CODE.sub(r'<code>\1</code>', escaped)
    s = RE_BOLD.sub(r'<b>\1</b>', s)
    s = RE_STRIKE.sub(r'<s>\1</s>', s)
    s = RE_ITALIC.sub(r'<i>\1</i>', s)
    return s


def plan_html(doc: plan_parser.PlanDocument) -> str:
    lines = doc.lines
    n = len(lines)
    heads = {h.line: h for h in doc.headings}
    tables = {t.line: t for t in doc.tables}       # t.line — строка заголовка таблицы (1-based)
    out: list[str] = []
    i = 0
    para: list[str] = []
    para_from = 0
    in_code = False
    code: list[str] = []
    code_from = 0

    def flush_para() -> None:
        nonlocal para
        if para:
            out.append(f'<p id="L{para_from}" data-line="{para_from}">'
                       + _inline(escape(' '.join(p.strip() for p in para))) + '</p>')
            para = []

    while i < n:
        raw = lines[i]
        line_no = i + 1
        stripped = raw.strip()
        if stripped.startswith('```'):
            flush_para()
            if in_code:
                out.append(f'<pre id="L{code_from}" data-line="{code_from}">'
                           + escape('\n'.join(code)) + '</pre>')
                code, in_code = [], False
            else:
                in_code, code_from, code = True, line_no, []
            i += 1
            continue
        if in_code:
            code.append(raw)
            i += 1
            continue
        if line_no in heads:
            flush_para()
            h = heads[line_no]
            lvl = min(h.level, 4)
            out.append(f'<h{lvl} id="L{line_no}" data-line="{line_no}">'
                       + _inline(escape(h.title)) + f'</h{lvl}>')
            i += 1
            continue
        if line_no in tables:
            flush_para()
            t = tables[line_no]
            rows_html = ''.join(
                '<tr>' + ''.join(f'<td>{_inline(escape(c))}</td>' for c in r) + '</tr>'
                for r in t.rows)
            head_html = ''.join(f'<th>{_inline(escape(c))}</th>' for c in t.header)
            out.append(f'<div class="tblwrap" id="L{line_no}" data-line="{line_no}">'
                       f'<table><thead><tr>{head_html}</tr></thead>'
                       f'<tbody>{rows_html}</tbody></table></div>')
            i += 2 + len(t.rows)
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        if stripped == '---':
            flush_para()
            out.append(f'<hr id="L{line_no}" data-line="{line_no}">')
            i += 1
            continue
        mb = RE_BULLET.match(raw) or RE_NUMBERED.match(raw)
        if mb:
            flush_para()
            items: list[tuple[int, str]] = []
            while i < n:
                m2 = RE_BULLET.match(lines[i]) or RE_NUMBERED.match(lines[i])
                if not m2:
                    break
                start_line = i + 1
                text = m2.group(2)
                i += 1
                while i < n and lines[i].startswith('  ') and lines[i].strip() \
                        and not (RE_BULLET.match(lines[i]) or RE_NUMBERED.match(lines[i])):
                    text += ' ' + lines[i].strip()
                    i += 1
                items.append((start_line, text))
            out.append('<ul>' + ''.join(
                f'<li id="L{ln}" data-line="{ln}">{_inline(escape(t))}</li>'
                for ln, t in items) + '</ul>')
            continue
        if not para:
            para_from = line_no
        para.append(raw)
        i += 1
    flush_para()
    if in_code:
        out.append(f'<pre id="L{code_from}">' + escape('\n'.join(code)) + '</pre>')
    return '\n'.join(out)
