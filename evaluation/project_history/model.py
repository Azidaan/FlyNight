"""Сущности истории проекта и допустимые состояния.

Состояний десять, и одно поле done для всего не годится: отрицательный
результат — это результат, а не ошибка, а закрытое направление — не то же,
что незавершённое.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

STATES: tuple[str, ...] = (
    'planned', 'ready', 'running', 'completed_positive', 'completed_negative',
    'inconclusive', 'closed', 'infrastructure', 'superseded', 'unknown')

STATE_LABELS: dict[str, str] = {
    'planned': 'запланирован',
    'ready': 'готов к запуску',
    'running': 'выполняется',
    'completed_positive': 'завершён · положительный результат',
    'completed_negative': 'завершён · отрицательный результат',
    'inconclusive': 'результат неоднозначен',
    'closed': 'направление закрыто',
    'infrastructure': 'инфраструктурное изменение',
    'superseded': 'заменён более поздней версией',
    'unknown': 'состояние не определено',
}

# Класс результата — то, по чему фильтрует пользователь (9.4).
RESULT_KINDS: tuple[str, ...] = (
    'positive', 'negative', 'inconclusive', 'infrastructure', 'none')
RESULT_LABELS: dict[str, str] = {
    'positive': 'положительный', 'negative': 'отрицательный',
    'inconclusive': 'неоднозначный', 'infrastructure': 'инфраструктурный',
    'none': 'результата пока нет',
}
RESULT_OF_STATE: dict[str, str] = {
    'planned': 'none', 'ready': 'none', 'running': 'none',
    'completed_positive': 'positive', 'completed_negative': 'negative',
    'inconclusive': 'inconclusive', 'closed': 'negative',
    'infrastructure': 'infrastructure', 'superseded': 'none', 'unknown': 'none',
}

EVIDENCE_KINDS: tuple[str, ...] = (
    'plan_section', 'commit', 'json', 'csv', 'markdown', 'runs_dir', 'figure',
    'file')
EVIDENCE_LEVELS: tuple[str, ...] = ('confirmed', 'derived', 'unconfirmed')

EVENT_KINDS: tuple[str, ...] = (
    'plan_changelog', 'plan_chronology', 'plan_closed', 'commit', 'manual',
    'uncommitted', 'unknown')
EVENT_KIND_LABELS: dict[str, str] = {
    'plan_changelog': 'журнал плана', 'plan_chronology': 'хронология плана',
    'plan_closed': 'закрытие по плану', 'commit': 'коммит',
    'manual': 'запись каталога', 'uncommitted': 'незакоммиченное состояние',
    'unknown': 'не классифицировано',
}

DATE_PRECISIONS: tuple[str, ...] = ('exact', 'day', 'range', 'month', 'derived')


def check_state(state: str) -> str:
    if state not in STATES:
        raise ValueError(f'недопустимое состояние {state!r}; '
                         f'допустимы: {", ".join(STATES)}')
    return state


@dataclass
class EvidenceReference:
    kind: str                      # EVIDENCE_KINDS
    ref: str                       # локальный путь (относительный) или hash
    locator: str = ''              # строка, раздел, ключ JSON
    claim: str = ''                # что подтверждает
    exists: bool | None = None     # существует ли источник сейчас
    sha256: str = ''               # если уже посчитана
    level: str = 'unconfirmed'     # EVIDENCE_LEVELS
    note: str = ''                 # расхождение, предупреждение

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricValue:
    key: str                       # ключ словаря метрик
    value: float | str | None = None
    baseline: float | str | None = None
    subset: str = ''
    source: str = ''               # путь к артефакту
    locator: str = ''
    note: str = ''
    # заполняется индексатором из словаря
    title: str = ''
    better: str = ''               # higher | lower | ''
    described: bool = False
    delta_pct: float | None = None
    source_exists: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectArea:
    id: str
    title: str
    description: str = ''
    tags: list[str] = field(default_factory=list)
    parent_id: str = ''
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    id: str
    title: str
    area: str
    state: str = 'unknown'
    tags: list[str] = field(default_factory=list)
    summary: str = ''
    question: str = ''
    hypothesis: str = ''
    expected: str = ''
    done: str = ''
    obtained: str = ''
    interpretation: str = ''
    decision: str = ''
    started: str = ''
    finished: str = ''
    detector: str = ''
    tracker: str = ''
    dataset: str = ''
    split: str = ''
    seeds: list[int] = field(default_factory=list)
    checkpoint: str = ''
    metrics: list[MetricValue] = field(default_factory=list)
    baseline: str = ''
    variant: str = ''
    commits: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    acceptance: list[dict] = field(default_factory=list)
    limitations: str = ''
    cannot_claim: str = ''
    next_step: str = ''
    evidence: list[EvidenceReference] = field(default_factory=list)
    confirmed: bool = False
    conditions: list[str] = field(default_factory=list)
    # производные (индексатор)
    result: str = 'none'
    has_commit: bool = False
    has_metrics: bool = False
    has_artifacts: bool = False
    source_kinds: list[str] = field(default_factory=list)
    missing_commits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['state_label'] = STATE_LABELS.get(self.state, self.state)
        d['result_label'] = RESULT_LABELS.get(self.result, self.result)
        return d


@dataclass
class CommitFile:
    path: str
    status: str                    # A M D R C T …
    added: int | None = None       # None — бинарный файл
    deleted: int | None = None
    exists: bool = False           # есть ли файл в рабочей директории сейчас

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommitRecord:
    hash: str
    short: str
    date: str                      # ISO с зоной
    author: str
    subject: str
    body: str = ''
    parents: list[str] = field(default_factory=list)
    branch: str = ''
    files: list[CommitFile] = field(default_factory=list)
    files_loaded: bool = False
    n_files: int | None = None
    added: int | None = None
    deleted: int | None = None
    experiments: list[str] = field(default_factory=list)
    plan_events: list[str] = field(default_factory=list)
    result: str = ''               # краткий результат из каталога/плана
    result_source: str = ''
    in_history: bool = True
    is_empty: bool = False
    is_merge: bool = False
    source: str = 'git'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimelineEvent:
    id: str
    when: str                      # ISO дата или дата-время
    when_end: str = ''             # для диапазонов
    date_precision: str = 'day'
    title: str = ''
    text: str = ''
    kind: str = 'unknown'          # EVENT_KINDS
    area: str = ''
    state: str = 'unknown'
    experiment_id: str = ''
    commits: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    plan_lines: list[int] = field(default_factory=list)   # [from, to]
    source: str = ''
    raw: str = ''                  # исходный фрагмент, если классификация не уверенна
    has_commit: bool = False
    has_artifact: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['state_label'] = STATE_LABELS.get(self.state, self.state)
        d['kind_label'] = EVENT_KIND_LABELS.get(self.kind, self.kind)
        return d
