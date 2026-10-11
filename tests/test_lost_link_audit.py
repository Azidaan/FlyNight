"""Модульные тесты: слияние номеров аудита связей, буфер трекера сборщика сигнатур, восстановление прежних версий карточек."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lost_link_audit import merge_ids  # noqa: E402
from scripts.recover_revisions import add_revisions, collect  # noqa: E402


def _fr(*ids):
    return {"name": "f", "labels": [{"id": str(i), "category": "car", "box2d": {}} for i in ids]}


def _ids(rows):
    return [[l["id"] for l in fr["labels"]] for fr in rows]


def test_merge_simple_link():
    rows, conf = merge_ids([_fr(1), _fr(2)], [{"old": "1", "new": "2"}])
    assert _ids(rows) == [["1"], ["1"]] and conf == 0


def test_merge_chain_conflict_gets_small_unique_int():
    # 1→2 и 2→3 сливают всё в «1»; в кадре 2 и 3 вместе — конфликт, второй рамке — свой номер или следующий за наибольшим
    res = [_fr(1), _fr(2, 3), _fr(3)]
    rows, conf = merge_ids(res, [{"old": "1", "new": "2"}, {"old": "2", "new": "3"}])
    ids = _ids(rows)
    assert conf == 1
    for fr in ids:
        assert len(fr) == len(set(fr))                    # номера в кадре различны
        assert all(int(i) <= 4 for i in fr)               # целые и малые (TrackEval строит таблицу длиной в номер)


def test_merge_conflict_own_id_taken_uses_max_plus_one():
    # «5» и «7» сливаются в «5»; в кадре рамки 5 и 7 → у 7 свой номер «7» свободен; добавим рамку, корень которой тоже «5»
    res = [_fr(5, 7, 9)]
    rows, conf = merge_ids(res, [{"old": "5", "new": "7"}, {"old": "7", "new": "9"}])
    fr = _ids(rows)[0]
    assert len(fr) == len(set(fr)) == 3 and conf == 2
    assert max(int(i) for i in fr) <= 9 + 2


def test_merge_no_links_keeps_ids():
    res = [_fr(3, 4), _fr(4)]
    rows, conf = merge_ids(res, [])
    assert _ids(rows) == [["3", "4"], ["4"]] and conf == 0


def test_tracker_config_buffer():
    from scripts.build_fly_signatures import tracker_config
    from scripts.freeze_bench import XYWH_CONFIG
    assert tracker_config(None) == XYWH_CONFIG and tracker_config(None)["track_buffer"] == 30
    c = tracker_config(150)
    assert c["track_buffer"] == 150 and {k: v for k, v in c.items() if k != "track_buffer"} == {k: v for k, v in XYWH_CONFIG.items() if k != "track_buffer"}
    assert XYWH_CONFIG["track_buffer"] == 30                # стенд не изменён


def _cat(state, text):
    return {"experiments": [{"id": "e", "state": state, "interpretation": text}]}


def test_revisions_replaced_text_kept():
    seen = collect([("a1", "2026-10-01", _cat("planned", "черновик")), ("b2", "2026-10-02", _cat("completed_negative", "старый вывод")),
                    ("c3", "2026-10-03", _cat("completed_negative", "новый вывод"))])
    cur = _cat("completed_negative", "новый вывод")
    assert add_revisions(cur, seen) == 1
    r = cur["experiments"][0]["revisions"][0]
    assert r["old"] == "старый вывод" and r["commit"] == "c3" and r["field"] == "interpretation"   # черновик до итога не берётся
    assert add_revisions(cur, seen) == 0                                                          # без повторов


def test_revisions_append_not_counted():
    seen = collect([("a1", "2026-10-01", _cat("closed", "вывод")), ("b2", "2026-10-02", _cat("closed", "вывод. Дополнение."))])
    cur = _cat("closed", "вывод. Дополнение.")
    assert add_revisions(cur, seen) == 0 and "revisions" not in cur["experiments"][0]
