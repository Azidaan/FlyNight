"""Парный бутстреп по видео для сквозных опытов eval_motion_source.py (карточка e2e_motion_source_v1).

Сравнения задаются парами «эталон:кандидат» (имена конфигураций, могут лежать в разных
отчётах). Для каждого сплита и метрики: среднее изменение по видео, 95 % ДИ перцентилей,
доля видео с улучшением. Разность метрик сплита целиком (TrackEval по всем видео) — отдельно.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_flyvis_qmatch_gate import paired_video_bootstrap  # noqa: E402
from scripts.run_bdd_video_rate_experiment import save_json  # noqa: E402

METRICS = ("HOTA", "IDF1", "AssA", "DetA", "IDSw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="эталон:кандидат")
    parser.add_argument("--splits", default="dev,test,val7")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    per_video, summary = {}, {}
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        per_video.update(report["per_video"]); summary.update(report["summary"])
    out = {"reports": [str(p) for p in args.reports], "comparisons": {}}
    for pair in args.pairs:
        ref, cand = pair.split(":")
        out["comparisons"][pair] = {}
        for split in args.splits.split(","):
            a = {"test_by_video": {v: m for v, m in sorted(per_video[ref][split].items())}}
            b = {"test_by_video": {v: per_video[cand][split][v] for v in a["test_by_video"]}}
            block = {}
            for metric in METRICS:
                boot = paired_video_bootstrap(a, b, metric)
                diffs = np.asarray(list(boot["per_video_differences"].values()))
                better = diffs < 0 if metric == "IDSw" else diffs > 0
                block[metric] = {"mean_per_video": boot["mean"] if "mean" in boot else float(diffs.mean()), "ci95": boot.get("ci95_percentile"),
                                 "videos_better": int(better.sum()), "videos_worse": int((diffs < 0 if metric != "IDSw" else diffs > 0).sum()), "videos": len(diffs),
                                 "split_total_delta": (summary[cand][split][metric] - summary[ref][split][metric]) if summary[cand][split].get(metric) is not None else None}
            out["comparisons"][pair][split] = block
            h = block["HOTA"]; i = block["IDF1"]
            print(f"[{pair:40s}] {split:5s} ΔHOTA сплит {h['split_total_delta']:+.3f}, по видео {h['mean_per_video']:+.3f} ДИ [{h['ci95'][0]:+.3f}; {h['ci95'][1]:+.3f}], лучше/хуже {h['videos_better']}/{h['videos_worse']} | ΔIDF1 сплит {i['split_total_delta']:+.3f} ДИ [{i['ci95'][0]:+.3f}; {i['ci95'][1]:+.3f}]", flush=True)
    save_json(args.output, out)
    print(f"[OK] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
