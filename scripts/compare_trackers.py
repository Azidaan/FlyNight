"""Сравнение ByteTrack и BoT-SORT на дефолтных параметрах (Этап 8, подэтап 8.1).

Почему live-прогон, а не кэш детекций: BoT-SORT принципиально не работает
на кэше — компенсация движения камеры (GMC) и опциональный ReID требуют
самих кадров. Поэтому ОБА трекера гоняются через model.track() (GPU,
одинаковый путь инференса — честное сравнение), результаты оцениваются
официальным bdd100k.eval per-поднабор.

Видео, входящее в несколько поднаборов (например night+rain), трекается
один раз: параметры дефолтные, результат от поднабора не зависит — оценка
идёт по подмножествам video_id. Прогон возобновляем: готовые <video>.json
и готовые строки CSV пропускаются.

Выход:
    experiments/results/tracker_comparison.csv
    notes/tracker_choice.md  (автогенерация: выбор по среднему HOTA)

Использование (после появления обученных весов):
    python scripts/compare_trackers.py --model weights/baseline_seed42/best.pt
    python scripts/compare_trackers.py --model ... --limit 20   # быстрая прикидка
    python scripts/compare_trackers.py --model x --dry-run      # только состав поднаборов
"""
import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import DATA_ROOT  # noqa: E402
from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)
from evaluation.tracking_metrics import (  # noqa: E402
    evaluate_tracking_bdd, run_tracker_on_videos,
)

RESULTS_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'tracker_comparison.csv'
COMPARE_RUNS = PROJECT_ROOT / 'runs' / 'track_compare'
NOTE_PATH = PROJECT_ROOT / 'notes' / 'tracker_choice.md'
SUBSETS = ('normal', 'night', 'rain', 'fog')
TRACKERS = ('bytetrack.yaml', 'botsort.yaml')
METRIC_KEYS = ('HOTA', 'IDF1', 'MOTA', 'IDSw', 'MOTP', 'DetA', 'AssA')
CSV_HEADER = ('tracker', 'subset', 'split', 'n_videos', *METRIC_KEYS)


def subset_videos(split: str, subsets: list[str]) -> dict[str, list[str]]:
    """video_id поднаборов из MOT split (джойн с Det-атрибутами)."""
    det_json = DATA_ROOT / 'labels' / f'bdd100k_labels_images_{split}.json'
    attrs = load_video_attributes(str(det_json))
    mot_ids = sorted(p.name for p in
                     (DATA_ROOT / 'images' / 'track' / split).iterdir()
                     if p.is_dir())
    return {s: filter_videos_by_subset(attrs, mot_ids, s) for s in subsets}


def load_done_rows() -> list[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_choice_note(rows: list[dict]) -> None:
    """notes/tracker_choice.md по всем строкам CSV. Перезаписывается целиком."""
    by_tracker: dict[str, list[dict]] = {}
    for r in rows:
        by_tracker.setdefault(r['tracker'], []).append(r)

    lines = [
        '# Выбор трекера (Этап 8, подэтап 8.1)',
        '',
        f'> Автосгенерировано `scripts/compare_trackers.py` {date.today()}.',
        '> Не редактировать вручную — перезаписывается при каждом запуске.',
        '',
        'Оба трекера — live через model.track() на дефолтных параметрах',
        'ultralytics (bytetrack.yaml / botsort.yaml, ReID у BoT-SORT выключен',
        'по умолчанию), оценка — официальный bdd100k.eval по val-поднаборам.',
        '',
        '| трекер | поднабор | видео | ' + ' | '.join(METRIC_KEYS) + ' |',
        '|---|---|---|' + '---|' * len(METRIC_KEYS),
    ]
    for r in rows:
        lines.append(
            f'| {r["tracker"]} | {r["subset"]} | {r["n_videos"]} | '
            + ' | '.join(str(r.get(k, '')) for k in METRIC_KEYS) + ' |')

    means = {}
    for name, trows in by_tracker.items():
        hota = [float(r['HOTA']) for r in trows if r.get('HOTA')]
        idf1 = [float(r['IDF1']) for r in trows if r.get('IDF1')]
        if hota:
            means[name] = (sum(hota) / len(hota),
                           sum(idf1) / len(idf1) if idf1 else float('nan'))
    lines += ['', 'Средние по поднаборам (невзвешенные):', '',
              '| трекер | HOTA | IDF1 |', '|---|---|---|']
    for name, (h, i) in sorted(means.items(), key=lambda kv: -kv[1][0]):
        lines.append(f'| {name} | {h:.2f} | {i:.2f} |')

    if len(means) >= 2:
        ranked = sorted(means.items(), key=lambda kv: -kv[1][0])
        (win, (wh, _)), (run, (rh, _)) = ranked[0], ranked[1]
        lines += ['', f'**Выбор: {win}** (средний HOTA {wh:.2f} против {rh:.2f} '
                      f'у {run}, разница {wh - rh:+.2f}).']
    else:
        lines += ['', '**Выбор не зафиксирован** — в CSV пока один трекер.']

    lines += [
        '',
        'Примечания:',
        '- Адаптация порогов по Q (Этап 9, `tracking/lqa_guided_tracker.py`)',
        '  и сеточный поиск Этапа 8.2 построены поверх ByteTrack; если',
        '  BoT-SORT выигрывает с заметным отрывом (>1 HOTA), решение о',
        '  переносе адаптации принимается отдельно.',
        '- BoT-SORT на кэше детекций невозможен (GMC/ReID требуют кадров),',
        '  поэтому сравнение выполнено live для обоих трекеров.',
        '',
    ]
    NOTE_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[INFO] заметка обновлена: {NOTE_PATH}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='чекпойнт детектора (best.pt)')
    ap.add_argument('--split', default='val')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS))
    ap.add_argument('--trackers', nargs='+', default=list(TRACKERS))
    ap.add_argument('--limit', type=int, default=None,
                    help='не более N видео на поднабор (быстрая прикидка)')
    ap.add_argument('--dry-run', action='store_true',
                    help='показать состав поднаборов и план, не запуская трекер')
    args = ap.parse_args()

    by_subset = subset_videos(args.split, args.subsets)
    if args.limit:
        by_subset = {s: v[:args.limit] for s, v in by_subset.items()}
    all_ids = sorted({v for vids in by_subset.values() for v in vids})
    for s, vids in by_subset.items():
        print(f'[INFO] поднабор {s}: {len(vids)} видео')
    print(f'[INFO] уникальных видео: {len(all_ids)}, '
          f'трекеры: {", ".join(args.trackers)}')
    if args.dry_run:
        return 0

    done = {(r['tracker'], r['subset'], r['split']) for r in load_done_rows()}
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for tracker in args.trackers:
            name = Path(tracker).stem
            pending = [s for s in args.subsets
                       if (name, s, args.split) not in done]
            if not pending:
                print(f'[INFO] {name}: все поднаборы уже в CSV — пропуск')
                continue
            out_dir = COMPARE_RUNS / name
            print(f'[INFO] {name}: live-трекинг {len(all_ids)} видео '
                  f'(готовые json пропускаются)')
            run_tracker_on_videos(args.model, all_ids, out_dir,
                                  args.split, tracker)
            for subset in pending:
                vids = by_subset[subset]
                metrics = evaluate_tracking_bdd(
                    out_dir, vids, out_dir / f'_eval_{subset}.json', args.split)
                writer.writerow([name, subset, args.split, len(vids),
                                 *(metrics.get(k, '') for k in METRIC_KEYS)])
                f.flush()
                print(f'[INFO] {name}/{subset}: '
                      f'HOTA={metrics.get("HOTA", "?")} '
                      f'IDF1={metrics.get("IDF1", "?")}', flush=True)

    write_choice_note(load_done_rows())
    print(f'[INFO] готово: {RESULTS_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
