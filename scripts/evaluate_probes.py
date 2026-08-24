"""Оценка коротких прогонов по поднаборам: ночь, дождь, туман и размеры.

Зачем отдельно от evaluate_baseline.py: тот считает три сида baseline и
пишет свою таблицу. Здесь сравниваются между собой пробы из узла Е —
контроль без блока и версии с DRB, — и вопрос другой. Общая точность у
блока ниже, но тема работы про сложные сцены, а самое слабое место
базовой модели именно там: мелкие объекты ночью дают AP 0.051 против
0.485 на крупных при нормальном освещении. Общее среднее этого не видит.

Выход: experiments/results/probe_subsets.csv (модель, поднабор, метрика,
значение) плюс сводка в консоль с разницей относительно контроля.

Запуск: python scripts/evaluate_probes.py
        python scripts/evaluate_probes.py --models base12 w0 --step 2
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.detection_metrics import evaluate_on_subsets  # noqa: E402
from datasets.subset_filter import filter_videos_by_subset, load_video_attributes  # noqa: E402

PROBE_ROOT = PROJECT_ROOT / 'runs' / 'init_probe'
OUT_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'probe_subsets.csv'
SUBSETS = ('normal', 'night', 'rain', 'fog')
CONTROL = 'base12'
# По умолчанию — контроль и единственная версия блока, совместимая с
# текущим кодом. Пробы w0/w2/w4 считались до перехода на радиальные кольца,
# их модель новым кодом не читается: маска там задана по каждому отсчёту
# спектра, а не по кольцам, и размеры не сходятся.
DEFAULT = ('base12', 'drb_ad', 'drb_bp')
# Метрики, ради которых всё затевается.
KEY = ('mAP50', 'AP_small', 'AP_medium', 'AP_large',
       'AR50_small', 'AP50_small')
# Метрики трекинга. HOTA с разбивкой на DetA и AssA инструмент bdd100k не
# считает — он даёт классические CLEAR MOT. Но разбивка ошибок отвечает на
# тот же вопрос даже прямее: пропуски (FN) — это провал детекции, переброс
# личности (IDSw) — провал связывания, ложные (FP) — провал точности.
TRACK_KEY = ('MOTA', 'IDF1', 'MOTP', 'FN', 'FP', 'IDSw', 'MT', 'ML')


def weights_of(pid: str) -> Path:
    """Веса по имени: сначала среди проб узла Е, потом среди обученных моделей.

    Второй путь нужен, чтобы сравнивать пробы с настоящим стосерийным
    baseline, не трогая evaluate_baseline.py — тот переписывает свою
    таблицу целиком и при запуске на одном сиде потерял бы два других.
    """
    probe = PROBE_ROOT / pid / 'seed42' / 'weights' / 'best.pt'
    if probe.exists():
        return probe
    # Имя вида «drb_ad100_seed42» — ищем каталог самого прогона. Копия в
    # weights/ делается после обучения, у идущего прогона её ещё нет, а
    # best.pt в его каталоге уже есть и пригоден к оценке.
    if '_seed' in pid:
        grp, _, sd = pid.rpartition('_seed')
        if sd.isdigit():
            for base in (PROJECT_ROOT / 'runs', PROBE_ROOT):
                w = base / grp / f'seed{sd}' / 'weights' / 'best.pt'
                if w.exists():
                    return w
    return PROJECT_ROOT / 'weights' / pid / 'best.pt'


def run_tracking(todo, subsets, out_csv: Path) -> int:
    """Метрики трекинга по поднаборам. Пишет свою таблицу, чужие не трогает."""
    from evaluation.tracking_metrics import evaluate_tracking_bdd, run_tracker_on_videos

    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(p.name for p in
                     (PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track'
                      / 'val').iterdir() if p.is_dir())

    rows, res = [], {}
    for name, w in todo:
        res[name] = {}
        for subset in subsets:
            vids = filter_videos_by_subset(attrs, all_ids, subset)
            print(f'\n=== {name} / {subset}: {len(vids)} видео ===', flush=True)
            track_dir = PROJECT_ROOT / 'runs' / 'track' / f'{name}_{subset}'
            run_tracker_on_videos(str(w), vids, track_dir)
            m = evaluate_tracking_bdd(
                track_dir, vids,
                PROJECT_ROOT / 'experiments' / 'results'
                / f'track_{name}_{subset}.json')
            res[name][subset] = m
            for metric, value in m.items():
                rows.append([name, subset, metric, value])
            print(f'[OK ] {subset}: ' + ', '.join(
                f'{k} {m[k]:.1f}' for k in TRACK_KEY
                if isinstance(m.get(k), (int, float))), flush=True)

    out = out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['модель', 'поднабор', 'метрика', 'значение'])
        wr.writerows(rows)

    print('\nИз чего складывается потеря качества трекинга:')
    for name in res:
        for subset, m in res[name].items():
            fn, fp, ids = m.get('FN'), m.get('FP'), m.get('IDSw')
            if None in (fn, fp, ids):
                continue
            tot = fn + fp + ids
            print(f'  {name} / {subset}')
            print(f'    пропуски (провал детекции)  {fn:>7.0f}  {fn/tot:>5.0%}')
            print(f'    ложные (провал точности)    {fp:>7.0f}  {fp/tot:>5.0%}')
            print(f'    перебросы (провал связей)   {ids:>7.0f}  {ids/tot:>5.0%}')
            worst = max((fn, 'детекция'), (fp, 'точность'), (ids, 'связывание'))[1]
            print(f'    -> главный источник потерь: {worst}')
    print(f'\n[OK ] {out}')
    return 0


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', default=list(DEFAULT),
                    help='какие прогоны из узла Е оценивать')
    ap.add_argument('--step', type=int, default=1,
                    help='шаг прореживания кадров; 1 — полная выборка')
    ap.add_argument('--subsets', nargs='+', default=list(SUBSETS),
                    help='какие поднаборы считать')
    ap.add_argument('--out', default=None,
                    help='куда писать таблицу; по умолчанию probe_subsets.csv. '
                         'Отдельный файл нужен для разовых замеров: таблица '
                         'переписывается целиком, и прошлые строки пропали бы')
    ap.add_argument('--tracking', action='store_true',
                    help='вместо детекции посчитать метрики трекинга: HOTA '
                         'раскладывается на DetA и AssA, и видно, детекция '
                         'узкое место или связывание (долго, ~30-60 мин)')
    args = ap.parse_args()
    subsets = tuple(args.subsets)
    out_csv = Path(args.out) if args.out else OUT_CSV
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / 'experiments' / 'results' / out_csv

    todo = [(m, weights_of(m)) for m in args.models]
    missing = [m for m, w in todo if not w.exists()]
    if missing:
        print(f'[WARN] нет весов, пропускаю: {", ".join(missing)}')
    todo = [(m, w) for m, w in todo if w.exists()]
    if not todo:
        print('[ERR ] оценивать нечего — ни одного прогона не найдено')
        return 1

    if args.tracking:
        # Своя таблица: probe_tracking.csv переписывается целиком, и прогон
        # одной модели стёр бы уже посчитанный baseline.
        if args.out is None:
            out_csv = PROJECT_ROOT / 'experiments' / 'results' / 'probe_tracking.csv'
        return run_tracking(todo, subsets, out_csv)

    rows, res = [], {}
    for name, w in todo:
        print(f'\n=== {name}: оценка по поднаборам ===', flush=True)
        # Своя рабочая папка на прогон: иначе все пробы писали бы в один
        # каталог (имя берётся из родителя весов, а он у всех «weights»).
        det = evaluate_on_subsets(
            str(w), list(subsets), frame_step=args.step,
            workdir=PROJECT_ROOT / 'runs' / 'eval_probe' / name)
        res[name] = det
        for subset, metrics in det.items():
            for metric, value in metrics.items():
                rows.append([name, subset, metric, f'{value:.6f}'])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['модель', 'поднабор', 'метрика', 'значение'])
        wr.writerows(rows)

    base = res.get(CONTROL)
    for metric in KEY:
        print(f'\n{metric} по поднаборам'
              + (f' (в скобках — разница с контролем {CONTROL})' if base else ''))
        head = f'{"модель":<12}' + ''.join(f'{s:>18}' for s in subsets)
        print(head)
        print('-' * len(head))
        for name in res:
            line = f'{name:<12}'
            for s in subsets:
                v = res[name].get(s, {}).get(metric)
                if v is None:
                    line += f'{"—":>18}'
                elif base and name != CONTROL and s in base:
                    d = v - base[s].get(metric, v)
                    line += f'{v:>11.4f} ({d:+.4f})'
                else:
                    line += f'{v:>18.4f}'
            print(line)

    if base and len(res) > 1:
        print('\nГлавная клетка — мелкие объекты ночью:')
        for name in res:
            v = res[name].get('night', {}).get('AP_small')
            if v is None:
                continue
            d = v - base['night']['AP_small']
            mark = '' if name == CONTROL else f'   {d:+.4f} к контролю'
            print(f'  {name:<12} {v:.4f}{mark}')

    print(f'\n[OK ] {OUT_CSV}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
