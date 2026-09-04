"""Порог трекера, зависящий от условий: настройка на обучающих видео, замер на отложенных.

Зачем. Три замера 04-05.09 сошлись на том, что ночью детектор не слепнет, а
недооценивает себя: 22.7% объектов найдены верно и отсечены порогом; ByteTrack
не достаёт их конструктивно; и отсекаются они несправедливо — ночная детекция
с тем же баллом вернее дневной на 0.098 в рабочей полосе. Сетка порогов это
подтвердила независимо: оптимум днём 0.25, ночью 0.15, знак держится на обеих
половинах выборки и по обеим метрикам.

Здесь из наблюдения делается МЕТОД. Условие на применении неизвестно — метки
«ночь» в кадре нет, — поэтому порог выводится из наблюдаемой величины: средней
яркости видео.

Протокол, и он тут главное. Всё, что настраивается, настраивается ТОЛЬКО на
обучающей половине видео; всё, что заявляется, меряется на отложенной. Иначе
получается прирост-мираж: подбор коэффициентов и замер на одних данных дают
улучшение, исчезающее на новых кадрах.

Разбиение ПО ВИДЕО и внутри каждого условия через одно, чтобы обе половины
несли и день, и ночь, а похожие сцены одной поездки не собрались в одну.

Четыре плеча сравнения на отложенной половине:

    0.25          умолчание ByteTrack — практическая точка отсчёта;
    лучший фикс.  лучший ЕДИНЫЙ порог, выбранный НА ОБУЧАЮЩЕЙ половине —
                  научная точка отсчёта: с ней сравнение честное, потому что
                  победа над одним лишь умолчанием ничего не доказывает;
    регрессия     tau = a * яркость + b, коэффициенты с обучающей половины;
    две ступени   яркость выше/ниже порога разделения -> два значения tau,
                  и порог разделения, и оба значения — с обучающей половины.

Две ступени включены намеренно: регрессия по 25 точкам, где оптимум каждого
видео сам оценён по одному короткому ролику, может оказаться подгонкой шума.
Простое правило устойчивее и для текста защитимее — но только если выигрывает.

Запуск: python scripts/adaptive_threshold.py
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)
from evaluation.tracking_metrics import evaluate_tracking_bdd  # noqa: E402
from tracking.brightness_estimator import get_brightness_map  # noqa: E402
from tracking.brightness_regression import (  # noqa: E402
    apply_brightness_adaptive_threshold, fit_brightness_to_threshold,
)
from tracking.offline_association import make_tracker_args, run_association  # noqa: E402

CACHE = PROJECT_ROOT / 'runs' / 'det_cache_ctl'
RUNS = PROJECT_ROOT / 'runs' / 'adaptive_thr'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'adaptive_threshold.csv'


def evaluate(vids, split, tag, args_by_video=None, args_all=None):
    """Ассоциация + официальная оценка BDD100K. Возвращает (MOTA, IDF1)."""
    out_dir = RUNS / tag
    run_association(CACHE, vids, out_dir, split,
                    per_video_args=args_by_video, args=args_all)
    m = evaluate_tracking_bdd(out_dir, vids, out_dir / '_eval.json', split)
    def pick(*keys):
        for k in keys:
            if k in m and not isinstance(m[k], list):
                return float(m[k])
        return float('nan')
    return pick('MOTA', 'mMOTA'), pick('IDF1', 'mIDF1')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', default='val')
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.4])
    ap.add_argument('--buffer', type=int, default=30)
    ap.add_argument('--metric', default='MOTA', choices=['MOTA', 'IDF1'],
                    help='по какой метрике ищется оптимум каждого видео')
    args = ap.parse_args()

    have = sorted(p.stem for p in (CACHE / args.split).glob('*.npz'))
    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / f'bdd100k_labels_images_{args.split}.json')
    attrs = load_video_attributes(str(det_json))

    train, test = [], []
    for cond in ('normal', 'night'):
        vids = [v for v in filter_videos_by_subset(attrs, have, cond) if v in have]
        train += vids[0::2]
        test += vids[1::2]
    print(f'[INFO] обучающих видео {len(train)}, отложенных {len(test)}')

    bright = get_brightness_map(train + test, args.split)
    print(f'[INFO] яркость посчитана для {len(bright)} видео')

    # ---- 1. оптимум КАЖДОГО обучающего видео ----
    print('\n=== подбор на обучающей половине ===')
    per_video_opt: dict[str, float] = {}
    for i, v in enumerate(train, 1):
        best, best_val = None, -1e9
        for thr in args.thresholds:
            mota, idf1 = evaluate([v], args.split, f'tr_{v}_t{thr:g}',
                                  args_all=make_tracker_args(
                                      track_high_thresh=thr,
                                      track_buffer=args.buffer))
            val = mota if args.metric == 'MOTA' else idf1
            if val > best_val:
                best, best_val = thr, val
        per_video_opt[v] = best
        print(f'[INFO] {i}/{len(train)} {v}: яркость {bright.get(v, float("nan")):.3f}'
              f'  оптимум {best:g}  {args.metric}={best_val:.2f}', flush=True)

    # ---- 2. лучший ЕДИНЫЙ порог на обучающей половине ----
    fixed_scores = {}
    for thr in args.thresholds:
        mota, idf1 = evaluate(train, args.split, f'trall_t{thr:g}',
                              args_all=make_tracker_args(
                                  track_high_thresh=thr, track_buffer=args.buffer))
        fixed_scores[thr] = mota if args.metric == 'MOTA' else idf1
    best_fixed = max(fixed_scores, key=fixed_scores.get)
    print(f'\n[INFO] лучший единый порог на обучающей половине: {best_fixed:g}')

    # ---- 3. две настройки правила, обе с обучающей половины ----
    xs = [bright[v] for v in train if v in bright]
    ys = [per_video_opt[v] for v in train if v in bright]
    a, b, r2 = fit_brightness_to_threshold(xs, ys)
    print(f'[INFO] регрессия: tau = {a:.4f} * яркость + {b:.4f}, R2 = {r2:.3f}')

    cut = statistics.median(xs)
    dark = [per_video_opt[v] for v in train if bright.get(v, 1.0) <= cut]
    light = [per_video_opt[v] for v in train if bright.get(v, 1.0) > cut]
    tau_dark = statistics.median(dark) if dark else best_fixed
    tau_light = statistics.median(light) if light else best_fixed
    print(f'[INFO] две ступени: яркость <= {cut:.3f} -> tau {tau_dark:g}, '
          f'иначе tau {tau_light:g}')

    # ---- 4. замер на отложенной половине ----
    print('\n=== замер на отложенной половине ===')
    arms = {}
    arms['умолчание 0.25'] = evaluate(
        test, args.split, 'te_fixed0.25',
        args_all=make_tracker_args(track_high_thresh=0.25, track_buffer=args.buffer))
    arms[f'лучший фикс. {best_fixed:g}'] = evaluate(
        test, args.split, f'te_fixed{best_fixed:g}',
        args_all=make_tracker_args(track_high_thresh=best_fixed,
                                   track_buffer=args.buffer))
    reg_args = {v: make_tracker_args(
        track_high_thresh=apply_brightness_adaptive_threshold(
            bright.get(v, 0.5), a, b), track_buffer=args.buffer) for v in test}
    arms['регрессия по яркости'] = evaluate(
        test, args.split, 'te_reg', args_by_video=reg_args)
    step_args = {v: make_tracker_args(
        track_high_thresh=(tau_dark if bright.get(v, 1.0) <= cut else tau_light),
        track_buffer=args.buffer) for v in test}
    arms['две ступени по яркости'] = evaluate(
        test, args.split, 'te_step', args_by_video=step_args)

    base = arms['умолчание 0.25']
    ref = arms[f'лучший фикс. {best_fixed:g}']
    print('\n%-26s %9s %9s %11s %12s' % (
        'плечо', 'MOTA', 'IDF1', 'к умолч.', 'к лучш.фикс.'))
    rows = []
    for name, (mota, idf1) in arms.items():
        print('%-26s %9.3f %9.3f %+11.3f %+12.3f' % (
            name, mota, idf1, mota - base[0], mota - ref[0]))
        rows.append({'плечо': name, 'MOTA': round(mota, 4), 'IDF1': round(idf1, 4),
                     'дельта_к_умолчанию': round(mota - base[0], 4),
                     'дельта_к_лучшему_фикс': round(mota - ref[0], 4)})

    print('\n=== приговор ===')
    win = max(('регрессия по яркости', 'две ступени по яркости'),
              key=lambda k: arms[k][0])
    gain = arms[win][0] - ref[0]
    # Планка, а не голый знак. Плечи считаются на ОДНИХ И ТЕХ ЖЕ видео и на
    # одних детекциях, поэтому сравнение парное и абсолютный уровень не важен.
    # Но масштаб «сколько тут вообще ничего не значит» нужен, и его даёт
    # разброс между всеми плечами: если победа не больше расстояния между
    # плечами, которые обязаны различаться, — не различилось ничто.
    spread = max(v[0] for v in arms.values()) - min(v[0] for v in arms.values())
    print(f'  лучшее адаптивное плечо: {win}')
    print(f'  прирост к лучшему единому порогу: {gain:+.3f} MOTA')
    print(f'  разброс между всеми плечами:      {spread:.3f} MOTA')
    if gain > spread:
        print('\n  Прирост ПРЕВЫШАЕТ разброс между плечами. Адаптация даёт то,')
        print('  чего не даёт никакой фиксированный выбор.')
    elif gain > 0:
        print('\n  Прирост НЕ ПРЕВЫШАЕТ разброса между плечами: знак верный,')
        print('  величина неотличима от нуля. Разный оптимум по условиям')
        print('  существует (проверено разбиением), но порог трекера — слишком')
        print('  грубый рычаг, чтобы его монетизировать. Отдельно учесть, что')
        print('  отложенная половина СМЕШАННАЯ: лучший единый порог лежит между')
        print('  оптимумами условий и почти ничего не теряет ни на одном.')
    else:
        print(f'\n  Прирост отсутствует ({gain:+.3f} MOTA). Извлечь разный')
        print('  оптимум из яркости не удалось: признак слаб или выборки мало.')
    if r2 < 0.2:
        print(f'\n  Отдельно: регрессия по яркости несостоятельна (R2 = {r2:.3f}).')
        print('  Оптимум ОДНОГО ролика оценивается шумно — двести кадров и')
        print('  десяток объектов, MOTA такой выборки скачет на единицы.')
        print('  Правило «две ступени» держится только медианой внутри группы.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
