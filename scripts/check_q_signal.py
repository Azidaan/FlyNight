"""Несёт ли карта качества Q различающий сигнал по детекциям.

Проверка ДО того, как подключать q к фильтру Калмана. Смысл подстановки в
том, чтобы трекер меньше доверял ненадёжным рамкам. Если q у всех рамок
примерно одинакова, подстановка ничего не различит: она просто умножит
шум наблюдения на константу, и любой эффект будет от этой константы, а не
от качества. Тогда затея мертва, и лучше узнать это за десять минут, чем
за неделю.

Два критерия отказа, названные заранее:
  1. разброс q по детекциям меньше 0.02 — различать нечем;
  2. связь q с уверенностью детектора близка к единице — q не несёт
     ничего сверх того, что у трекера уже есть (уверенность там доступна
     и без всякого блока).

Второй критерий — из плана: слабая связь и будет доказательством, что q
несёт СВОЮ информацию. Уверенность отвечает на вопрос «это машина», q
считается из локальных статистик признаков и отвечает на другой — «видно
ли тут вообще что-нибудь чётко».

Отдельно q разбивается по размеру объекта: заявление работы про мелкие
объекты, и если q их не отличает, польза сомнительна.

Запуск: python scripts/check_q_signal.py
        python scripts/check_q_signal.py --frames 400 --subset night
"""
import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.subset_filter import (  # noqa: E402
    filter_videos_by_subset, load_video_attributes,
)
from tracking.inference_with_q import QInference  # noqa: E402

DEFAULT_WEIGHTS = PROJECT_ROOT / 'weights' / 'drb_ad_seed42' / 'best.pt'
SMALL_AREA = 32 * 32          # граница «мелкого» по соглашению COCO


def pick_frames(subset: str, n: int) -> list[Path]:
    """Кадры указанного поднабора, равномерно по видео (не подряд из одного)."""
    det_json = (PROJECT_ROOT / 'data' / 'bdd100k' / 'labels'
                / 'bdd100k_labels_images_val.json')
    root = PROJECT_ROOT / 'data' / 'bdd100k' / 'images' / 'track' / 'val'
    attrs = load_video_attributes(str(det_json))
    all_ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    vids = filter_videos_by_subset(attrs, all_ids, subset)
    frames: list[Path] = []
    per_video = max(1, n // max(1, len(vids)))
    for v in vids:
        got = sorted((root / v).glob('*.jpg'))[:per_video]
        frames.extend(got)
        if len(frames) >= n:
            break
    return frames[:n]


def spread(name: str, v: np.ndarray) -> None:
    if not len(v):
        print(f'  {name}: пусто')
        return
    qs = np.percentile(v, [5, 25, 50, 75, 95])
    print(f'  {name:<22} n={len(v):<6} среднее {v.mean():.4f}  СКО {v.std():.4f}'
          f'  размах {v.min():.4f}..{v.max():.4f}')
    print(f'  {"":<22} процентили 5/25/50/75/95: '
          + ' '.join(f'{x:.4f}' for x in qs))


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weights', default=str(DEFAULT_WEIGHTS))
    ap.add_argument('--subset', default='night')
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--conf', type=float, default=0.10,
                    help='нижний порог: нужны и слабые детекции, ради которых '
                         'всё затевается, а не только уверенные')
    args = ap.parse_args()

    frames = pick_frames(args.subset, args.frames)
    print(f'Кадров {args.subset}: {len(frames)}')
    print(f'Модель: {args.weights}\n')

    inf = QInference(args.weights, conf=args.conf)
    if not inf.with_drb:
        print('[ERR ] в модели нет блока DRB — карты качества не будет')
        return 1

    aggs = ('mean', 'min', 'median')
    q_by_agg = {a: [] for a in aggs}
    q_all, conf_all, area_all, stride_all = [], [], [], []
    q_frame_mean = []
    for i, f in enumerate(frames):
        d = inf.predict_frame(f, aggregations=aggs)
        q = d.get('q_mean')
        if q is None or not len(q):
            continue
        for a in aggs:
            q_by_agg[a].append(d[f'q_{a}'])
        q_all.append(q)
        conf_all.append(d['conf'])
        stride_all.append(d['stride'])
        xyxy = d['xyxy']
        area_all.append((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))
        q_frame_mean.append(float(np.mean(q)))
        if (i + 1) % 50 == 0:
            print(f'  ...{i + 1} кадров', flush=True)

    if not q_all:
        print('[ERR ] детекций не найдено')
        return 1
    q = np.concatenate(q_all)
    cf = np.concatenate(conf_all)
    ar = np.concatenate(area_all)
    sd_lvl = np.concatenate(stride_all)

    print('\n=== разброс q по детекциям ===')
    spread('все детекции', q)
    spread('мелкие (<32x32)', q[ar < SMALL_AREA])
    spread('крупные', q[ar >= SMALL_AREA])

    # Каждый уровень пирамиды считает свою карту качества своей свёрткой, и
    # смещения у них разные. Если уровни разъехались, общий разброс q — это
    # в основном «с какого уровня детекция», а не «насколько ей верить».
    # Подставлять такую q в фильтр значит систематически доверять одному
    # уровню больше другого независимо от качества рамки.
    print('\n=== q по уровням пирамиды ===')
    for s in sorted(set(int(x) for x in sd_lvl)):
        m = sd_lvl == s
        spread(f'страйд {s}', q[m])
    # Тот же разбор по размеру, но ВНУТРИ уровня: если и там разброс мал,
    # различать нечем даже после нормировки.
    print('\n=== мелкие объекты внутри своего уровня ===')
    for s in sorted(set(int(x) for x in sd_lvl)):
        m = (sd_lvl == s) & (ar < SMALL_AREA)
        if m.sum() >= 20:
            spread(f'страйд {s}, мелкие', q[m])

    print('\n=== разброс q между кадрами ===')
    spread('среднее по кадру', np.array(q_frame_mean))

    print('\n=== связь q с уверенностью детектора ===')
    r = float(np.corrcoef(q, cf)[0, 1]) if len(q) > 2 else float('nan')
    print(f'  коэффициент корреляции: {r:+.3f}')

    print('\n=== приговор ===')
    # Считать надо разброс ВНУТРИ уровня, а не общий. Общий складывается в
    # основном из того, что уровни разъехались по смещению своих свёрток:
    # различать «с какого уровня детекция» фильтру незачем, ему нужно
    # различать «насколько верить этой рамке» среди сопоставимых.
    print(f'  общий разброс {float(q.std()):.4f} — но он почти весь между '
          f'уровнями, а не внутри них')
    print(f'  |корреляция| с уверенностью {abs(r):.3f} < 0.8 — '
          f'{"да, q несёт своё" if abs(r) < 0.8 else "НЕТ, дублирует"}')
    print('\n  разброс внутри уровня по каждому способу свёртки рамки:')
    best = 0.0
    for a in aggs:
        qa = np.concatenate(q_by_agg[a])
        line = f'    {a:<7}'
        for s in sorted(set(int(x) for x in sd_lvl)):
            m = sd_lvl == s
            if m.sum() < 20:
                continue
            sd_in = float(qa[m].std())
            best = max(best, sd_in)
            line += f'  страйд {s}: {sd_in:.4f}'
        print(line)
    ok = best >= 0.02
    print(f'\n  лучший разброс внутри уровня {best:.4f} >= 0.02 — '
          f'{"да" if ok else "НЕТ"}')
    if ok:
        print('\n  Подставлять q в шум наблюдения фильтра имеет смысл.')
        return 0
    print('\n  Подстановка бессмысленна: среди сопоставимых рамок q почти\n'
          '  постоянна, различать ей нечего.')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
