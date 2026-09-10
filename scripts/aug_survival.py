"""Выбивает ли аугментация молодые объекты чаще прочих.

Зачем. Гейт Этапа 2 показал устойчивый дефицит: доля объектов возраста 0-1,
дошедшая до функции потерь, ниже доли в метках (9.81% против 10.96%), причём
обучение полностью детерминировано — значит это не случайность. Разложение по
корзинам показало, что дефицит сосредоточен в корзине 0.

Гипотеза. Мозаичная аугментация склеивает четыре кадра и обрезает результат;
рамки, вышедшие за границу, удаляются, а `RandomPerspective.box_candidates`
дополнительно отбрасывает рамки, потерявшие после преобразования более 90%
площади или ставшие уже двух пикселей. Молодые объекты в среднем мельче и чаще
находятся у края кадра — они входят в сцену. Значит они выбывают чаще прочих.

Если гипотеза верна, аугментация систематически ослабляет ровно тот сигнал,
который метод собирается усиливать, и это надо знать до Этапа 5.

Как меряем. Одна эпоха в трёх режимах, отличающихся ровно одной настройкой:

    full     — как в baseline (мозаика 1.0, масштаб и сдвиг по умолчанию);
    no_mosaic — мозаика отключена, остальное как в baseline;
    no_geom   — отключены мозаика, масштаб и сдвиг.

Доля каждой корзины, дошедшая до функции потерь, сравнивается с долей в
метках обучающего списка. Обучение здесь не цель: хватает одного прохода
загрузчика, поэтому эпоха одна.

Ожидание, записанное заранее: в режиме `full` корзина 0 недопредставлена,
в `no_geom` дефицит исчезает или становится заметно меньше.

Запуск: python scripts/aug_survival.py
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.birth_gate import (  # noqa: E402
    BIRTH_ROOT, GEN, MINI_ROOT, WEIGHTS, age_share, make_lists,
)

RUNS = PROJECT_ROOT / 'runs' / 'aug_survival'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'aug_survival.csv'
NAMES = {0: 'возраст 0', 1: 'возраст 1', 2: 'возраст 2',
         3: 'возраст 3-4', 4: 'возраст 5+', -1: 'без признака'}

MODES = {
    'full': {},
    'no_mosaic': {'mosaic': 0.0},
    'no_geom': {'mosaic': 0.0, 'scale': 0.0, 'translate': 0.0},
}


def label_shares(list_path: Path) -> dict[int, float]:
    """Доля каждой корзины в самих файлах меток обучающего списка."""
    from collections import Counter
    c, n = Counter(), 0
    for img in list_path.read_text(encoding='utf-8').split():
        parts = list(Path(img).parts)
        if 'images' not in parts:
            continue
        parts[parts.index('images')] = 'labels'
        lab = Path(*parts).with_suffix('.txt')
        if not lab.exists():
            continue
        for line in lab.read_text(encoding='utf-8').splitlines():
            if line.strip():
                v = float(line.split()[0])
                c[round((v - int(v)) / 0.1) - 1] += 1
                n += 1
    return {b: 100.0 * c[b] / n for b in sorted(c)} if n else {}


def run_epoch(tag: str, yml: Path, over: dict, seed: int) -> dict[int, float]:
    """Одна эпоха с заданными настройками аугментации; доли по корзинам."""
    from ultralytics import YOLO

    from training import birth_weight
    birth_weight.reset()
    birth_weight.install(alpha=0.0)

    out_dir = RUNS / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model = YOLO(str(WEIGHTS))
    model.train(data=str(yml), epochs=1, imgsz=640, batch=8, workers=4,
                seed=seed, deterministic=True, project=str(RUNS), name=tag,
                exist_ok=True, verbose=False, plots=False, val=False, **over)

    # Корзина -1 — объекты БЕЗ признака, то есть валидационные: их метки
    # обычные, целочисленные. Валидатор тоже считает потери и потому проходит
    # через патч. Доля этих объектов зависит от режима: при мозаике обучающие
    # объекты проходят через потери вчетверо чаще, и валидация «весит» меньше.
    # Если их не исключить, доли корзин между режимами несопоставимы.
    stats = {b: n for b, n in birth_weight.STATS.items() if b >= 0}
    n_val = birth_weight.STATS.get(-1, 0)
    total = sum(stats.values())
    shares = {b: 100.0 * n / total for b, n in stats.items()} if total else {}
    print(f'    объектов обучения {total}, валидации отброшено {n_val}')
    birth_weight.uninstall()
    return shares


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    args = ap.parse_args()

    yml = make_lists('aug_survival', BIRTH_ROOT, MINI_ROOT)
    ref = label_shares(GEN / 'aug_survival_train.txt')

    print('=== ВЫЖИВАЕМОСТЬ ОБЪЕКТОВ ПРИ АУГМЕНТАЦИИ ===')
    print('гипотеза: мозаика и геометрия выбивают молодые объекты чаще прочих,')
    print('          потому что те мельче и чаще у края кадра')
    print('ожидание: в режиме full корзина 0 недопредставлена, в no_geom')
    print('          дефицит исчезает или заметно уменьшается')
    print(f'режимы: {args.modes}')
    print(f'сид {args.seed}, одна эпоха на режим\n')
    print('доли корзин в МЕТКАХ обучающего списка:')
    for b in sorted(ref):
        print('  %-14s %6.2f%%' % (NAMES.get(b, b), ref[b]))
    print(flush=True)

    got = {}
    for m in args.modes:
        if m not in MODES:
            print(f'[ПРОПУСК] неизвестный режим {m}')
            continue
        print(f'--- {m}: правки {MODES[m] or "нет, как в baseline"} ---',
              flush=True)
        got[m] = run_epoch(m, yml, MODES[m], args.seed)

    rows = []
    for b in sorted(ref):
        rows.append(('метки', NAMES.get(b, b), 'доля_%', round(ref[b], 2)))

    print('\n=== РЕЗУЛЬТАТ: доля корзины, дошедшая до функции потерь ===')
    hdr = '%-14s %9s' % ('корзина', 'метки')
    for m in got:
        hdr += ' %11s' % m
    print(hdr)
    for b in sorted(ref):
        line = '%-14s %8.2f%%' % (NAMES.get(b, b), ref[b])
        for m in got:
            v = got[m].get(b, 0.0)
            line += ' %10.2f%%' % v
            rows.append((m, NAMES.get(b, b), 'доля_%', round(v, 2)))
        print(line)

    print('\n=== ДЕФИЦИТ относительно меток, процентных пунктов ===')
    hdr = '%-14s' % 'корзина'
    for m in got:
        hdr += ' %11s' % m
    print(hdr)
    for b in sorted(ref):
        line = '%-14s' % NAMES.get(b, b)
        for m in got:
            d = got[m].get(b, 0.0) - ref[b]
            line += ' %+10.2f ' % d
            rows.append((m, NAMES.get(b, b), 'дефицит_пп', round(d, 2)))
        print(line)

    print('\n=== ОТНОСИТЕЛЬНАЯ ВЫЖИВАЕМОСТЬ (доля дошедших / доля в метках) ===')
    print('единица означает «выбывают наравне со всеми», меньше единицы —')
    print('«выбывают чаще прочих»')
    hdr = '%-14s' % 'корзина'
    for m in got:
        hdr += ' %11s' % m
    print(hdr)
    for b in sorted(ref):
        line = '%-14s' % NAMES.get(b, b)
        for m in got:
            r = got[m].get(b, 0.0) / ref[b] if ref[b] else 0.0
            line += ' %11.3f' % r
            rows.append((m, NAMES.get(b, b), 'выживаемость', round(r, 3)))
        print(line)

    print('\n=== ВЫЖИВАЕМОСТЬ ВОЗРАСТА 0 ОТНОСИТЕЛЬНО ЗРЕЛЫХ ===')
    print('главный показатель: он не зависит от того, сколько раз объект')
    print('проходит через потери, и потому сравним между режимами')
    rel = {}
    for m in got:
        s0 = got[m].get(0, 0.0) / ref[0] if ref.get(0) else 0.0
        s4 = got[m].get(4, 0.0) / ref[4] if ref.get(4) else 0.0
        rel[m] = s0 / s4 if s4 else 0.0
        rows.append((m, 'возраст 0', 'выживаемость_к_зрелым', round(rel[m], 3)))
        print('  %-12s %.3f' % (m, rel[m]))

    if 'full' in got and 'no_geom' in got:
        d_full = rel['full'] - 1.0
        d_geom = rel['no_geom'] - 1.0
        print(f'\nотставание возраста 0 от зрелых: full {100 * d_full:+.1f}%, '
              f'no_geom {100 * d_geom:+.1f}%')
        if d_full < -0.02 and d_geom > d_full + 0.02:
            print('ВЫВОД: гипотеза подтверждена — геометрическая аугментация')
            print('  выбивает молодые объекты чаще прочих. Значит она ослабляет')
            print('  ровно тот сигнал, который усиливает метод, и параметры')
            print('  аугментации надо включить в подбор на Этапе 5.')
        elif d_full >= -0.1:
            print('ВЫВОД: дефицита нет, гипотеза не подтверждена.')
        else:
            print('ВЫВОД: дефицит есть, но отключение геометрии его не убирает —')
            print('  причина не в ней. Искать в другом месте конвейера.')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['режим', 'корзина', 'метрика', 'значение'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
