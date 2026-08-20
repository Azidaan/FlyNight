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

PROBE_ROOT = PROJECT_ROOT / 'runs' / 'init_probe'
OUT_CSV = PROJECT_ROOT / 'experiments' / 'results' / 'probe_subsets.csv'
SUBSETS = ('normal', 'night', 'rain', 'fog')
CONTROL = 'base12'
# По умолчанию — контроль и единственная версия блока, совместимая с
# текущим кодом. Пробы w0/w2/w4 считались до перехода на радиальные кольца,
# их модель новым кодом не читается: маска там задана по каждому отсчёту
# спектра, а не по кольцам, и размеры не сходятся.
DEFAULT = ('base12', 'drb12', 'boost')
# Метрики, ради которых всё затевается.
KEY = ('mAP50', 'AP_small', 'AP_medium', 'AP_large')


def weights_of(pid: str) -> Path:
    return PROBE_ROOT / pid / 'seed42' / 'weights' / 'best.pt'


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', default=list(DEFAULT),
                    help='какие прогоны из узла Е оценивать')
    ap.add_argument('--step', type=int, default=1,
                    help='шаг прореживания кадров; 1 — полная выборка')
    args = ap.parse_args()

    todo = [(m, weights_of(m)) for m in args.models]
    missing = [m for m, w in todo if not w.exists()]
    if missing:
        print(f'[WARN] нет весов, пропускаю: {", ".join(missing)}')
    todo = [(m, w) for m, w in todo if w.exists()]
    if not todo:
        print('[ERR ] оценивать нечего — ни одного прогона не найдено')
        return 1

    rows, res = [], {}
    for name, w in todo:
        print(f'\n=== {name}: оценка по поднаборам ===', flush=True)
        # Своя рабочая папка на прогон: иначе все пробы писали бы в один
        # каталог (имя берётся из родителя весов, а он у всех «weights»).
        det = evaluate_on_subsets(
            str(w), list(SUBSETS), frame_step=args.step,
            workdir=PROJECT_ROOT / 'runs' / 'eval_probe' / name)
        res[name] = det
        for subset, metrics in det.items():
            for metric, value in metrics.items():
                rows.append([name, subset, metric, f'{value:.6f}'])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['модель', 'поднабор', 'метрика', 'значение'])
        wr.writerows(rows)

    base = res.get(CONTROL)
    for metric in KEY:
        print(f'\n{metric} по поднаборам'
              + (f' (в скобках — разница с контролем {CONTROL})' if base else ''))
        head = f'{"модель":<12}' + ''.join(f'{s:>18}' for s in SUBSETS)
        print(head)
        print('-' * len(head))
        for name in res:
            line = f'{name:<12}'
            for s in SUBSETS:
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
