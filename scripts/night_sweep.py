"""Ночная кампания замеров: порог уверенности и разрешение входа.

Главный вопрос работы, до сих пор не отвеченный: из 127 687 объектов ночного
поднабора трекер не находит 59 765 — 47%, это четыре пятых всех потерь. Но
неизвестно, какого рода эта потеря:

    (а) объекты РЕАЛЬНО не видны при данном разрешении — тогда лечение
        архитектурное: разрешение входа, голова P2, и это инженерия;
    (б) объекты находятся, но их балл ниже порога и они отсекаются — тогда
        лечение в пороге и в карте качества, и это ровно тема работы.

Замер разделяет диагнозы: валидация при пороге 0.001 против 0.25 и при
разрешении 1280 против 640. Разница по полноте мелких отвечает на вопрос.

Обучения нет, только прогоны валидации на зафиксированной контрольной модели.

Запуск: python scripts/night_sweep.py
"""
import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

WEIGHTS = PROJECT_ROOT / 'weights' / 'baseline_seed42' / 'best.pt'
DATA = PROJECT_ROOT / 'runs' / 'eval_probe' / 'base12' / 'yaml' / 'eval_night_val.yaml'
OUT = PROJECT_ROOT / 'experiments' / 'results' / 'night_sweep.csv'

# (имя, разрешение, порог уверенности)
RUNS = [
    ('640_conf025', 640, 0.25),
    ('640_conf001', 640, 0.001),
    ('1280_conf001', 1280, 0.001),
    ('960_conf001', 960, 0.001),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings('ignore')
    from ultralytics import YOLO

    for p in (WEIGHTS, DATA):
        if not p.exists():
            raise SystemExit(f'нет файла: {p}')

    rows = []
    for name, imgsz, conf in RUNS:
        t0 = time.time()
        print(f'\n[RUN ] {name}: imgsz={imgsz}, conf={conf}', flush=True)
        model = YOLO(str(WEIGHTS))
        # batch на 1280 уменьшаем — иначе не влезет в 12 ГБ
        b = args.batch if imgsz <= 960 else max(2, args.batch // 4)
        m = model.val(data=str(DATA), imgsz=imgsz, conf=conf, batch=b,
                      device=0, verbose=False, plots=False, save_json=False,
                      project=str(PROJECT_ROOT / 'runs' / 'night_sweep'), name=name,
                      exist_ok=True)
        d = m.results_dict
        got = {
            'mAP50': float(d.get('metrics/mAP50(B)', float('nan'))),
            'mAP50-95': float(d.get('metrics/mAP50-95(B)', float('nan'))),
            'precision': float(d.get('metrics/precision(B)', float('nan'))),
            'recall': float(d.get('metrics/recall(B)', float('nan'))),
        }
        # разбивка по размерам, если инструмент её отдаёт
        for attr, key in (('ap_small', 'AP_small'), ('ap_medium', 'AP_medium'),
                          ('ap_large', 'AP_large')):
            v = getattr(m.box, attr, None)
            if v is not None:
                try:
                    got[key] = float(v)
                except Exception:
                    pass
        got['секунд'] = round(time.time() - t0, 1)
        for k, v in got.items():
            rows.append((name, imgsz, conf, k, v))
        print('  ' + '  '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                               for k, v in got.items()), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['прогон', 'разрешение', 'порог', 'метрика', 'значение'])
        w.writerows(rows)
    print(f'\n[OK ] {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
