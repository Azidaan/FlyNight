"""Гейт исполнимости (Этап 2): доходит ли признак возраста до функции потерь.

Вопрос гейта. Метки YOLO — ровно пять чисел в строке, номера трека там нет,
мозаичная аугментация склеивает четыре кадра и переупорядочивает объекты.
Прежде чем строить метод, надо убедиться, что признак «возраст объекта в
траектории» физически доезжает до функции потерь неискажённым.

Способ доставки — дробная часть класса (`scripts/make_birth_labels.py`),
снятие — патч `v8DetectionLoss.preprocess` (`training/birth_weight.py`).

Два прогона по три эпохи на mini-BDD, отличающиеся РОВНО одним условием:

    control — обычные метки mini-BDD, патч не ставится;
    birth   — закодированные метки, патч установлен, alpha = 0.

Критерии, записанные ДО прогона:

    1. Прогон `birth` не падает и доходит до конца.
    2. Доля объектов возраста 0-1, увиденная функцией потерь, совпадает с
       посчитанной по разметке (8.44% на mini-BDD) с точностью 0.5 пп.
       Это доказывает, что мозаика не теряет и не путает признак.
    3. Метрики `birth` и `control` совпадают: при alpha = 0 класс
       восстанавливается флорингом, поэтому обучение обязано быть тождественным.
       Допуск на недетерминизм CUDA — 0.005 mAP50-95.

Провал критерия 3 означает, что кодирование меняет обучение само по себе,
и метод в таком виде строить нельзя.

Запуск: python scripts/birth_gate.py --epochs 3
"""
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GEN = PROJECT_ROOT / 'training' / 'configs' / 'generated'
MINI_TRAIN = GEN / 'mini_bdd_smoke_train.txt'
MINI_VAL = GEN / 'mini_bdd_smoke_val.txt'
WEIGHTS = PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt'
RUNS = PROJECT_ROOT / 'runs' / 'birth_gate'
NAMES = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
         4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'}


def make_lists(tag: str, src_root: str, dst_root: str) -> Path:
    """Списки кадров и data-yaml с подменённым корнем набора."""
    GEN.mkdir(parents=True, exist_ok=True)
    out = {}
    for split, src in (('train', MINI_TRAIN), ('val', MINI_VAL)):
        lines = src.read_text(encoding='utf-8').splitlines()
        repl = [l.replace(src_root, dst_root) for l in lines if l.strip()]
        p = GEN / f'{tag}_{split}.txt'
        p.write_text('\n'.join(repl) + '\n', encoding='utf-8')
        out[split] = p
    yml = GEN / f'{tag}.yaml'
    body = [f'# Автогенерация: scripts/birth_gate.py, конфигурация {tag}',
            f'train: {out["train"].as_posix()}',
            f'val: {out["val"].as_posix()}', 'names:']
    body += [f'  {i}: {n}' for i, n in NAMES.items()]
    yml.write_text('\n'.join(body) + '\n', encoding='utf-8')
    return yml


def run(tag: str, yml: Path, epochs: int, patched: bool, seed: int) -> dict:
    from ultralytics import YOLO

    if patched:
        from training import birth_weight
        birth_weight.reset()
        birth_weight.install(alpha=0.0)

    out_dir = RUNS / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    model = YOLO(str(WEIGHTS))
    model.train(data=str(yml), epochs=epochs, imgsz=640, batch=8, workers=4,
                seed=seed, deterministic=True, project=str(RUNS), name=tag,
                exist_ok=True, verbose=False, plots=False, val=True)
    m = model.trainer.metrics
    res = {k.split('(')[0].split('/')[-1]: float(v) for k, v in m.items()
           if isinstance(v, (int, float))}

    stats = {}
    if patched:
        from training import birth_weight
        stats = birth_weight.report()
        birth_weight.uninstall()
    return {'metrics': res, 'stats': stats}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if not WEIGHTS.exists():
        raise SystemExit(f'нет предобученных весов: {WEIGHTS}')
    if not (PROJECT_ROOT / 'data' / 'birth_smoke' / 'labels').exists():
        raise SystemExit('нет меток с признаком: запустите '
                         'scripts/make_birth_labels.py')

    # критерии печатаются ДО чисел
    print('=== ГЕЙТ ИСПОЛНИМОСТИ (Этап 2) ===')
    print('критерий 1: прогон с патчем доходит до конца')
    print('критерий 2: доля возраста 0-1 в потерях = 8.44% ± 0.5 пп')
    print('критерий 3: метрики birth и control совпадают, допуск 0.005 mAP50-95')
    print(f'эпох {args.epochs}, сид {args.seed}\n', flush=True)

    yml_ctl = make_lists('birth_gate_control',
                         'data\\bdd100k_mot_mini', 'data\\bdd100k_mot_mini')
    yml_bir = make_lists('birth_gate_birth',
                         'data\\bdd100k_mot_mini', 'data\\birth_smoke')

    print('--- прогон control (обычные метки, без патча) ---', flush=True)
    ctl = run('control', yml_ctl, args.epochs, patched=False, seed=args.seed)
    print('--- прогон birth (метки с признаком, патч установлен) ---', flush=True)
    bir = run('birth', yml_bir, args.epochs, patched=True, seed=args.seed)

    print('\n=== РЕЗУЛЬТАТ ГЕЙТА ===')
    key = 'mAP50-95' if 'mAP50-95' in ctl['metrics'] else None
    if key is None:
        for k in ctl['metrics']:
            if '50-95' in k:
                key = k
                break
    print('%-14s %12s %12s %10s' % ('метрика', 'control', 'birth', 'разница'))
    ok3 = True
    for k in sorted(set(ctl['metrics']) & set(bir['metrics'])):
        a, b = ctl['metrics'][k], bir['metrics'][k]
        print('%-14s %12.5f %12.5f %+10.5f' % (k, a, b, b - a))
        if key and k == key and abs(b - a) > 0.005:
            ok3 = False

    young = bir['stats'].get('young')
    ok1 = bool(bir['metrics'])
    ok2 = young is not None and abs(young - 8.44) <= 0.5

    print()
    print(f'критерий 1 (прогон дошёл до конца):        {"ПРОЙДЕН" if ok1 else "ПРОВАЛЕН"}')
    print(f'критерий 2 (доля возраста 0-1 = 8.44±0.5): '
          f'{"ПРОЙДЕН" if ok2 else "ПРОВАЛЕН"}'
          + (f', получено {young:.2f}%' if young is not None else ', нет данных'))
    print(f'критерий 3 (birth = control):              {"ПРОЙДЕН" if ok3 else "ПРОВАЛЕН"}')
    print()
    print('ВЫВОД: ' + ('гейт пройден, признак доезжает до функции потерь '
                       'неискажённым — можно строить Этап 4'
                       if (ok1 and ok2 and ok3) else
                       'гейт НЕ пройден, разбираться до перехода к Этапу 4'))
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == '__main__':
    sys.exit(main())
