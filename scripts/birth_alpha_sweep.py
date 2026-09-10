"""Этап 5, короткая часть: подбор веса alpha на 12 эпохах.

Зачем коротко. План запрещает гнать полные стоэпошные прогоны до того, как
известно, работает ли механизм вообще. Двенадцать эпох на прореженном train
(56 350 кадров) стоят около 42 минут против 5.9 часа у полного прогона.

Почему контроль тоже 12 эпох. Число эпох задаёт расписание скорости обучения:
при 100 эпохах спад растянут на сотню, и режим другой. Сравнивать 12-эпошный
прогон со стоэпошным нельзя.

Меняется РОВНО одно условие — alpha. Все конфигурации обучаются на одних и тех
же метках с признаком возраста; при alpha = 0 веса не применяются, и это
доказано тождественным: гейт Этапа 2 дал совпадение с обычным обучением до
пятого знака по всем метрикам.

ЧТО СЧИТАЕТСЯ УСПЕХОМ (записано до прогона):

  идея живёт  — доля найденных на возрасте 0 выросла, на возрасте 10+ осталась
                в пределах шума. Эффект адресный;
  идея пустая — выросло всё сразу. Это перекос рабочей точки, а не метод: то же
                самое даёт бесплатное снижение new_track_thresh;
  идея мертва — не выросло ничего, либо общий mAP просел сильнее, чем выиграла
                мишень.

Третий исход наиболее вероятен по литературе: в работах по добавочным функциям
потерь зафиксировано, что при росте веса общий mAP падает. Поэтому берутся три
значения alpha, а не одно.

Порядок: сперва обучаются ВСЕ конфигурации, потом строятся кэши и считается
лестница возраста. Так поломка на замере не отнимает уже потраченное обучение.

Запуск: python scripts/birth_alpha_sweep.py --alphas 0 0.5 1.0 2.0 --epochs 12
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GEN = PROJECT_ROOT / 'training' / 'configs' / 'generated'
PY = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
WEIGHTS = PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt'
RUNS = PROJECT_ROOT / 'runs' / 'birth_alpha'
CACHES = PROJECT_ROOT / 'runs' / 'birth_alpha_cache'
BIRTH_ROOT = 'data\\bdd100k_birth'
FULL_ROOT = 'data\\bdd100k\\images\\track\\train'
NAMES = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
         4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle'}


def make_yaml() -> Path:
    """train — метки с признаком, val — обычные.

    Валидатор не проходит через патч функции потерь и сравнивал бы целые
    предсказанные классы с дробными эталонными, обнуляя метрики (проверено на
    гейте Этапа 2). Признак нужен только при обучении.
    """
    src = (GEN / 'full_step5_train.txt').read_text(encoding='utf-8').splitlines()
    train = GEN / 'birth_alpha_train.txt'
    train.write_text('\n'.join(l.replace(FULL_ROOT, BIRTH_ROOT + '\\images')
                               for l in src if l.strip()) + '\n',
                     encoding='utf-8')
    yml = GEN / 'birth_alpha.yaml'
    body = ['# Автогенерация: scripts/birth_alpha_sweep.py',
            f'train: {train.as_posix()}',
            f'val: {(GEN / "full_step5_val.txt").as_posix()}', 'names:']
    body += [f'  {i}: {n}' for i, n in NAMES.items()]
    yml.write_text('\n'.join(body) + '\n', encoding='utf-8')
    return yml


def train_one(alpha: float, yml: Path, epochs: int, seed: int,
              profile: str) -> Path:
    from ultralytics import YOLO

    from training import birth_weight
    birth_weight.reset()
    birth_weight.install(alpha=alpha, profile=profile)
    tag = f'a{alpha:g}'.replace('.', '')
    model = YOLO(str(WEIGHTS))
    model.train(data=str(yml), epochs=epochs, imgsz=640, batch=16, workers=4,
                seed=seed, deterministic=True, project=str(RUNS), name=tag,
                exist_ok=True, verbose=False, plots=False)
    birth_weight.report()
    birth_weight.uninstall()
    return RUNS / tag / 'weights' / 'best.pt'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--alphas', nargs='+', type=float, default=[0, 0.5, 1.0, 2.0])
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--profile', default='smooth', choices=['smooth', 'binary'])
    ap.add_argument('--videos', type=int, default=50,
                    help='видео для кэша при замере лестницы возраста')
    ap.add_argument('--skip-train', action='store_true')
    args = ap.parse_args()

    yml = make_yaml()
    print('=== ЭТАП 5, ПОДБОР alpha ===')
    print(f'alpha: {args.alphas}, эпох {args.epochs}, профиль {args.profile}')
    print(f'обучение: {yml}')
    print('успех = рост доли найденных на возрасте 0 БЕЗ роста на возрасте 10+')
    print('провал = рост везде (перекос порога) или падение общего mAP\n',
          flush=True)

    weights = {}
    for a in args.alphas:
        tag = f'a{a:g}'.replace('.', '')
        best = RUNS / tag / 'weights' / 'best.pt'
        if args.skip_train and best.exists():
            print(f'--- alpha={a}: веса уже есть, обучение пропущено ---')
        else:
            print(f'--- обучение alpha={a} ---', flush=True)
            best = train_one(a, yml, args.epochs, args.seed, args.profile)
        weights[a] = best

    print('\n=== замер лестницы возраста ===', flush=True)
    for a, w in weights.items():
        tag = f'a{a:g}'.replace('.', '')
        cache = CACHES / tag
        if not (cache / 'val').exists():
            print(f'--- кэш детекций alpha={a} ---', flush=True)
            subprocess.run([str(PY), '-u', 'scripts/build_full_val_cache.py',
                            '--cache', str(cache.relative_to(PROJECT_ROOT)),
                            '--imgsz', '960', '--limit', str(args.videos),
                            '--model', str(w.relative_to(PROJECT_ROOT))],
                           cwd=str(PROJECT_ROOT), check=False)
        print(f'--- лестница возраста alpha={a} ---', flush=True)
        subprocess.run([str(PY), '-u', 'scripts/birth_cost.py',
                        '--cache', str((cache / 'val').relative_to(PROJECT_ROOT)),
                        '--suffix', f'_alpha_{tag}'],
                       cwd=str(PROJECT_ROOT), check=False)

    print('\n[OK ] прогоны и замеры завершены; сводка — build_alpha_table.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
