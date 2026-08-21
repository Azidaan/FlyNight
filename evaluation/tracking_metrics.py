"""Метрики трекинга через официальные bdd100k eval tools (Этап 3, Часть 2.6).

Конвейер:
1. run_tracker_on_videos — прогон ByteTrack (встроенный в Ultralytics) по
   каждому видео поднабора; результаты пишутся в scalabel-формате BDD
   (по одному JSON на видео) — конвертация для eval tools не нужна.
2. evaluate_tracking_bdd — вызов `python -m bdd100k.eval.run mot` с GT
   из labels/box_track_20 (копируются только видео поднабора) и результатами.
   Возвращает HOTA / IDF1 / MOTA / ID switches.

Категории результатов — имена классов BDD (CLASS_NAMES), category_id как в GT.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.bdd100k_loader import CLASS_NAMES  # noqa: E402

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
VENV_PY = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'


def run_tracker_on_videos(
    model_path: str,
    video_ids: list[str],
    out_dir: str | Path,
    split: str = 'val',
    tracker: str = 'bytetrack.yaml',
) -> Path:
    """Прогоняет трекер по видео; пишет <out_dir>/<video>.json в scalabel-формате."""
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = DATA / 'images' / 'track' / split

    for i, vid in enumerate(video_ids, 1):
        out_json = out_dir / f'{vid}.json'
        if out_json.exists():
            continue
        model = YOLO(model_path)  # свежий экземпляр = чистое состояние трекера
        frames_dir = img_root / vid
        results = model.track(source=str(frames_dir), tracker=tracker,
                              stream=True, verbose=False, persist=True)
        scalabel_frames = []
        for frame_idx, (res, frame_path) in enumerate(
                zip(results, sorted(frames_dir.glob('*.jpg')))):
            labels = []
            boxes = res.boxes
            if boxes is not None and boxes.id is not None:
                for k in range(len(boxes)):
                    x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[k])
                    labels.append({
                        'id': str(int(boxes.id[k])),
                        'category': CLASS_NAMES[int(boxes.cls[k])],
                        'score': float(boxes.conf[k]),
                        'box2d': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    })
            scalabel_frames.append({
                'videoName': vid, 'name': frame_path.name,
                'frameIndex': frame_idx, 'labels': labels,
            })
        out_json.write_text(json.dumps(scalabel_frames), encoding='utf-8')
        print(f'[INFO] трекинг {i}/{len(video_ids)}: {vid} '
              f'({len(scalabel_frames)} кадров)', flush=True)
    return out_dir


def evaluate_tracking_bdd(
    result_dir: str | Path,
    video_ids: list[str],
    out_json: str | Path,
    split: str = 'val',
) -> dict:
    """Запускает официальный скрипт оценки MOT BDD100K на поднаборе видео.

    GT-файлы нужных видео копируются во временную папку, чтобы оценка шла
    только по поднабору. Возвращает сводные метрики (HOTA/IDF1/MOTA и др.).
    """
    result_dir, out_json = Path(result_dir), Path(out_json)
    gt_src = DATA / 'labels' / 'box_track_20' / split

    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        gt_dir = Path(tmp) / 'gt'
        res_dir = Path(tmp) / 'res'
        gt_dir.mkdir(); res_dir.mkdir()
        for vid in video_ids:
            shutil.copy2(gt_src / f'{vid}.json', gt_dir / f'{vid}.json')
            src = result_dir / f'{vid}.json'
            if not src.exists():
                raise FileNotFoundError(f'Нет результата трекинга: {src}')
            shutil.copy2(src, res_dir / f'{vid}.json')

        # Задача передаётся ключом --task, а не позиционно, и называется
        # box_track. Прежний вызов ('mot' позиционно) остался от старой
        # версии инструмента и падал с кодом 2 на разборе аргументов.
        # Через свою обёртку, а не напрямую: scalabel писался под pydantic 1
        # и с установленным pydantic 2 падает на собственных же данных, см.
        # evaluation/scalabel_compat.py. Задача передаётся ключом --task и
        # называется box_track; прежний вызов ('mot' позиционно) остался от
        # старой версии инструмента.
        cmd = [str(VENV_PY), str(PROJECT_ROOT / 'evaluation' / 'run_bdd_eval.py'),
               '--task', 'box_track',
               '--gt', str(gt_dir), '--result', str(res_dir),
               # В один процесс: штатная многопроцессность инструмента на
               # Windows зависает намертво — пусковая заглушка venv плюс
               # нестандартный главный модуль, дочерние процессы не стартуют,
               # а родитель их ждёт. Считается дольше, но считается.
               '--nproc', '1',
               '--out-file', str(out_json)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            raise RuntimeError(
                f'bdd100k.eval.run box_track завершился с кодом {proc.returncode}')

    def flat(v):
        """Метрика в одно число.

        Инструмент отдаёт список из трёх словарей: по классам, по группам
        классов и сводка с ключами AVERAGE (среднее по классам) и OVERALL
        (по всем объектам сразу). Берём OVERALL — он не даёт редким классам
        вроде поезда перевесить частые.
        """
        if isinstance(v, list) and v and isinstance(v[-1], dict):
            return v[-1].get('OVERALL', v[-1].get('AVERAGE'))
        return v

    data = json.loads(Path(out_json).read_text(encoding='utf-8'))
    data = {k: flat(v) for k, v in data.items()}
    # сводка: в выходе bdd100k метрики лежат по категориям + OVERALL
    overall = data.get('OVERALL', data)
    # HOTA с разбивкой на DetA и AssA инструмент bdd100k не считает — он даёт
    # классические CLEAR MOT. Разбивка ошибок отвечает на тот же вопрос даже
    # прямее: пропуски (FN) — провал детекции, перебросы (IDSw) — провал
    # связывания, ложные (FP) — провал точности. MT/PT/ML — сколько треков
    # велось устойчиво, частично и было потеряно.
    keys = ('MOTA', 'MOTP', 'IDF1', 'FN', 'FP', 'IDSw', 'MT', 'PT', 'ML', 'FM')
    summary = {k: overall[k] for k in keys if k in overall}
    return summary or data
