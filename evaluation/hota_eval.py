"""HOTA / AssA / DetA для результатов трекинга через TrackEval.

Зачем отдельный модуль. Штатный конвейер оценки (evaluation/tracking_metrics.py,
пакеты bdd100k 1.0.1 и scalabel 0.3.1) считает метрики через py-motmetrics и
отдаёт только IDF1 / MOTA / MOTP / IDSw — поля HOTA, DetA и AssA там пустые, и
это ограничение пакетов, а не настройки. Между тем именно HOTA разделяет вклад
обнаружения (DetA) и связывания (AssA), а вся живая линия работы — про
связывание. Считать прирост связывания по IDF1 можно, но AssA заявлена в плане.

Здесь используется официальная реализация TrackEval (пакет trackeval, версия
1.3.0) с её же классом набора данных BDD100K — то есть метрика считается тем
кодом, на который ссылаются авторы HOTA, а не собственным пересчётом.

Формат. TrackEval ждёт покадровые JSON с ключом `index`, а разметка BDD100K и
наш вывод трекера используют `frameIndex`; кроме того TrackEval читает атрибут
`Crowd` с большой буквы, а в разметке он `crowd`. Поэтому оба набора файлов
переписываются во временную папку с нужными именами полей. Сами рамки, классы и
идентификаторы не меняются.

Использование:
    from evaluation.hota_eval import evaluate_hota
    m = evaluate_hota('runs/gate_a/baseline', video_ids)

    .venv/Scripts/python.exe scripts/eval_hota.py --dir runs/gate_a/baseline
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / 'data' / 'bdd100k'
# классы, которые понимает класс BDD100K из TrackEval
VALID = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'train',
         'motorcycle', 'bicycle')


def _convert(frames: list, is_gt: bool) -> list:
    """Привести покадровый список к тому виду, который читает TrackEval."""
    out = []
    for fr in frames:
        labels = []
        for l in fr.get('labels') or []:
            if l.get('category') not in VALID:
                continue                      # прочие классы TrackEval не знает
            item = {'id': l['id'], 'category': l['category'],
                    'box2d': l['box2d']}
            if is_gt:
                a = l.get('attributes') or {}
                item['attributes'] = {'Crowd': bool(a.get('crowd', False))}
            else:
                item['score'] = float(l.get('score', 1.0))
            labels.append(item)
        out.append({'name': fr.get('name', ''),
                    'videoName': fr.get('videoName', ''),
                    'index': fr.get('index', fr.get('frameIndex')),
                    'labels': labels})
    out.sort(key=lambda x: x['index'])
    return out


def evaluate_hota(result_dir: str | Path, video_ids: list[str],
                  split: str = 'val', tracker_name: str = 'run',
                  quiet: bool = True, details_path: str | Path | None = None) -> dict:
    """HOTA/DetA/AssA/IDF1/MOTA/IDSw для каталога scalabel-результатов.

    Возвращает и усреднение по классам (как mHOTA в статьях по BDD100K), и
    усреднение по объектам — их нельзя путать между собой при сравнении с
    чужими работами.
    """
    import trackeval

    result_dir = Path(result_dir)
    gt_src = DATA / 'labels' / 'box_track_20' / split

    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        tmp = Path(tmp)
        gt_dir = tmp / 'gt'
        trk_dir = tmp / 'trackers' / tracker_name / 'data'
        gt_dir.mkdir(parents=True)
        trk_dir.mkdir(parents=True)

        for vid in video_ids:
            gt = json.loads((gt_src / f'{vid}.json').read_text(encoding='utf-8'))
            res_path = result_dir / f'{vid}.json'
            if not res_path.exists():
                raise FileNotFoundError(f'нет результата трекинга: {res_path}')
            res = json.loads(res_path.read_text(encoding='utf-8'))
            if len(res) != len(gt):
                raise ValueError(f'{vid}: кадров в результате {len(res)}, '
                                 f'в разметке {len(gt)} — TrackEval требует совпадения')
            (gt_dir / f'{vid}.json').write_text(
                json.dumps(_convert(gt, True)), encoding='utf-8')
            (trk_dir / f'{vid}.json').write_text(
                json.dumps(_convert(res, False)), encoding='utf-8')

        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({'USE_PARALLEL': False, 'PRINT_RESULTS': not quiet,
                         'PRINT_CONFIG': False, 'TIME_PROGRESS': False,
                         'OUTPUT_SUMMARY': False, 'OUTPUT_DETAILED': False,
                         'PLOT_CURVES': False, 'PRINT_ONLY_COMBINED': True})
        ds_cfg = trackeval.datasets.BDD100K.get_default_dataset_config()
        ds_cfg.update({'GT_FOLDER': str(gt_dir),
                       'TRACKERS_FOLDER': str(tmp / 'trackers'),
                       'TRACKERS_TO_EVAL': [tracker_name],
                       'OUTPUT_FOLDER': str(tmp / 'out'),
                       'PRINT_CONFIG': False})
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                   trackeval.metrics.Identity()]
        res, _ = trackeval.Evaluator(eval_cfg).evaluate(
            [trackeval.datasets.BDD100K(ds_cfg)], metrics)

    sequence_results = res['BDD100K'][tracker_name]
    if details_path is not None:
        # Сохраняем компоненты по видео для корректного повторного
        # агрегирования метрик, а не усреднения готовых процентов HOTA.
        import gzip
        details_path = Path(details_path)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if details_path.suffix == '.gz' else open
        with opener(details_path, 'wt', encoding='utf-8') as fh:
            json.dump({k: v for k, v in sequence_results.items() if k != 'COMBINED_SEQ'},
                      fh, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x))
    combined = sequence_results['COMBINED_SEQ']
    out = {}
    # cls_comb_cls_av — среднее по классам (аналог mHOTA/mIDF1 в статьях),
    # cls_comb_det_av — среднее по объектам (ближе к OVERALL нашего конвейера)
    for key, suffix in (('cls_comb_cls_av', '_cls_av'), ('cls_comb_det_av', '')):
        if key not in combined:
            continue
        block = combined[key]
        h = block.get('HOTA', {})
        for name in ('HOTA', 'DetA', 'AssA', 'LocA'):
            v = h.get(name)
            if v is not None:
                out[name + suffix] = float(v.mean() * 100)
        c = block.get('CLEAR', {})
        if not suffix:
            for name in ('CLR_TP', 'CLR_FP', 'CLR_FN', 'Frag'):
                if name in c:
                    out[name] = int(c[name])
        for src, dst in (('MOTA', 'MOTA'), ('MOTP', 'MOTP'), ('IDSW', 'IDSw')):
            if src in c:
                out[dst + suffix] = float(c[src]) * (100 if dst != 'IDSw' else 1)
        i = block.get('Identity', {})
        if 'IDF1' in i:
            out['IDF1' + suffix] = float(i['IDF1']) * 100
    return out
