"""Smoke-тест LQA-guided трекера (Этап 9, п. 9.3-9.4).

Юнит: детекция с conf=0.45 при τ_base=0.5, k=0.2:
  q=1.0 → τ_i=0.5  → 0.45 < 0.5  → второй проход (слабая);
  q=0.0 → τ_i=0.3  → 0.45 >= 0.3 → первый проход (надёжная).
Разделение проверяется напрямую через _split_detections.

Интеграция: q-кэш одного видео (yolo26s + случайный DRB) → ассоциация
LQAGuidedByteTrack на кэше → корректный scalabel-вывод; сравнение с
обычным ByteTrack на тех же детекциях (треки должны отличаться —
адаптация реально меняет разбиение).

Запуск: python scripts/smoke/test_lqa_guided_tracker.py [video_id]
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.lqa_guided_tracker import LQAGuidedByteTrack, _DetSliceQ  # noqa: E402
from tracking.offline_association import make_tracker_args  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 'b1cebfb7-284f5117'
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = '') -> None:
    print(f'[{"OK " if cond else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not cond:
        FAILURES.append(name)


def unit_split() -> None:
    args = make_tracker_args(track_high_thresh=0.5)
    args.lqa_k = 0.2
    tracker = LQAGuidedByteTrack(args)

    xyxy = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], np.float32)
    conf = np.array([0.45, 0.45], np.float32)
    cls = np.zeros(2, np.int16)
    q = np.array([1.0, 0.0], np.float32)   # чистая и деградированная области
    dets = _DetSliceQ(xyxy, conf, cls, q)

    high, low, m_high, m_low = tracker._split_detections(dets)
    check('q=1.0 (τ=0.5): conf 0.45 -> второй проход', not m_high[0] and m_low[0])
    check('q=0.0 (τ=0.3): conf 0.45 -> первый проход', bool(m_high[1]))
    check('размеры разбиения', len(high) == 1 and len(low) == 1)

    # Полный update: из первого прохода должен родиться новый трек
    # (порог инициации для q=0 тоже снижен: 0.5-0.2=0.3 < 0.45).
    # Принадлежность проверяем по координатам: столбец idx (7) хранит
    # индекс ВНУТРИ первого прохода (parse_bboxes нумерует после маски),
    # а не в исходном кадре — ловушка семантики ultralytics.
    out = tracker.update(dets)
    is_q0_box = len(out) == 1 and abs(float(out[0][0]) - 100.0) < 1.0
    check('update: новый трек только из детекции с q=0 (рамка x1=100)',
          is_q0_box,
          f'треков={len(out)}'
          + (f', x1={out[0][0]:.1f}' if len(out) else ''))

    # Обычный ByteTrack при том же пороге 0.5 не активирует ни одного трека.
    from ultralytics.trackers.byte_tracker import BYTETracker
    plain = BYTETracker(make_tracker_args(track_high_thresh=0.5))
    from tracking.offline_association import _DetSlice
    out_plain = plain.update(_DetSlice(xyxy, conf, cls))
    check('plain ByteTrack (τ=0.5): нет треков', len(out_plain) == 0,
          f'треков={len(out_plain)}')


def integration(tmp: Path) -> None:
    from models.yolo_neck import insert_drb
    from tracking.inference_with_q import QInference, build_detection_cache_with_q
    from tracking.offline_association import run_association

    # Собираем q-кэш случайно-инициализированной DRB-моделью: QInference
    # создаётся из pretrained, DRB вставляется до кэширования.
    import tracking.inference_with_q as iq
    model_path = str(PROJECT_ROOT / 'weights' / 'pretrained' / 'yolo26s.pt')
    orig_init = iq.QInference.__init__

    def patched_init(self, mp, **kw):
        orig_init(self, mp, **kw)
        insert_drb(self.det_model)
        self.with_drb = True
    iq.QInference.__init__ = patched_init
    try:
        cache_dir = tmp / 'qcache'
        build_detection_cache_with_q(model_path, [VIDEO], 'val', cache_dir,
                                     aggregations=('mean',))
    finally:
        iq.QInference.__init__ = orig_init

    t_args = make_tracker_args(track_high_thresh=0.5)
    t_args.lqa_k = 0.2
    lqa_dir = run_association(cache_dir, [VIDEO], tmp / 'lqa', 'val',
                              args=t_args, tracker_cls=LQAGuidedByteTrack,
                              q_key='q_mean')
    plain_dir = run_association(cache_dir, [VIDEO], tmp / 'plain', 'val',
                                args=make_tracker_args(track_high_thresh=0.5))

    import json
    lqa = json.loads((lqa_dir / f'{VIDEO}.json').read_text('utf-8'))
    plain = json.loads((plain_dir / f'{VIDEO}.json').read_text('utf-8'))
    n_lqa = sum(len(f['labels']) for f in lqa)
    n_plain = sum(len(f['labels']) for f in plain)
    check('интеграция: LQA-ассоциация выдала треки', n_lqa > 0, f'{n_lqa} рамок')
    check('интеграция: адаптация меняет результат vs plain',
          n_lqa != n_plain, f'lqa={n_lqa}, plain={n_plain}')


def main() -> int:
    unit_split()
    with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT / 'runs')) as tmp:
        integration(Path(tmp))
    print(f'\n--- {"OK: LQA-guided трекер работает" if not FAILURES else f"FAIL: {FAILURES}"} ---')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    raise SystemExit(main())
