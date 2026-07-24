"""Наложение карты качества Q на кадр (Этап 10, подэтап 10.3).

Colormap: красный — низкое качество, зелёный — высокое. По плану Б Этапа 10
карта нормализуется min-max внутри кадра (иначе на слабо-контрастных Q
картинка выглядит однородной и нечитаема на слайдах).

Использование как модуль: overlay_q_map_on_frame(frame_bgr, q_map, stride).
Как скрипт — сохраняет наложения для репрезентативных кадров:
    python visualization/q_map_overlay.py --model weights/drb_aug_full_seed42/best.pt \
        --frames data/bdd100k/images/track/val/<vid>/<frame>.jpg ...
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def overlay_q_map_on_frame(frame: np.ndarray, q_map: np.ndarray,
                           input_shape: tuple[int, int],
                           alpha: float = 0.4,
                           normalize: bool = True) -> np.ndarray:
    """Полупрозрачная Q поверх кадра (BGR). q_map — (H_q, W_q) в [0,1].

    input_shape — (H, W) letterbox-входа сети: Q сначала растягивается на
    него, затем срезается паддинг и растягивается на кадр (обратный letterbox).
    """
    oh, ow = frame.shape[:2]
    ih, iw = input_shape
    q = q_map.astype(np.float32)
    if normalize:
        q_min, q_max = float(q.min()), float(q.max())
        if q_max - q_min > 1e-6:
            q = (q - q_min) / (q_max - q_min)

    q_input = cv2.resize(q, (iw, ih), interpolation=cv2.INTER_LINEAR)
    gain = min(ih / oh, iw / ow)
    pad_x = int(round((iw - ow * gain) / 2))
    pad_y = int(round((ih - oh * gain) / 2))
    q_img = q_input[pad_y:ih - pad_y or None, pad_x:iw - pad_x or None]
    q_full = cv2.resize(q_img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap((255 * (1 - q_full)).astype(np.uint8),
                             cv2.COLORMAP_RdYlGn_r if hasattr(cv2, 'COLORMAP_RdYlGn_r')
                             else cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heat, alpha, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--frames', nargs='+', required=True)
    ap.add_argument('--stride', type=int, default=8,
                    help='уровень карты Q (8=P3, 16=P4, 32=P5)')
    ap.add_argument('--out-dir', default=str(PROJECT_ROOT / 'figs' / 'final' / 'q_maps_examples'))
    args = ap.parse_args()

    from tracking.inference_with_q import QInference
    qi = QInference(args.model)
    if not qi.with_drb:
        print('[FAIL] модель без DRB — карты Q отсутствуют')
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fp in args.frames:
        fp = Path(fp)
        qi.predict_frame(fp)
        q_maps = qi.q_maps_by_stride()
        q = q_maps[args.stride][0, 0].cpu().numpy()
        s0, q0 = next(iter(q_maps.items()))
        input_shape = (q0.shape[-2] * s0, q0.shape[-1] * s0)
        frame = cv2.imread(str(fp))
        overlaid = overlay_q_map_on_frame(frame, q, input_shape)
        out = out_dir / f'{fp.stem}_qP{args.stride}.jpg'
        cv2.imwrite(str(out), overlaid)
        print(f'[OK ] {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
