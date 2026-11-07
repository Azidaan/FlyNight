"""Run the Atlas CPU model on user-supplied images in a fresh, guarded process.

CLI: .venv/Scripts/python.exe scripts/flyvis_atlas_live_worker_20261107.py INPUT.json OUTPUT.json
API: analyze_grayscale([array_256x256_uint8, ...], kind="video", source_fps=30).
The caller receives plain JSON data and never imports Torch in its own process.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve()
MODEL = "flow/0000/000"
SIDE = 256
MAX_FRAMES = 12
BYTE_COUNT = SIDE * SIDE
ENCODED_LENGTH = 4*((BYTE_COUNT+2)//3)
CHECKPOINT = ROOT / "runs/flyvis/results/flow/0000/000/chkpts/chkpt_00000"
CHECKPOINT_SHA = "d0e42857e738d0315897c2d50fde9eb3fb1a3fb1f071d55dfc13d53d72bccc3f"
GUARD = ROOT / "runs/flyvis_repairs_20261106/unification_20261107"


def _validate_request(request):
    if not isinstance(request, dict) or set(request) - {"kind", "frames", "source_fps"}:
        raise ValueError("Ожидаются kind, frames и необязательная частота source_fps.")
    kind = request.get("kind")
    frames = request.get("frames")
    if kind not in ("image", "video") or not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("Ожидается изображение или последовательность длиной от 1 до 12 кадров.")
    if kind == "image" and len(frames) != 1:
        raise ValueError("Для изображения нужен один кадр.")
    fps = request.get("source_fps", 30)
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 120:
        raise ValueError("Частота кадров должна находиться в пределах 1–120 Гц.")
    decoded = []
    for encoded in frames:
        if not isinstance(encoded, str) or len(encoded) != ENCODED_LENGTH:
            raise ValueError("Размер кадра в base64 не соответствует 256×256.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Повреждена кодировка base64.") from exc
        if len(raw) != BYTE_COUNT:
            raise ValueError("В кадре должно быть 65 536 значений яркости.")
        decoded.append(raw)
    return kind, decoded, float(fps)


def _step(backbone, state, previous, current):
    """Same interpolation, clamping and Euler updates as FlyBackbone.step; retain only the last state."""
    import numpy as np
    t = backbone.torch
    net = backbone.net
    ix = t.as_tensor(np.asarray(net.stimulus.input_index), device=backbone.dev)
    buf = t.zeros((1, net.stimulus.n_nodes), device=backbone.dev, dtype=t.float32)
    alpha = t.arange(1, backbone.steps+1, dtype=t.float32, device=backbone.dev) / backbone.steps
    net.clamp()
    params = net._param_api()
    for fraction in alpha:
        signal = previous * (1-fraction) + current*fraction
        buf[:, ix] = signal.expand(1, ix.shape[0], -1)
        state = net._next_state(params, state, buf, backbone.dt)
    return state


def _worker(kind, frames, fps, guard):
    import cv2
    import numpy as np
    import torch
    from tracking.flyvis_activity import OUTPUT_CELL_TYPES
    from tracking.flyvis_motion import FlyVisMotionEstimator
    from tracking.foveated_eyes import EyeBank
    from tracking.z13_readout import FlyBackbone

    if torch.cuda.is_initialized():
        raise RuntimeError("До загрузки CPU-модели уже создан контекст CUDA.")
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Контрольная сумма весов модели не совпала.")
    # FlyVis chooses its import-time default by probing CUDA even on an explicit
    # CPU request. Select CPU in this private process, then restore the probe.
    from unittest.mock import patch
    with patch.object(torch.cuda, "is_available", return_value=False):
        model = FlyVisMotionEstimator(mode="full", model=MODEL, source_fps=fps,
                                      dt=0.02, fade_in_seconds=1., device="cpu")
    chosen = model.flyvis.NetworkView(MODEL).get_checkpoint("best")
    if Path(chosen).resolve() != CHECKPOINT:
        raise ValueError("Выбранная контрольная точка отличается от проверенной.")
    eye = EyeBank(model)
    bb = FlyBackbone(model, trainable=False)
    if any(p.device.type != "cpu" for p in bb.net.parameters()):
        raise RuntimeError("Вес модели размещён вне CPU.")
    eye_frames, map_frames = [], []
    state = previous = None
    with torch.inference_mode():
        for raw in frames:
            gray = np.frombuffer(raw, dtype=np.uint8).reshape(SIDE, SIDE)
            rendered = eye._render(gray, [(0, 0, SIDE, SIDE)])
            # Official BoxEye returns (batch, frames, 1, hexals); the historical
            # foveated runtime likewise selects the current time slice [:, 0].
            if rendered.shape != (1,1,1,721) or not torch.isfinite(rendered).all():
                raise ValueError("Некорректный ответ BoxEye.")
            hexals = rendered[:,0]
            state = bb.init(hexals) if state is None else _step(bb, state, previous, hexals)
            activity = state.nodes.activity[0, bb.index]
            if activity.shape != (34,721) or not torch.isfinite(activity).all():
                raise ValueError("Некорректная активность выходных клеток.")
            eye_frames.append(hexals[0,0].tolist())
            map_frames.append(activity.tolist())
            previous = hexals
    if torch.cuda.is_initialized():
        raise RuntimeError("Неожиданная инициализация CUDA.")
    return dict(kind=kind, model=MODEL, checkpoint_sha256=CHECKPOINT_SHA,
                channels=list(OUTPUT_CELL_TYPES), eye=eye_frames, maps=map_frames,
                source_fps=fps, dt=bb.dt, steps_per_frame=bb.steps, fade_in_seconds=1.,
                device="cpu", shape=[len(frames),34,721], unit="raw model activity",
                note="Реальный отклик модели на загруженную яркость. Первый кадр плавно вводится за одну модельную секунду. "
                     "Следующие кадры продолжают то же состояние. Между кадрами применяется линейная интерполяция. "
                     "На карте показана активность последнего шага. Пространственные карты архивных эпизодов "
                     "имеют другую шкалу и предварительную адаптацию яркости.")


def _publish_new(destination, data):
    """Atomically expose a complete output and fail if destination already exists."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="atlas-", suffix=".part",
                                         delete=False, mode="w", encoding="utf-8") as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, allow_nan=False, separators=(",",":"))
        os.link(tmp, destination)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def analyze_grayscale(frames, *, kind="image", source_fps=30):
    """Compute real CPU responses on one to twelve uint8 grayscale arrays using the project's interpreter."""
    import numpy as np
    encoded = []
    for frame in frames:
        if len(encoded) == MAX_FRAMES:
            raise ValueError("Максимум 12 кадров.")
        arr = np.asarray(frame)
        if arr.shape != (SIDE,SIDE) or arr.dtype != np.uint8:
            raise ValueError("Ожидается полутоновый кадр uint8 размером 256×256.")
        encoded.append(base64.b64encode(arr.tobytes(order="C")).decode("ascii"))
    request = dict(kind=kind, frames=encoded, source_fps=source_fps)
    _validate_request(request)
    with tempfile.TemporaryDirectory(prefix="flyvis-atlas-") as temp:
        inp, out = Path(temp)/"input.json", Path(temp)/"output.json"
        inp.write_text(json.dumps(request,separators=(",",":")),encoding="utf-8")
        cmd = [str(ROOT/".venv/Scripts/python.exe"),str(WORKER),str(inp),str(out)]
        proc = subprocess.run(cmd,cwd=ROOT,text=True,encoding="utf-8",errors="replace",
                              capture_output=True,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if proc.returncode:
            raise RuntimeError((proc.stderr or proc.stdout or str(proc.returncode))[-2500:])
        return json.loads(out.read_bytes())


def main():
    # Do not query or enumerate GPUs; enforce CPU on the runtime and tensors.
    for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(GUARD))
    from system_memory_guard import guard_entrypoint
    guard = guard_entrypoint(timeout=240)
    if len(sys.argv) != 3:
        raise SystemExit("Usage: flyvis_atlas_live_worker_20261107.py INPUT.json OUTPUT.json")
    path, out = map(Path, sys.argv[1:])
    if out.exists():
        raise FileExistsError(out)
    if path.stat().st_size > 1_100_000:
        raise ValueError("Вход превышает допустимый размер.")
    kind, frames, fps = _validate_request(json.loads(path.read_bytes()))
    _publish_new(out,_worker(kind,frames,fps,guard))


if __name__ == "__main__":
    main()
