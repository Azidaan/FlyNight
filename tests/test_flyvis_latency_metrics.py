import numpy as np
from tracking.flyvis_latency_metrics import response_metrics
from scripts.flyvis_y1_latency_20261104 import positions, frames_for


def test_immediate_and_delayed_reversal_have_correct_frame_alignment():
    f = np.zeros((79, 2));f[31:55, 0] = 2;f[55:, 0] = -2
    assert response_metrics(f, 0, 2, 'reverse')['latency_frames'] == 0
    f[55:57, 0] = 2
    assert response_metrics(f, 0, 2, 'reverse')['latency_frames'] == 2
    assert response_metrics(f * .25, 0, 2, 'reverse')['latency_frames'] == 2


def test_stop_settling_requires_three_frames_and_weak_signal_is_unidentified():
    f = np.zeros((79, 2));f[31:55, 0] = 2
    f[55:59, 0] = [1, 0, .8, .2]
    assert response_metrics(f, 0, 2, 'stop')['latency_frames'] == 3
    assert not response_metrics(np.zeros_like(f), 0, 2, 'stop')['eligible']


def test_integer_translation_and_stop_reuse_exact_pixels_without_padding():
    gray = (np.arange(720 * 1280).reshape(720, 1280) % 251).astype(np.uint8)
    case = dict(axis=0, speed=4, event='stop', contrast=1.)
    pos, vel = positions(case);frames = frames_for(gray, case)
    assert np.array_equal(np.diff(pos, axis=0), vel[1:])
    assert pos[54, 0] == 96 and pos[-1, 0] == 96
    assert np.array_equal(frames[55], frames[54])
    assert np.array_equal(frames[31, :, 4:], frames[30, :, :-4])


def test_reversal_returns_to_initial_image_and_input_has_no_future_frames():
    gray = np.random.default_rng(2).integers(0, 256, (720, 1280), dtype=np.uint8)
    case = dict(axis=1, speed=4, event='reverse', contrast=.25)
    frames = frames_for(gray, case)
    assert np.array_equal(frames[0], frames[-1])
    stop = frames_for(gray, dict(case, event='stop'))
    assert np.array_equal(frames[:55], stop[:55])
