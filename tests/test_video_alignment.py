from __future__ import annotations

import cv2
import numpy as np

from tracking.video_alignment import align_reference_frames, compact_frame


def test_alignment_recovers_irregularly_sampled_video_frames() -> None:
    raw = []
    for index in range(80):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        frame[:] = (20 + index, 40, 80)
        cv2.rectangle(frame, (index * 3 % 260, 60), (index * 3 % 260 + 40, 110), (220, 180, 30), -1)
        raw.append(compact_frame(frame))
    expected = [0, 1, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62]
    references = []
    for index in expected:
        ok, encoded = cv2.imencode(".jpg", raw[index], [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        references.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
    matches = align_reference_frames(raw, references, search_radius=20)
    assert [match.video_frame_index for match in matches] == expected


def test_compact_frame_accepts_grayscale() -> None:
    result = compact_frame(np.zeros((20, 30), dtype=np.uint8), width=10, height=5)
    assert result.shape == (5, 10, 3)
