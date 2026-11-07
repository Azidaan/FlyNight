"""Один канал глаза без лишней единичной оси; значения пикселей неизменны."""
from tracking.foveated_eyes import EyeBank as OriginalEyeBank


def canonical_eye_layout(value, count, columns):
    if value.shape[0] != count or value.numel() != count * columns:
        raise ValueError('Expected one receptor channel per requested eye')
    return value.reshape(count, 1, columns)


class CanonicalEyeBank(OriginalEyeBank):
    def _render(self, gray, crops):
        value = super()._render(gray, crops)
        return canonical_eye_layout(value, len(crops), len(self.points_local))
