"""Тот же причинный масочный путь; изменена только точность вычислений."""
import torch
from tracking.sam2_mask_recovery import MaskOutputRecovery

class BFloat16MaskOutputRecovery(MaskOutputRecovery):
    def step(self,*args,**kwargs):
        # Контекст завершается на каждом шаге: кэш приведённых весов не
        # удерживает старые модели разных видео. Научные пороги неизменны.
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return super().step(*args,**kwargs)
