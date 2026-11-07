"""Диагностика доступных добавлений с недоступной модели разметкой личности."""
from collections import Counter
from scipy.optimize import linear_sum_assignment
import numpy as np
from tracking.recovery_detection_support import box,iou


class PastIdentityAdmissionOracle:
    """История истинной личности обновляется только по исходной выдаче базы.

    Это инструмент анализа с ground truth, а не допустимый вход трекера.
    Политика допуска не оптимизирует глобальную HOTA и не является потолком.
    """
    def __init__(self,threshold=.5):self.threshold=threshold;self.history={};self.counts=Counter()

    def step(self,base,candidates,truth):
        truths=[g for g in truth if not (g.get('attributes') or {}).get('crowd',False)]
        pairs=[]
        if base and truths:
            scores=np.array([[iou(box(b),box(g)) if b['category']==g['category'] else 0. for g in truths] for b in base])
            rows,cols=linear_sum_assignment(-scores)
            pairs=[(base[r],truths[c]) for r,c in zip(rows,cols) if scores[r,c]>=self.threshold]
        occupied={(g['category'],str(g['id'])) for _,g in pairs};by_gt={(g['category'],str(g['id'])):g for g in truths};choices=[]
        for candidate in candidates:
            self.counts['candidates']+=1;key=(candidate['category'],str(candidate['id']));target=self.history.get(key)
            if target is None:self.counts['no_previous_identity']+=1;continue
            if target not in by_gt:self.counts['target_absent']+=1;continue
            if target in occupied:self.counts['target_covered_by_base']+=1;continue
            overlap=iou(box(candidate),box(by_gt[target]))
            if overlap<self.threshold:self.counts['wrong_geometry_or_identity']+=1;continue
            choices.append((overlap,str(candidate['id']),target,candidate))
        accepted=[];used=set()
        for overlap,ident,target,candidate in sorted(choices,key=lambda x:(-x[0],x[1])):
            if target in used:self.counts['duplicate_true_object']+=1;continue
            accepted.append(candidate);used.add(target);self.counts['admitted']+=1
        # Ни текущая разметка, ни собственный допуск не создают прошлое наблюдение.
        for b,g in pairs:self.history[(b['category'],str(b['id']))]=(g['category'],str(g['id']))
        return sorted(accepted,key=lambda x:str(x['id']))
