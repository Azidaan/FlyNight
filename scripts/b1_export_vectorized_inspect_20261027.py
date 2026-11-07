"""Прочитать расхождения остановленного экспорта; проверки не обходятся."""
from pathlib import Path
from datetime import datetime, timezone
import json,sys,hashlib
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/b123_20261027/b1_base_export_vectorized'


def main():
    limit=json.loads((OUT/'execution_deadline.json').read_text())['deadline']
    assert datetime.now(timezone.utc)<datetime.fromisoformat(limit)
    from scripts.b1_npz_recovery_20261026 import install
    install()
    import numpy as np
    from tracking import b1_appearance
    from tracking.b1_appearance_fast import relative_features
    from scripts import b1_export_base_20261027 as worker
    b1_appearance.relative_features=relative_features
    worker.OUT=OUT/'inspection_outputs'
    service=dict(deadline=limit,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Repeat read-only reconstruction only to inspect assertion; no guard disabled or models fitted')
    (OUT/'inspection_protocol.json').write_text(json.dumps(service,indent=2)+'\n')
    def trace(frame,event,arg):
        if event=='exception' and frame.f_code.co_filename==worker.__file__ and 'built' in frame.f_locals:
            if arg[0] is AssertionError:
                x=frame.f_locals;built=x['built'];data=x['data'];expected=data['F1']
                missing=np.isnan(built)!=np.isnan(expected)
                ix=np.argwhere(missing)
                report=dict(missing_by_feature=missing.sum((0,1)).tolist(),
                    examples=[dict(i=int(i),uid=int(data['uid'][i]),candidate=int(k),feature=int(c),
                                   expected=None if np.isnan(expected[i,k,c]) else float(expected[i,k,c]),
                                   actual=None if np.isnan(built[i,k,c]) else float(built[i,k,c]),
                                   valid=bool(data['mask'][i,k])) for i,k,c in ix[:12]])
                (OUT/'inspection.json').write_text(json.dumps(report,indent=2)+'\n')
        return trace
    sys.settrace(trace)
    try:worker.main()
    finally:sys.settrace(None)


if __name__=='__main__':main()
