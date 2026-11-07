"""Сохранить нулевое внешнее дополнение прежнего пакетного формата."""
from pathlib import Path
from datetime import datetime, timezone
import sys,json,os,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/b1_base_export_vectorized'
SERVICE=RUN/'recovery1'


def main(prepare):
    if prepare:
        SERVICE.mkdir(exist_ok=False)
        deadline=json.loads((RUN/'execution_deadline.json').read_text())['deadline']
        assert datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
        paths=[Path(__file__),ROOT/'tracking/b1_appearance.py',ROOT/'tracking/b1_appearance_fast.py',
               ROOT/'scripts/b1_export_base_20261027.py',RUN/'inspection.json',RUN/'protocol.json']
        p=dict(deadline=deadline,old_failure='controller_result.json',
            correction='Only outer padded tagged geometry8:17 restored to the original merge() zeros, using original feature chunk widths per split. Real features untouched. Original assertions retained.',
            scientific_settings_changed=False,sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths})
        (SERVICE/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
        return
    p=json.loads((SERVICE/'protocol.json').read_text())
    assert datetime.now(timezone.utc)<datetime.fromisoformat(p['deadline'])
    for name,digest in p['sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    from scripts.b1_npz_recovery_20261026 import install
    install()
    import numpy as np
    from tracking import b1_appearance
    from tracking.b1_appearance_fast import relative_features
    from scripts import b1_export_base_20261027 as worker
    base=ROOT/'runs/b1_real_inputs_20261026'
    config=json.loads((base/'protocol.json').read_text())
    with np.load(base/'base_inputs.npz') as z:uid,names=z['uid'],z['name']
    widths={}
    for split in ('fit84','fit20'):
        sizes=[]
        for path in (ROOT/f'runs/z13_region_assignment_20261025/recovery2/features/D/{split}').glob('chunk_*.npz'):
            with np.load(path) as z:sizes.append(z['mask'].shape[1])
        for video in config[split]:widths[video]=max(sizes)
    order=np.argsort(uid);counter=0
    original=b1_appearance.appearance_features
    b1_appearance.relative_features=relative_features
    def appearance(*args,**kwargs):
        nonlocal counter
        features,bank=original(*args,**kwargs)
        width=widths[str(names[order[counter]])];counter+=1
        assert not np.asarray(args[1])[width:].any()
        features[width:,8:17]=0
        return features,bank
    b1_appearance.appearance_features=appearance
    worker.OUT=SERVICE/'outputs'
    worker.main()
    assert counter==len(uid)


if __name__=='__main__':main('--prepare' in sys.argv)
