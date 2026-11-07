"""Проверить полный экспорт, разделение, временные метки и повторное получение меток допуска."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha
from tracking.recovery_admission_oracle import PastIdentityAdmissionOracle
from tracking.recovery_admission_features import FEATURE_NAMES

RUN=BASE/'admission_dataset_audit'
def run():
    p=read(RUN/'protocol.json');folder=BASE/'admission_dataset';report=read(folder/'report.json');assert report['complete']
    controller=read(folder/'controller_result.json');assert controller['all_owned_ended'] and controller['exit_code']==0
    dataset=read(folder/'protocol.json');end=datetime.fromisoformat(read(RUN/'execution_deadline.json')['deadline']).timestamp()
    from evaluation.hota_eval import VALID
    assert not set(dataset['videos']['train']) & set(dataset['videos']['dev'])
    total=0;counts={-1:0,0:0,1:0};proof=[]
    for index,item in enumerate(report['videos']):
        if time.time()>end-3 or (RUN/'stop.request').exists():raise TimeoutError('Dataset audit budget')
        write(RUN/'status.json',dict(state='running',video=item['video'],index=index,updated=datetime.now(timezone.utc).isoformat()),required=False)
        split=item['split'];video=item['video'];path=folder/split/video
        assert sha(path/'features.npz')==item['features_sha256'] and sha(path/'records.json')==item['records_sha256']
        z=np.load(path/'features.npz');x,y=z['X'],z['y'];records=read(path/'records.json');assert len(records)==len(x)==len(y)==item['candidates']
        assert x.shape[1]==len(FEATURE_NAMES) and np.isfinite(x).all() and set(y.tolist())<={-1,0,1}
        images=set()
        for row,label,features in zip(records,y,x):
            assert row['past_frame']<row['frame'] and row['target']==int(label)
            assert features[FEATURE_NAMES.index('past_available')]==float(row['past_frame']>=0)
            images.update([row['past'],row['current']]);counts[int(label)]+=1
        assert all((path/name).is_file() for name in images)
        base=read(path/'baseline.json');supported=read(path/'supported.json');truth=read(ROOT/f'data/bdd100k/labels/box_track_20/train/{video}.json')
        assert len(base)==len(supported)==len(truth)
        by_ref={}
        for row in records:by_ref.setdefault(row['reference'],{})[row['id']]=row['target']
        oracle=PastIdentityAdmissionOracle();checked=0
        for ref,(b,s,g) in enumerate(zip(base,supported,truth)):
            assert s['labels'][:len(b['labels'])]==b['labels']
            candidates=s['labels'][len(b['labels']):]
            known={c['id']:(c['category'],str(c['id'])) in oracle.history for c in candidates}
            accepted={c['id'] for c in oracle.step(b['labels'],candidates,[q for q in g['labels'] if q['category'] in VALID])}
            for candidate in candidates:
                expected=-1 if not known[candidate['id']] else int(candidate['id'] in accepted)
                assert by_ref[ref][candidate['id']]==expected;checked+=1
        assert checked==len(records);total+=checked;proof.append(dict(video=video,split=split,candidates=checked,labels_replayed=True,strict_past=True))
    result=dict(passed=True,videos=len(proof),candidates=total,labels={str(k):v for k,v in counts.items()},
        train_dev_disjoint=True,labels_replayed_from_saved_outputs=True,unknown_not_negative=True,
        strict_past_all_records=True,finite_features=True,all_crop_files_present=True,proof=proof,
        report_sha256=sha(folder/'report.json'),source_sha256=sha(__file__),updated=datetime.now(timezone.utc).isoformat())
    write(folder/'audit.json',result);write(RUN/'report.json',result)
    write(RUN/'status.json',dict(state='completed',updated=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    try:run()
    except Exception as e:write(RUN/'status.json',dict(state='failed',error=repr(e),updated=datetime.now(timezone.utc).isoformat()));raise
