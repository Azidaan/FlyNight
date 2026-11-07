"""Проверить, что визуальный экспорт не изменил числовой контроль и метки."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN,read,write,sha,matching

def main():
    folder=RUN/'appearance_features';report=read(folder/'report.json');controller=read(folder/'controller_result.json')
    assert report['complete'] and controller['exit_code']==0 and controller['all_owned_ended']
    assert not any(matching(p,b) for p,b in controller['owned'].items())
    expected={(x['split'],x['video']) for x in read(RUN/'admission_dataset/report.json')['videos']}
    assert expected=={(x['split'],x['video']) for x in report['videos']} and len(expected)==165
    total=0
    for item in report['videos']:
        p=folder/item['split']/(item['video']+'.npz');source=RUN/'admission_dataset'/item['split']/item['video']/'features.npz'
        assert sha(p)==item['output_sha256'] and sha(source)==item['source_feature_sha256']
        original=np.load(source);encoded=np.load(p)
        assert np.array_equal(original['X'],encoded['X']) and np.array_equal(original['y'],encoded['y'])
        assert encoded['E'].shape==(len(original['y']),2,report['feature_dim']) and np.isfinite(encoded['E']).all()
        total+=len(original['y'])
    assert report['batchnorm_buffers_unchanged'] and not report['domain_adaptation_batchnorm']
    write(folder/'audit.json',dict(passed=True,videos=165,candidates=total,numeric_features_exact=True,labels_exact=True,
        embeddings_finite=True,source_hashes_verified=True,controller_all_owned_ended=True,report_sha256=sha(folder/'report.json'),source_sha256=sha(__file__)))
    print(json.dumps(dict(passed=True,videos=165,candidates=total)))

if __name__=='__main__':main()
