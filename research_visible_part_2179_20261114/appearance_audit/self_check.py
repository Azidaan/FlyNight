"""Analytic orthogonal-vector fixture for the independent scoring audit."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for key in ['TEMP','TMP','TMPDIR']:os.environ[key]=str(ROOT/'.cache/tmp')
os.environ['CUDA_VISIBLE_DEVICES']=''
fixture=HERE/'synthetic_fixture_v2';fixture.mkdir(exist_ok=False)
records=[];vectors={}
for fill in ['imagenet_mean','black']:
    for subset in range(32):
        for role,frame in [('target_ref',1159),('competitor_ref',1164),('query',1165)]:
            key=f'{fill}_{subset}_{role}'
            vec=np.zeros(256,dtype=np.float32)
            axis=1 if role=='target_ref' or (role=='competitor_ref' and not subset&1) else 0
            vec[axis]=1
            vectors[key]=vec
            records.append(dict(key=key,frame=frame,subset=subset,fill=fill,role=role))
meta=fixture/'metadata.json';features=fixture/'features.npz';output=fixture/'audit.json'
meta.write_text(json.dumps(dict(records=records)),encoding='utf8')
np.savez_compressed(features,**vectors)
r=subprocess.run([sys.executable,str(HERE/'evaluate_pixels.py'),'--metadata',str(meta),'--features',str(features),'--output',str(output)],capture_output=True,text=True)
assert r.returncode==0,r.stderr
data=json.loads(output.read_text(encoding='utf8'))
assert data['passed'] and data['records_checked']==192 and data['scores_computed']==320
for entry in data['factorial']:
    changes=entry['mode'] in ['joint','references_only','competitor_reference_only']
    assert entry['full_gap']==.5 and entry['empty_gap']==(0 if changes else .5)
    for component in entry['components']:
        expected=.5 if component['bit']==1 and changes else 0
        assert component['full_context_deletion_effect']==expected
        assert component['mean_conditional_effect']==expected
        assert abs(component['shapley_effect']-expected)<1e-12
bad=fixture/'bad_norm.npz';badout=fixture/'bad_norm_result.json';first=records[0]['key'];vectors[first]=2*vectors[first];np.savez_compressed(bad,**vectors)
negative=subprocess.run([sys.executable,str(HERE/'evaluate_pixels.py'),'--metadata',str(meta),'--features',str(bad),'--output',str(badout)],capture_output=True,text=True)
assert negative.returncode!=0 and not badout.exists()
with (HERE/'self_check_v2.json').open('x',encoding='utf8') as f:json.dump(dict(passed=True,analytic_fixture='Two orthogonal unit vectors. Red alone changes competitor reference from orthogonal to identical query. Other pixels have no effect.',expected_joint_red_effect=.5,other_effects=0,invalid_norm_rejected=True,science_data_used=False),f,indent=2)
print('SCALAR_AUDIT_SELF_CHECK_PASSED')
