"""Exact old/new replay of real saved heads, without training or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tracking import learned_recovery_admission as current_admission
from tracking import reliability_head as current_reliability
torch.set_num_threads(2)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def original(filename):
    spec=importlib.util.spec_from_file_location('original_'+Path(filename).stem,OUT/'original'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
old_admission=original('learned_recovery_admission.py')
old_reliability=original('reliability_head.py')
before=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
for name,expected in before['source_sha256'].items():assert sha(OUT/'original'/Path(name).name)==expected

run=ROOT/'runs/recovery_research_20261027/admission_learning'
registered=json.loads((run/'report.json').read_text(encoding='utf-8'))
v=registered['video_rows']['dev'][0]
source=ROOT/'runs/recovery_research_20261027/appearance_features/dev'/(v['video']+'.npz')
assert sha(source)==registered['source_features_sha256'][source.relative_to(ROOT).as_posix()]
with np.load(source) as z:X=z['X'][:64].copy();E=z['E'][:64].copy()
admission=[];arrays={}
for item in registered['models']:
    cp=run/item['checkpoint'];norm=run/item['normalization']
    assert sha(cp)==item['checkpoint_sha256'] and sha(norm)==item['normalization_sha256']
    old=old_admission.AdmissionModel(cp,norm,'cpu')
    rng=torch.get_rng_state().clone()
    current=current_admission.AdmissionModel(cp,norm,'cpu')
    assert torch.equal(rng,torch.get_rng_state())
    a=old.probabilities(X,E);b=current.probabilities(X,E)
    assert np.array_equal(a,b)
    ref=np.load(run/f"{item['arm']}_seed{item['seed']}_dev_probabilities.npy",mmap_mode='r')[v['start']:v['start']+len(X)]
    error=float(np.max(np.abs(ref-b)))
    assert error==next(r['cpu_vs_saved_max_abs'] for r in before['actual_admission'] if r['arm']==item['arm'] and r['seed']==item['seed'])
    key=f"{item['arm']}_{item['seed']}";arrays[key+'_original']=a;arrays[key+'_current']=b;arrays[key+'_saved']=np.asarray(ref)
    admission.append(dict(arm=item['arm'],seed=item['seed'],rows=len(X),old_new_exact=True,cpu_vs_saved_max_abs=error,
                          checkpoint_sha256=sha(cp),normalization_sha256=sha(norm)))

reliability=[]
rng=np.random.default_rng(81)
pred=np.column_stack([rng.uniform(0,1280,64),rng.uniform(0,720,64),rng.uniform(10,200,64),rng.uniform(10,200,64)])
dis=np.column_stack([pred[:,:2]-pred[:,2:]/2,pred[:,:2]+pred[:,2:]/2])
dis[::7]=np.nan
xf=rng.normal(size=(64,len(current_reliability.FLY_FEATURES)))
class Lookup:
    def query(self,*_):return xf,np.arange(len(xf))%2==0
for entry in before['actual_reliability']:
    path=ROOT/entry['path'];assert sha(path)==entry['sha256']
    spec=json.loads(path.read_text(encoding='utf-8'))
    for branch in ['base','fly']:
        old=old_reliability.LogisticHead(spec[branch]);new=current_reliability.LogisticHead(spec[branch])
        x=rng.normal(size=(64,len(spec[branch]['features'])))
        assert np.array_equal(old(x),new(x))
    old=old_reliability.ReliabilityFunction(old_reliability.LogisticHead(spec['base']),old_reliability.LogisticHead(spec['fly']),Lookup())
    new=current_reliability.ReliabilityFunction(current_reliability.LogisticHead(spec['base']),current_reliability.LogisticHead(spec['fly']),Lookup())
    a=old(pred,dis,11);b=new(pred,dis,11)
    assert np.array_equal(a,b)
    key=path.stem;arrays[key+'_original']=a;arrays[key+'_current']=b
    reliability.append(dict(path=entry['path'],sha256=sha(path),head_and_wrapper_old_new_exact=True,rows=len(pred)))
np.savez_compressed(OUT/'replayed_outputs.npz',**arrays)
assert not torch.cuda.is_initialized()
report=dict(admission=admission,reliability=reliability,source_sha256={name:sha(ROOT/name) for name in [
    'tracking/learned_recovery_admission.py','tracking/reliability_head.py','tracking/flyvis_checkpoint.py',
    'tests/test_learned_head_contracts.py']},outputs_sha256=sha(OUT/'replayed_outputs.npz'),script_sha256=sha(__file__),
    historical_files_unchanged=True,cuda_initialized=False,boundary='64-row CPU compatibility replay; no retraining or new tracker quality measurement. Correct historical weights are not declared malformed.')
(OUT/'after_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
