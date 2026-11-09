"""One fixed final fit on the previously registered91 historical pairs."""
from common import *
start('cpu')
import numpy as np
from tracking.channel_identity_head import geometry_transform,fit_ridge_logistic,decision_function
p=read(OUT/'execution_protocol.json');verify(p)
assert not (OUT/'deployment.json').exists()
rows=read(TRAIN/'features.json')['records'];assert len(rows)==91
geometry=np.array([r['geometry'][:7] for r in rows]);labels=np.array([r['label']=='own' for r in rows],int)
g,_,normalization=geometry_transform(geometry,geometry);models={};undefined={}
models['G7']=fit_ridge_logistic(g,labels)
for arm in ['C34','R134_spatial','R134_temporal']:
    valid=np.array([r['variants'][arm]['full']['valid'] for r in rows],bool)
    # Same fitting population rule as the frozen fold implementation.
    assert valid.all(),'Unexpected undefined training map: compare original implementation before proceeding'
    channels=np.array([r['variants'][arm]['full']['channels'] for r in rows])
    assert channels.shape==(91,34)
    models[arm]=fit_ridge_logistic(np.column_stack([g,channels]),labels);undefined[arm]=int((~valid).sum())
assert all(m['gradient_max']<=1e-7 for m in models.values())
write('deployment.json',dict(passed=True,geometry_normalization=normalization,models=models,
    pairs=91,videos=len({r['video'] for r in rows}),own=int(labels.sum()),foreign=int((1-labels).sum()),
    undefined=undefined,threshold=0,regularization=1,execution_sha256=sha(OUT/'execution_protocol.json'),
    no_validation_labels=True,no_hyperparameter_selection=True))
status('deployment_ready',models=4)
print('DEPLOYMENT',len(rows),{k:m['gradient_max'] for k,m in models.items()},flush=True)
