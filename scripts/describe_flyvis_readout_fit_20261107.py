"""One post-outcome descriptive fit/dev/eval check, with no fitting or selection."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.flyvis_readout_comparison_20261107 import OUT,read,sha,write
from scripts.audit_flyvis_readout_comparison_20261107 import predict,credits
if __name__=='__main__':
    a=read(OUT/'independent_audit.json');assert a['passed'];selected=read(OUT/'selection.json')
    rows={k:{} for k in ['baseline',*selected['best_per_family']]}
    for split in ['fit84','fit20','evaluate26']:
        with np.load(OUT/('data_'+split+'.npz')) as z:g=z['g'];y=z['y']
        predictions={'baseline':g.sum(-1)}
        for kind,config in selected['best_per_family'].items():
            values=[]
            for key in config['keys']:
                file=OUT/'models'/(key+'.npz');assert sha(file)==selected['manifest'][key]['sha256']
                with np.load(file) as z:params={k:z[k] for k in z.files}
                values.append(predict(kind,g,params))
            predictions[kind]=np.mean(values,0)
        for kind,scores in predictions.items():
            margin=scores[:,0]-scores[:,1]
            rows[kind][split]=dict(accuracy=float(credits(scores,y).mean()),bce=float(np.mean(np.logaddexp(0,margin)-y*margin)))
    write(OUT/'descriptive_fit.json',dict(post_outcome=True,no_fitting_or_selection=True,conditions=rows,selection_sha256=sha(OUT/'selection.json'),audit_sha256=sha(OUT/'independent_audit.json')))
    print(rows)
