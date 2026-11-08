"""Independent alpha-LAP and policy replay; ready label/geometry audits reused."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_bbox_cost_pilot_20261108.common import *
from research_bbox_cost_pilot_20261108.audit_replay import replay


def main(obs):
    import numpy as np
    p=phase();prior=read(PRIOR_PILOT/'independent_audit.json');assert prior['comparison_sha256']==sha(PRIOR_PILOT/'comparison_report.json');report=read(RUN/'comparison_report.json');entries=read(INVENTORY/'protocol.json')['videos'];checks=Counter()
    for video,e in entries.items():
        if e['split']=='val20':continue
        data=read(INVENTORY/'native'/(video+'.json'))
        with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        labels=read(PRIOR_PILOT/'labels'/(video+'.json'))['rows']
        for arm,cfg in p['configs'].items():
            r=read(RUN/'outcomes'/arm/(video+'.json'));out,c=replay(native,data,dict(cap=cfg['cap'],threshold=.2,bidirectional=False),r['decisions'],cfg['alpha']);checks.update(c);assert hashlib.sha256(out.tobytes()).hexdigest()==r['outputs_sha256'];computed=Counter(N=sum(l['known'] for l in labels.values()))
            for d in r['decisions']:computed['revoked']+=len(d['revoked'])
            for l in r['links']:computed['links']+=1;computed[l['kind']]+=1
            computed['J']=computed['correct']-computed['foreign'];assert dict(computed)==r['totals']==report['results'][e['split']]['per_video'][video][arm];checks['outcome_ledgers']+=1
            if cfg['alpha']==.2:checks['ready_alpha_point2_outcomes_reused']+=1
        status('bounded_cost_independent_audit',video=video,checks=dict(checks),guard=obs.read())
    fit=report['results']['fit20']['totals'];eligible=[a for a in p['configs'] if fit[a].get('foreign',0)==0];chosen=max(eligible,key=lambda a:(fit[a].get('correct',0),-p['configs'][a]['cap'],-p['configs'][a]['alpha'])) if eligible else 'B0';assert chosen==report['selected']==read(RUN/'selection.json')['selected'];assert sha(RUN/'selection.json')==report['selection_sha256'];names=list(report['results']['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)))
    for arm in p['configs']:
        rows=[report['results']['evaluate26']['per_video'][v][arm] for v in names];d=np.array([r['J'] for r in rows]);n=np.array([r['N'] for r in rows]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));value=report['comparisons'][arm+'-B0'];assert value['delta_J']==int(d.sum());assert np.allclose(value['ci95'],np.percentile(sample,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(sample,[2.5/20,100-2.5/20]),atol=1e-12,rtol=0);checks['bootstrap_intervals']+=1
    phase();write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),prior_independent_geometry_and_labels_sha256=sha(PRIOR_PILOT/'independent_audit.json'),scope='Independent causal public-ID reconstruction, alpha-dependent optional SciPy LAP objective, all outcomes, sealed selection and CIs. Ready geometry and actual-anchor labels reuse their completed independent audit.'),new=True)


if __name__=='__main__':main(guard())
