"""Audit the new gate only; reuse prior verified neural/map computation."""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter
from research_native_confirm_risk_20261108.common import *


def main(obs):
    import numpy as np
    from research_native_confirm_full_20261108.audit import identity_output
    from research_singleton_verify_full_audit_20261108.audit import score_from_saved
    phase('phase.json');p=read(RUN/'protocol.json');selection=read(RUN/'selection.json');report=read(RUN/'comparison_report.json');assert sha(RUN/'selection.json')==report['selection_sha256'];checks=Counter();norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');mean=np.asarray(norm['mean']);sd=np.where(np.asarray(norm['sd'])>=1e-12,norm['sd'],1);source_manifest=read(PRIOR/'features_complete.json')['files']
    fit_totals={}
    videos={v:split for split,r in report['results'].items() for v in r['B0']['per_video']}
    for number,(video,split) in enumerate(videos.items()):
        metadata=read(TRACE/'native_trace'/(video+'.json'))
        with np.load(TRACE/'native_trace'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        sp=PRIOR/'scores'/(video+'.json');scores=read(sp)['edges'] if sp.exists() else {};cache={}
        def feature(seq,kind):
            key=sequence_key(video,seq)
            if key not in cache:
                path=PRIOR/'features'/(key+'.npz');assert sha(path)==source_manifest[key]
                with np.load(path,allow_pickle=False) as z:cache[key]={k:z[k].astype(float) if k!='DIS_VALID' else z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE']-mean)/sd
            return cache[key][kind]
        # The prior independent audit covered all four real representations.
        # Newly activated matched-frequency permutation/hash controls are checked here.
        for event in metadata['events']:
            for e in event['edges']:
                record=scores[f"{event['frame']}|{e['old_tid']}|{e['new_tid']}"]['scores']
                for kind in ['PERM','RANDOM']:
                    value,support=score_from_saved(kind,video,e['row_data'],e['box'],event['frame'],e['column'],e['transform'],feature);assert support==e['support'] and (value is None)==(record[kind] is None)
                    if value is not None:assert abs(value-record[kind])<1e-12
                    checks['new_null_control_scores']+=1
        for arm in report['results'][split]:
            path=RUN/'outcomes'/arm/(video+'.json');r=read(path);assert sha(path)==report['artifacts'][path.relative_to(ROOT).as_posix()];source=ROOT/r['reused_outcome_source'];assert sha(source)==r['reused_outcome_sha256'];source_data=read(source);assert source_data['decisions']==r['decisions'] and source_data['links']==r['links']
            expected,c=identity_output(r['config'],native,metadata,scores,p,r['decisions']);assert hashlib.sha256(expected.tobytes()).hexdigest()==r['outputs_sha256'];checks.update(c);totals=Counter(N=r['totals']['N'])
            for d in r['decisions']:totals['revoked']+=len(d['revoked'])
            for link in r['links']:
                label=link['label'];kind='correct' if label['known'] and label['positive'] else 'foreign' if label['known'] else 'unknown';assert kind==link['kind'];totals['links']+=1;totals[kind]+=1;totals['direct_current_known']+=bool(label.get('direct_current_known'))
            totals['J']=totals['correct']-totals['foreign'];assert dict(totals)==r['totals'];checks['outcome_ledgers']+=1
        status('empirical_error_gate_independent_audit',video=video,complete=number+1,total=len(videos),checks=dict(checks),guard=obs.read())
    # Check every candidate's empirical constraint and the deterministic selection.
    order=p['calibration']['source_candidates'];rank=lambda r:(r['totals'].get('correct',0),-r['cap'],-order.index(r['source']))
    assert max((r for r in selection['candidates'] if r['eligible']),key=rank)==selection['selected']
    assert max((r for r in selection['candidates'] if r['eligible'] and r['source']=='GEOMETRY'),key=rank)==selection['age_fit']
    for r in selection['candidates']:assert r['eligible']==(r['totals'].get('foreign',0)==0);assert r['totals'].get('J',0)==r['totals'].get('correct',0)-r['totals'].get('foreign',0)
    eval_videos=sorted(report['results']['evaluate26']['B0']['per_video']);idx=np.random.default_rng(42).integers(0,len(eval_videos),(20000,len(eval_videos)))
    for key,c in report['comparisons'].items():
        a,b=key.split('-');A=report['results']['evaluate26'][a]['per_video'];B=report['results']['evaluate26'][b]['per_video'];d=np.array([A[v].get('J',0)-B[v].get('J',0) for v in eval_videos]);n=np.array([A[v]['N'] for v in eval_videos]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));assert c['delta_J']==int(d.sum());assert np.allclose(c['ci95'],np.percentile(sample,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(c['ci_corrected'],np.percentile(sample,[2.5/16,100-2.5/16]),atol=1e-12,rtol=0);checks['independent_intervals']+=1
    chosen=selection['selected_system'];good=lambda a:report['comparisons'][a+'-B0']['ci95'][0]>0 and report['results']['evaluate26'][a]['totals'].get('foreign',0)==0
    assert report['full_gate']['proceed']==bool(selection['selected']['totals'].get('correct',0)>0 and (good(chosen) or good('AGE_FIT')))
    phase('phase.json');write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),prior_real_map_audit_sha256=sha(PRIOR/'independent_audit.json'),scope='All causal public outputs and optional assignments reconstructed independently. New null-control arithmetic checked, real maps reused from prior independent audit. Selection, ledgers, intervals and gate checked. No model/tracker/evaluator run.'),new=True)


if __name__=='__main__':main(guard())
