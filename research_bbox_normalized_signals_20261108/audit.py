"""Independent saved-map arithmetic and alpha-LAP reconstruction; no neural rerun."""
import os,sys,copy
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_normalized_signals_20261108.common import *


def main(obs):
    import numpy as np
    from research_singleton_verify_full_audit_20261108.audit import score_from_saved
    from research_bbox_cost_pilot_20261108.audit_replay import replay
    phase();phase('comparison_phase.json');report=read(RUN/'comparison_report.json');selection=read(RUN/'selection.json');inventory=read(RUN/'inventory.json');byvideo=defaultdict(list);manifest=read(RUN/'features_complete.json')['files'];norm=read(ROOT/'runs/spatial_identity_maps_20261107/image_normalization.json');mean=np.array(norm['mean']);sd=np.where(np.array(norm['sd'])>=1e-12,norm['sd'],1);checks=Counter();entries=read(INVENTORY/'protocol.json')['videos']
    for e in inventory['events']:byvideo[e['video']].append(e)
    for video,entry in entries.items():
        if entry['split']=='val20':continue
        rawmeta=read(INVENTORY/'native'/(video+'.json'))
        with np.load(INVENTORY/'native'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        cache={};path=RUN/'scores'/(video+'.json');scores=read(path)['edges'] if path.exists() else {};lookup={edge['frame']:{} for edge in byvideo[video]}
        def feature(sequence,name):
            key=sequence_key(video,sequence)
            if key not in cache:
                path=RUN/'features'/(key+'.npz');assert sha(path)==manifest[key]
                with np.load(path,allow_pickle=False) as z:cache[key]={k:z[k].astype(float) if k!='DIS_VALID' else z[k] for k in z.files}
                cache[key]['IMAGE']=(cache[key]['IMAGE']-mean)/sd
            return cache[key][name]
        for e in byvideo[video]:
            old=np.array(e['row_data']['history'][-1]['box']);box=np.array(e['box']);scale=(box[2:]-box[:2])/(old[2:]-old[:2]);assert np.array_equal(scale,e['transform']['scale']) and e['transform']['offset']==[0.,0.]
            key=f"{e['frame']}|{e['old_tid']}|{e['new_tid']}";lookup[e['frame']][(e['old_tid'],e['new_tid'])]=e
            for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                if e['unavailable']:
                    assert scores[key]['scores'][arm] is None
                    if len(e['row_data']['history'])>=2:value,support=score_from_saved('SUPPORT',video,e['row_data'],e['box'],e['frame'],e['column'],e['transform'],feature);assert value is None and support==0
                else:
                    value,support=score_from_saved(arm,video,e['row_data'],e['box'],e['frame'],e['column'],e['transform'],feature);actual=scores[key]['scores'][arm];assert support==scores[key]['support'];assert (value is None)==(actual is None)
                    if value is not None:assert abs(value-actual)<1e-10,(video,key,arm,value,actual)
                checks['independent_spatial_scores']+=1
        labels=read(PRIOR_PILOT/'labels'/(video+'.json'))['rows']
        for arm,cfg in selection['configs'].items():
            r=read(RUN/'outcomes'/arm/(video+'.json'))
            if arm in ['B0','BBOX']:
                assert sha(ROOT/r['reused_outcome_source'])==r['reused_outcome_sha256'];assert read(ROOT/r['reused_outcome_source'])['outputs_sha256']==r['outputs_sha256'];checks['ready_controls_reused']+=1
            else:
                filtered=[]
                for d in rawmeta['events']:
                    rows=[]
                    for row in d['rows']:
                        eligible=[c for c in row['candidates'] if c['forward_iou']>=.2]
                        if row['age']>2 or len(eligible)!=1 or eligible[0]['new_tid'] is None:continue
                        c=eligible[0];key=(row['old_tid'],c['new_tid']);event=lookup.get(d['frame'],{}).get(key)
                        if event is None:continue
                        value=scores[f"{d['frame']}|{row['old_tid']}|{c['new_tid']}"]['scores'][cfg['score_name']];threshold=cfg['threshold'];accepted=threshold!='off' and value is not None and np.isfinite(value) and value>=threshold
                        if accepted:rows.append(row)
                    filtered.append(dict(d,rows=rows))
                expected,c=replay(native,dict(rawmeta,events=filtered),dict(cap=2,threshold=.2,bidirectional=False),r['decisions'],.025);checks.update(c);assert hashlib.sha256(expected.tobytes()).hexdigest()==r['outputs_sha256']
            counts=Counter(N=sum(l['known'] for l in labels.values()))
            for d in r['decisions']:counts['revoked']+=len(d['revoked'])
            for l in r['links']:counts['links']+=1;counts[l['kind']]+=1
            counts['J']=counts['correct']-counts['foreign'];assert dict(counts)==r['totals']==report['results'][entry['split']]['per_video'][video][arm];checks['outcome_ledgers']+=1
        status('normalized_maps_independent_audit',video=video,checks=dict(checks),guard=obs.read())
    fit=report['results']['fit20']['totals'];order=['SUPPORT','IMAGE','R1','DIS','FLYVIS'];eligible=[a for a in order if fit[a].get('correct',0)>0 and fit[a].get('foreign',0)==0];selected=max(eligible,key=lambda a:(fit[a].get('correct',0),-fit[a].get('links',0),-order.index(a))) if eligible else 'B0';assert eligible==selection['eligible_systems'] and selected==selection['selected']==report['selected'];assert sha(RUN/'selection.json')==report['selection_sha256'];names=list(report['results']['evaluate26']['per_video']);idx=np.random.default_rng(42).integers(0,len(names),(20000,len(names)))
    for key,value in report['comparisons'].items():
        a,b=key.split('-');per=report['results']['evaluate26']['per_video'];d=np.array([per[v][a]['J']-per[v][b]['J'] for v in names]);n=np.array([per[v][a]['N'] for v in names]);sample=100*d[idx].sum(1)/np.maximum(1,n[idx].sum(1));assert value['delta_J']==int(d.sum());assert np.allclose(value['ci95'],np.percentile(sample,[2.5,97.5]),atol=1e-12,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(sample,[2.5/20,100-2.5/20]),atol=1e-12,rtol=0);checks['bootstrap_intervals']+=1
    phase();phase('comparison_phase.json');write(RUN/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(RUN/'comparison_report.json'),scope='Independent box-relative mapping, saved-map correlation/permutation/motion, canonical causal ID reconstruction, optional SciPy LAP, ready controls, fit selection and intervals. No full benefit or proof of anatomical part matching.'),new=True)


if __name__=='__main__':main(guard())
