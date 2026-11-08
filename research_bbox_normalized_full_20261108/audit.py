"""Independent identity-link reconstruction, saved-map arithmetic and full metrics."""
import os,sys,gzip
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from collections import Counter,defaultdict
from research_bbox_normalized_full_20261108.common import *


def identity_output(config,native,metadata,scores,p,decisions):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    logs={d['frame']:d for d in decisions};events={d['frame']:d for d in metadata['events']};aliases={};owner={};seen=set();result=native.copy();checks=Counter()
    for frame in range(metadata['frames']):
        mask=native[:,0]==frame;ids=list(map(int,native[mask,5]));event=events.get(frame,dict(newly_confirmed=[],edges=[]));d=logs.get(frame,dict(links=[],revoked=[],allowed=[]));root=lambda t:aliases.get(t,t)
        for revoked in d['revoked']:
            tid=revoked['native_id'];canonical=revoked['old_canonical_id'];kept=revoked['kept_native_id'];assert tid in ids and kept in ids and tid!=kept and aliases[tid]==canonical and root(kept)==canonical;del aliases[tid]
        current={root(t) for t in ids};fresh=set(event['newly_confirmed']) & set(ids)-seen;eligible={}
        for e in event['edges']:
            accepted=bool(config.get('enabled',True)) and e['age']<=config['cap']
            if accepted and config.get('score_name') is not None:
                threshold=config['threshold'];value=scores[f"{frame}|{e['old_tid']}|{e['new_tid']}"]['scores'][config['score_name']];accepted=threshold!='off' and value is not None and value>=threshold
            old=e['old_tid'];new=e['new_tid'];canonical=root(old)
            if not accepted or new not in fresh or old not in seen or old in ids or owner.get(canonical)!=old or canonical in current or e['cost']>=1:continue
            assert e['row_data']['state']==2 and e['row_data']['history'][-1]['frame']<frame
            eligible[(old,new)]=dict(old_tid=old,new_tid=new,canonical_id=canonical,cost=e['cost'],corrected_cost=e['cost']-p['alpha'])
        assert sorted(eligible)==sorted((e['old_tid'],e['new_tid']) for e in d['allowed'])
        rows=sorted({a for a,b in eligible});columns=sorted({b for a,b in eligible});R=len(rows);C=len(columns)
        if R and C:
            M=np.full((R+C,R+C),np.inf);M[R:,C:]=0
            for i in range(R):M[i,C+i]=p['matching_threshold']/2
            for j in range(C):M[R+j,j]=p['matching_threshold']/2
            for (a,b),e in eligible.items():M[rows.index(a),columns.index(b)]=np.float32(e['corrected_cost'])
            ii,jj=linear_sum_assignment(M);optimal=M[ii,jj].sum();actual=sum(float(np.float32(e['corrected_cost'])) for e in d['links'])+(R+C-2*len(d['links']))*p['matching_threshold']/2;assert abs(actual-optimal)<1e-5;checks['optional_assignments']+=1
        assert len({e['old_tid'] for e in d['links']})==len(d['links'])==len({e['new_tid'] for e in d['links']})
        for e in d['links']:
            wanted=eligible[(e['old_tid'],e['new_tid'])];assert e==wanted;aliases[e['new_tid']]=e['canonical_id']
        canonical=[root(t) for t in ids];assert len(set(canonical))==len(ids);result[mask,5]=canonical
        for tid in ids:owner[root(tid)]=tid
        seen.update(ids);checks['causal_frames']+=1
    return result,checks


def main(obs):
    import numpy as np
    from datasets.bdd100k_loader import CLASS_NAMES
    from evaluation.hota_eval import VALID
    from research_singleton_free_20261108.full_audit import independent_returns
    from research_singleton_verify_full_audit_20261108.audit import score_from_saved
    full_phase();p=read(FULL/'protocol.json');report=read(FULL/'comparison_report.json');arms=list(p['parameters']);videos=list(p['videos']);checks=Counter();components={a:defaultdict(list) for a in arms};metrics={a:[] for a in arms};totals={a:Counter() for a in arms}
    for number,(video,entry) in enumerate(p['videos'].items()):
        metadata=read(FULL/'native_trace'/(video+'.json'))
        with np.load(FULL/'native_trace'/(video+'.npz'),allow_pickle=False) as z:native=z['outputs']
        with np.load(ROOT/entry['raw_array'],allow_pickle=False) as z:assert int(z['frame_count'])==metadata['frames'] and np.array_equal(native[:,:8],z['outputs'])
        anchors=read(FULL/'runtime_inputs'/(video+'.json'))['anchors'];gt={int(f['frameIndex']):f for f in read(ROOT/f'data/bdd100k/labels/box_track_20/val/{video}.json')};truth={}
        for frame,clock in anchors.items():
            fr=gt[clock['frameIndex']];assert fr['name']==clock['name'];objects=[]
            for obj in fr.get('labels') or []:
                if obj['category'] not in CLASS_NAMES.values() or (obj.get('attributes') or {}).get('crowd',False):continue
                objects.append((str(obj['id']),obj['category'],[obj['box2d'][k] for k in ['x1','y1','x2','y2']]))
            truth[int(frame)]=objects
        saved_scores=read(FULL/'scores'/(video+'.json'))['edges'];feature_receipt=read(FULL/'feature_receipts'/(video+'.json'));feature_cache={}
        def features(seq,kind):
            key=sequence_key(video,seq);path=FULL/'features'/video/(key+'.'+kind+'.npy');relative=path.relative_to(ROOT).as_posix()
            if relative not in feature_cache:assert sha(path)==feature_receipt['artifacts'][relative];feature_cache[relative]=np.load(path,allow_pickle=False).astype(float)
            return feature_cache[relative]
        for event in metadata['events']:
            for e in event['edges']:
                if e.get('unavailable'):
                    assert all(v is None for v in saved_scores[f"{event['frame']}|{e['old_tid']}|{e['new_tid']}"]['scores'].values());continue
                score=saved_scores[f"{event['frame']}|{e['old_tid']}|{e['new_tid']}"]['scores']
                for arm in ['SUPPORT','RANDOM','IMAGE','R1','DIS','FLYVIS','PERM']:
                    value,support=score_from_saved(arm,video,e['row_data'],e['box'],event['frame'],e['column'],e['transform'],features);assert support==e['support'];assert (value is None)==(score[arm] is None)
                    if value is not None:assert abs(value-score[arm])<1e-12,(video,event['frame'],arm,value,score[arm])
                    checks['independent_spatial_scores']+=1
        returns={}
        for arm in arms:
            rawpath=FULL/'rollouts'/arm/(video+'.npz');receipt=read(rawpath.with_suffix('.receipt.json'))
            for n,h in receipt['artifacts'].items():assert sha(ROOT/n)==h
            with np.load(rawpath,allow_pickle=False) as z:raw=z['outputs'];assert int(z['frame_count'])==entry['frames']
            assert np.array_equal(raw[:,[0,1,2,3,4,6,7]],native[:,[0,1,2,3,4,6,7]])
            decisions=read(FULL/'rollouts'/arm/(video+'.decisions.json'))['decisions'];expected,c=identity_output(p['variants'][arm],native,metadata,saved_scores,p,decisions);assert np.array_equal(expected[:,:8],raw);checks.update(c)
            signature=hashlib.sha256(raw.tobytes()).hexdigest()
            if signature not in returns:returns[signature]=independent_returns(raw,truth);checks['independent_return_trajectories']+=1
            r=read(FULL/'additional'/arm/(video+'.json'))
            for key in ['totals','events','duplicate_extras']:assert r[key]==returns[signature][key],(video,arm,key)
            totals[arm].update(r['totals']);checks['return_reports']+=1
            m=read(FULL/'metrics'/arm/(video+'.json'));metrics[arm].append(m['metrics']);details=FULL/'metrics'/arm/(video+'.details.json.gz');assert sha(details)==m['details_sha256'] and sha(FULL/'outputs'/arm/(video+'.json'))==m['output_sha256']
            with gzip.open(details,'rt',encoding='utf-8') as f:d=json.load(f)[video]
            for key in ['TP','FP','FN']:
                value=np.sum([d[c]['HOTA']['HOTA_'+key] for c in VALID if c in d],axis=0);assert np.allclose(value,m['components'][key],atol=1e-10,rtol=0);components[arm][key].append(value)
            aa=np.sum([np.asarray(d[c]['HOTA']['HOTA_TP'])*d[c]['HOTA']['AssA'] for c in VALID if c in d],axis=0);assert np.allclose(aa,m['components']['AA'],atol=1e-10,rtol=0);components[arm]['AA'].append(aa)
            for key in ['IDSw','CLR_FP','CLR_FN']:assert sum(d[c]['CLEAR']['IDSW' if key=='IDSw' else key] for c in VALID if c in d)==m['metrics'][key]
            for key in ['IDTP','IDFP','IDFN']:assert sum(d[c]['Identity'][key] for c in VALID if c in d)==m['components'][key]
            checks['official_metric_components']+=1
        status('native_confirm_full_independent_audit',video=video,complete=number+1,total=len(videos),checks=dict(checks),guard=obs.read())
    N=len(videos);idx=np.random.default_rng(42).integers(0,N,(20000,N));counts=np.asarray([np.bincount(row,minlength=N) for row in idx]);weights=np.vstack([np.ones((1,N)),counts]);samples={};points={}
    with np.load(FULL/'bootstrap_counts.npz',allow_pickle=False) as z:assert np.array_equal(z['counts'],counts)
    for arm,values in components.items():
        c={k:np.asarray(v) for k,v in values.items()};tp=weights@c['TP'];fp=weights@c['FP'];fn=weights@c['FN'];aa=weights@c['AA'];hota=100*np.sqrt(tp/np.maximum(1,tp+fp+fn)*aa/np.maximum(1,tp)).mean(axis=1);points[arm]=hota[0];samples[arm]=hota[1:];assert abs(points[arm]-report['pooled'][arm]['HOTA'])<1e-9;assert dict(totals[arm])==read(FULL/'additional_report.json')['arms'][arm]['totals']
    for key,value in report['comparisons'].items():
        a,b=key.split('-');delta=samples[a]-samples[b];assert abs(points[a]-points[b]-value['delta_pooled_HOTA'])<1e-9;assert np.allclose(value['ci95'],np.percentile(delta,[2.5,97.5]),atol=1e-9,rtol=0);assert np.allclose(value['ci_corrected'],np.percentile(delta,[2.5/20,100-2.5/20]),atol=1e-9,rtol=0)
        harm=dict(AssA_mean=float(np.mean([m['AssA'] for m in metrics[a]])-np.mean([m['AssA'] for m in metrics[b]])),IDF1_mean=float(np.mean([m['IDF1'] for m in metrics[a]])-np.mean([m['IDF1'] for m in metrics[b]])),IDSw=sum(m['IDSw'] for m in metrics[a])-sum(m['IDSw'] for m in metrics[b]),FP=sum(m['CLR_FP'] for m in metrics[a])-sum(m['CLR_FP'] for m in metrics[b]),FN=sum(m['CLR_FN'] for m in metrics[a])-sum(m['CLR_FN'] for m in metrics[b]),wrong_known_returns=totals[a]['wrong_returns']-totals[b]['wrong_returns']);assert all(abs(harm[k]-value['harm'][k])<1e-9 for k in harm);safe=harm['AssA_mean']>=0 and harm['IDF1_mean']>=0 and all(harm[k]<=0 for k in ['IDSw','FP','FN','wrong_known_returns']);assert safe==value['harm_passed'];checks['independent_intervals']+=1
    accepted={}
    for arm in p['eligible_systems']:
        c=report['comparisons'][arm+'-B0'];gain=dict(correct_returns=totals[arm]['correct_returns']-totals['B0']['correct_returns'],sustained_success=totals[arm]['sustained_success']-totals['B0']['sustained_success']);assert gain==report['identity_gain'][arm];accepted[arm]=bool(c['ci_corrected'][0]>0 and c['harm_passed'] and gain['correct_returns']>0 and gain['sustained_success']>0)
    assert report['any_preregistered_system_accepted']==any(accepted.values());assert accepted==report['accepted'];assert report['problem1_acceptance']==accepted.get(p['selected_system'],False)
    increment=all(report['comparisons']['FLYVIS-'+arm]['ci_corrected'][0]>0 and report['comparisons']['FLYVIS-'+arm]['harm_passed'] for arm in arms if arm!='FLYVIS');assert report['accepted_flyvis_system_and_increment']==bool(accepted.get('FLYVIS',False) and increment)
    full_phase();write(FULL/'independent_audit.json',dict(passed=True,checks=dict(checks),comparison_sha256=sha(FULL/'comparison_report.json'),additional_sha256=sha(FULL/'additional_report.json'),accepted=accepted,problem1_acceptance=report['problem1_acceptance'],scope='All native physical outputs, independent causal alias reconstruction and optional assignment objective, spatial map arithmetic, sparse identity returns, official metric components and registered intervals. No rerun of tracker/model/evaluator.'),new=True)


if __name__=='__main__':main(guard())
