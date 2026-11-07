"""Одна контролируемая проверка признака MASA в готовом сшивателе, без нового GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
from datetime import datetime
import sys,time,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.recovery_research_service_20261027 import RUN as BASE,read,write,sha,now

def merge_map(signature,pairs,probabilities,threshold=.6):
    import numpy as np
    ids=signature['track_id'];frames={}
    for row in signature['outputs']:frames.setdefault(int(row[5]),set()).add(int(row[0]))
    parent={};used_a=set();used_b=set();accepted=[];merged={t:set(f) for t,f in frames.items()}
    def root(t):
        seen=set()
        while t in parent and t not in seen:seen.add(t);t=parent[t]
        return t
    for k in np.argsort(-probabilities):
        if probabilities[k]<=threshold:break
        a,b=map(int,pairs[k]);ra,rb=root(int(ids[a])),root(int(ids[b]))
        if a in used_a or b in used_b or ra==rb or merged.get(ra,set())&merged.get(rb,set()):continue
        used_a.add(a);used_b.add(b);parent[rb]=ra;accepted.append(int(k));merged[ra]=merged.get(ra,set())|merged.pop(rb,set())
    return {int(t):root(int(t)) for t in ids},accepted

def apply_map(rows,mapping):
    result=copy.deepcopy(rows)
    for row in result:
        for label in row['labels']:label['id']=str(mapping.get(int(label['id']),int(label['id'])))
    return result

def main():
    import numpy as np,joblib
    from scripts import stitch_eval_v2 as V
    from scripts.z13_continue import paired
    from evaluation.hota_eval import evaluate_hota
    run=BASE/'masa_feature_contribution';p=read(run/'protocol.json');end=datetime.fromisoformat(read(run/'execution_deadline.json')['deadline']).timestamp();start=time.monotonic()
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    root=ROOT/'runs/z13_val20_matched_20261024';old=read(root/'native_masa_recovery2/report.json');head=joblib.load(ROOT/'runs/z13_center_hota_20261021/masa_head.joblib');model=head['model'];cols=head['cols']
    assert cols==[V.NAMES.index(c) for c in V.HEADS['base_masa']] and V.NAMES[cols[-1]]=='masa_cos'
    V._set(150,5.,True);V.WIN['dino_dir']=root/'xywh/dino_compat';V.WIN['masa_dir']=root/'xywh/masa'
    arms=['neutral_masa','shuffled42','shuffled43','shuffled44'];per={'raw':{},'full':{},**{name:{} for name in arms}};proof={};timing={}
    def status(**kw):
        if time.time()>end-5 or (run/'stop.request').exists():raise TimeoutError('MASA feature contribution budget')
        write(run/'status.json',dict(state='running',updated=now(),**kw),required=False)
    for i,v in enumerate(p['videos']):
        began=time.monotonic();status(video=v,index=i,total=20,stage='Готовые признаки и неизменные кандидаты')
        data=V.video_pairs((v,str(root/'xywh/signatures'),str(root/'cache'),'val'));signature=np.load(root/'xywh/signatures'/v/'signatures.npz');x=data['X'][:,cols];scores=model.predict_proba(x)[:,1] if len(x) else np.zeros(0)
        mapping,accepted=merge_map(signature,data['pairs'],scores);raw=read(root/'xywh/evaluated_raw'/(v+'.json'));full=apply_map(raw,mapping)
        assert full==read(root/'xywh/evaluated_masa'/(v+'.json')),(v,'Full frozen replay differs')
        assert len(accepted)==old['masa_links']['xywh'][v]
        per['raw'][v]=old['per_video']['xywh_raw'][v];per['full'][v]=old['per_video']['xywh_masa'][v]
        np.savez_compressed(run/(v+'_features.npz'),X=x,pairs=data['pairs'],full_scores=scores)
        write(run/'raw'/(v+'.json'),raw);write(run/'full'/(v+'.json'),full)
        item=dict(full_replay_exact=True,candidates=len(data['pairs']),full_links=len(accepted),unknown_candidate_labels=int((data['y']<0).sum()),arms={})
        for name in arms:
            xx=x.copy()
            if name=='neutral_masa':xx[:,-1]=model[0].mean_[-1]
            else:
                seed=int(name.removeprefix('shuffled'));perm=np.random.default_rng(seed).permutation(len(xx));xx[:,-1]=x[perm,-1]
            probabilities=model.predict_proba(xx)[:,1] if len(xx) else np.zeros(0);roots,links=merge_map(signature,data['pairs'],probabilities);out=apply_map(raw,roots)
            status(video=v,index=i,total=20,variant=name,stage='Полная HOTA фиксированного контроля')
            write(run/name/(v+'.json'),out);per[name][v]=evaluate_hota(run/name,[v],split='val')
            item['arms'][name]=dict(links=len(links),changed_pair_scores=int(np.count_nonzero(probabilities!=scores)),output_sha256=sha(run/name/(v+'.json')))
        proof[v]=item;timing[v]=time.monotonic()-began;write(run/'per_video.json',per);write(run/'progress.json',dict(proof=proof,timing=timing))
        if i==0:
            cost=2*timing[v]*19+120;available=end-time.time();write(run/'price.json',dict(first_full_video_seconds=timing[v],conservative_remaining_seconds=cost,available_seconds=available,proceed=cost<available));assert cost<available
    pairs={'full_vs_raw':old['paired']['xywh_masa-xywh_raw']}
    for name in arms:
        pairs['full_vs_'+name]=paired({v:per['full'][v]['HOTA'] for v in p['videos']},{v:per[name][v]['HOTA'] for v in p['videos']})
        pairs[name+'_vs_raw']=paired({v:per[name][v]['HOTA'] for v in p['videos']},{v:per['raw'][v]['HOTA'] for v in p['videos']})
    pooled={'raw':old['pooled']['xywh_raw'],'full':old['pooled']['xywh_masa']}
    for name in arms:status(stage='Совокупная HOTA',variant=name);pooled[name]=evaluate_hota(run/name,p['videos'],split='val')
    write(run/'report.json',dict(complete=True,per_video=per,pooled=pooled,paired=pairs,proof=proof,timing=timing,
        feature_added_value_supported=all(pairs['full_vs_'+a]['ci95'][0]>0 for a in arms),new_training=False,new_gpu_inference=False,
        ground_truth_used_for_scores_or_candidates=False,seconds=time.monotonic()-start,limitation=p['limitation']))
    write(run/'status.json',dict(state='completed',updated=now()))

if __name__=='__main__':main()
