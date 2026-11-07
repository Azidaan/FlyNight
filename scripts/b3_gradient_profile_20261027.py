"""Б3: реальная наблюдаемая история внешнего вида -> память; проверка реализации."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone,timedelta
import sys,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/b123_20261027/b3_gradient';BASE=ROOT/'runs/b1_real_inputs_20261026'
MSK=timezone(timedelta(hours=3))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.next')
    temp.write_bytes((json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode());temp.replace(p)


def prepare():
    assert not RUN.exists();RUN.mkdir()
    p=read(BASE/'protocol.json')
    with np.load(BASE/'base_inputs.npz') as z:data={k:z[k] for k in ('uid','name','geo','mask')}
    with np.load(BASE/'supervision.npz') as z:labels=z['ident']
    with np.load(BASE/'base_predictions.npz') as z:probability=z['identity_probability']
    index={int(u):i for i,u in enumerate(data['uid'])}
    previous=read(ROOT/'runs/z13_box_supervision_followup_20261025/temporal_readiness.json')['previous_query']
    selected=None
    for video in p['fit84']:
        pointers={int(r['uid']):r for r in read(BASE/'inputs'/(video+'.json'))['rows']}
        for uid in sorted(pointers):
            row=pointers[uid];i=index[uid];n=len(row['choices'])-1
            if n<2 or not (labels[i,1:n+1]>=0).any():continue
            past=[];cursor=uid
            while str(cursor) in previous and len(past)<4:
                cursor=int(previous[str(cursor)]['previous_uid'])
                r=pointers.get(cursor)
                if r is None or r['current_frame']<=row['reference_frame']:break
                assert row['reference_frame']<r['current_frame']<row['current_frame']
                past.append(cursor)
            if len(past)>=2:selected=(video,row,i,list(reversed(past)));break
        if selected:break
    assert selected is not None,'No observable chronological memory clip'
    video,row,i,past=selected
    path=ROOT/f'runs/z13_masa_regions_20261026/recovery1/videos/{video}.npz'
    with np.load(path) as z:uid=z['uid'];regions=z['region'].astype(np.float32);exemplar=z['exemplar'].astype(np.float32)
    mindex={int(u):j for j,u in enumerate(uid)}
    last=exemplar[mindex[row['uid']]];events=[np.r_[last,1.]];observed=[]
    for u in past:
        j=index[u];tag=np.flatnonzero(data['mask'][j]&(data['geo'][j,:,-1]>.5))
        k=int(tag[0]) if len(tag) else 0
        present=k>0
        if present:last=regions[mindex[u],k-1]
        events.append(np.r_[last,float(present)]);observed.append(dict(uid=u,present=present,slot=k))
    n=len(row['choices'])-1
    np.savez_compressed(RUN/'observed_inputs.npz',history=np.asarray(events,np.float32),
        candidates=regions[mindex[row['uid']],:n],base=probability[i,1:n+1])
    np.save(RUN/'training_targets.npy',labels[i,1:n+1])
    metadata=dict(video=video,uid=row['uid'],reference_frame=row['reference_frame'],current_frame=row['current_frame'],
        previous_observations=observed,history_events=len(events),real_regions=n,
        input_selection='First fit84 query with >=2 real regions, known supervision available, and >=2 previous observations after its observed exemplar; no selection by correctness or benefit.',
        ground_truth_used_for_memory=False,current_candidates_used_to_update_memory=False)
    write(RUN/'sample.json',metadata)
    files=['scripts/b3_gradient_profile_20261027.py','tracking/b3_object_memory.py','tracking/b1_learning.py',
           'tracking/z13_readout.py','tracking/connectome_null.py',path.relative_to(ROOT).as_posix(),
           'runs/b123_20261027/b3_gradient/observed_inputs.npz','runs/b123_20261027/b3_gradient/training_targets.npy',
           'runs/b123_20261027/b3_gradient/sample.json']
    write(RUN/'protocol.json',dict(id='b3_object_memory_gradient_v1',created=datetime.now(MSK).isoformat(),
        timezone='Europe/Moscow',author_stop='2026-10-27T07:20:00+03:00',budget_seconds=300,device='cuda:0',
        arms=['C','D','W','B'],seed=26,action='Learn appearance257 -> receptor721 adapter, recurrent memory, same candidate comparison head; additive identity correction to frozen appearance base.',
        topologies='C stock frozen; D stock trainable; W same trainable core with type_sign degree/sign-preserving target permutation seed26; B ordinary existing spatial GRU. Same adapters/head/seed.',
        clock='One fixed computational update per observed query event, not a simulation of physical frame timing. Input history strictly chronological and earlier than current candidates.',
        training='Three scratch Adam steps, known real-region identity BCE; labels separate from observed inputs. No checkpoints promoted.',
        acceptance=['Zero output preserves base probabilities exactly','Finite loss and nonzero adapter gradient after head update; D/W finite nonzero core gradient',
                    'Report actual rewiring invariants and budgets; real inputs use only past observed assignments, not true identity'],
        limitations='One real clip and three scratch steps, not memory accuracy, topology superiority, HOTA or scientific B3 acceptance. Adapter has substantially more parameters than the core; ordinary control budget differs and is reported.',
        sha256={f:sha(ROOT/f) for f in files}))
    write(RUN/'status.json',dict(state='ready',stage='registered real memory implementation check',updated=datetime.now(MSK).isoformat()))


def run():
    p=read(RUN/'protocol.json')
    for f,h in p['sha256'].items():assert sha(ROOT/f)==h,f
    import torch
    torch.set_num_threads(2)
    from tracking.b3_object_memory import ObjectMemoryBranch
    from tracking.z13_readout import FlyBackbone,make_modules
    from tracking.connectome_null import rewire
    from scripts.z13_target_model import build_arm
    from tracking.gpu_lock import GpuLock
    with np.load(RUN/'observed_inputs.npz') as z:inputs={k:z[k] for k in z.files}
    history=torch.as_tensor(inputs['history'][None],device='cuda')
    candidates=torch.as_tensor(inputs['candidates'][None],device='cuda')
    base=torch.as_tensor(inputs['base'][None],device='cuda')
    labels=torch.as_tensor(np.load(RUN/'training_targets.npy')[None],device='cuda')
    known=labels>=0;records={};began=time.monotonic()
    with GpuLock('flyvis_fovea_gpu'):
        for arm in p['arms']:
            write(RUN/'status.json',dict(state='running',stage='real appearance memory backward',arm=arm,updated=datetime.now(MSK).isoformat()))
            _,geom,stock=build_arm('C',torch,None);rewiring=None
            if arm=='W':rewiring=rewire(stock.net,'type_sign',26);assert rewiring['in_degree_preserved'] and rewiring['fraction_changed']>0
            if arm=='B':
                _,cls=make_modules(torch);torch.manual_seed(26);bb=cls(geom.neighbors).cuda();bb.dev=torch.device('cuda')
            else:bb=FlyBackbone(stock.est,trainable=arm in ('D','W'))
            torch.manual_seed(26);torch.cuda.manual_seed_all(26)
            model=ObjectMemoryBranch(len(geom.points)).cuda()
            groups=[dict(params=list(model.parameters()),lr=.001)]
            if arm in ('D','W'):groups+=bb.param_groups()
            if arm=='B':groups+=[dict(params=list(bb.parameters()),lr=.001)]
            optimizer=torch.optim.Adam(groups);losses=[];norms=[];adapter=[]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
            for step in range(3):
                optimizer.zero_grad(set_to_none=True);probability,delta=model(history,candidates,base,bb)
                if step==0:exact=torch.equal(probability,base);assert exact
                loss=torch.nn.functional.binary_cross_entropy(probability[known].clamp(1e-9,1-1e-9),labels[known].to(probability.dtype))+.01*delta.square().mean()
                if arm in ('D','W'):loss=loss+bb.penalty()
                assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach()));adapter.append(float(model.input.weight.grad.norm()))
                if arm in ('D','W'):norms.append([float(x.grad.norm()) if x.grad is not None else 0 for x in bb.params])
                torch.nn.utils.clip_grad_norm_([x for g in groups for x in g['params']],1.);optimizer.step()
                if arm in ('D','W'):bb.clamp()
            torch.cuda.synchronize();assert adapter[-1]>0 and np.isfinite(adapter).all()
            if arm in ('D','W'):assert np.isfinite(norms).all() and sum(norms[-1])>0
            records[arm]=dict(zero_start_exact=exact,steps=3,losses=losses,adapter_gradient_norms=adapter,
                core_gradient_norms=norms,rewiring=rewiring,seconds=time.monotonic()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),adapter_and_head_parameters=sum(x.numel() for x in model.parameters()),
                backend_trainable_parameters=sum(x.numel() for g in groups[1:] for x in g['params']))
            del bb,model,optimizer
    write(RUN/'report.json',dict(passed=True,stage='real memory implementation only',arms=records,
        total_seconds=time.monotonic()-began,scientific_hypothesis_tested=False,hota_measured=False,
        memory_accuracy_measured=False,scratch_weights_promoted=False,limitation=p['limitations']))
    write(RUN/'status.json',dict(state='completed',stage='memory path implementation checked; scientific pilot pending',updated=datetime.now(MSK).isoformat()))


if __name__=='__main__':
    try:prepare() if '--prepare' in sys.argv else run()
    except Exception as e:
        if RUN.exists():write(RUN/'failure.json',dict(error=repr(e),updated=datetime.now(MSK).isoformat()))
        raise
