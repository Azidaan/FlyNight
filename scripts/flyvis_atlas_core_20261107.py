"""Prospective, single-pulse perturbation of every cell type in frozen FlyVis."""
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
OUT = Path('runs/flyvis_atlas_core_20261107')
PRE = Path('runs/flyvis_exclusive_t4t5_20261107')
PAIR = Path('runs/z13_temporal_pair_readiness_20261025')
FULL = Path('runs/z13_spatial_set_readout_20261025/recovery2/features/C/train26')
WEIGHT = Path('runs/z13_region_assignment_20261025/recovery2/features/C/train26')
NORM = Path('runs/z13_target_center_dev_20261018/C/normalization.json')
GUARD = Path('runs/flyvis_repairs_20261106/unification_20261107/system_memory_guard.py')
TYPES = Path('runs/flyvis_state_subgroups_ram_stream4_20261107/node_types.npy')
SPECS = Path('tools/flyvis-runtime/flyvis/connectome/fib25-fib19_v2.2.json')
CHECK = Path('runs/flyvis/results/flow/0000/000/best_chkpt')
SOURCES = [Path(__file__).resolve().relative_to(ROOT),Path('scripts/flyvis_atlas_core_audit_20261107.py'),
           Path('tracking/z13_readout.py'),Path('tracking/flyvis_motion.py'),Path('tracking/foveated_eyes.py'),
           Path('tracking/flyvis_checkpoint.py'),Path('tools/flyvis-runtime/flyvis/network/network.py'),
           NORM,CHECK,SPECS,TYPES,GUARD,PRE/'choices.csv.gz',PRE/'inputs_used.json',PRE/'measurement.json',PAIR/'protocol.json']


def sha(path):
    dig = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            dig.update(block)
    return dig.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def selected():
    with gzip.open(PRE/'choices.csv.gz','rt',encoding='ascii') as f:
        rows = [r for r in csv.DictReader(f) if r['split']=='evaluate26' and r['exactly2']=='1' and r['label_reason']=='ok' and r['exclusive_choice']]
    # Round robin by within-video UID; no ordering based on scores or outcomes.
    by={}
    for r in sorted(rows,key=lambda r:int(r['uid'])):
        by.setdefault(r['video'],[]).append(r)
    rows=[by[v][i] for i in range(max(map(len,by.values()))) for v in sorted(by) if i<len(by[v])]
    assert len(rows)==172 and len(by)==23
    return rows


def prepare():
    import numpy as np
    assert not OUT.exists(), 'Refuse to overwrite prior run'
    previous=read(PRE/'measurement.json')
    assert sha(PRE/'choices.csv.gz')==previous['rows_sha256'] and sha(PRE/'inputs_used.json')==previous['inputs_sha256']
    rows=selected()
    cell_types=np.load(TYPES,allow_pickle=False).astype(str)
    types, n=np.unique(cell_types,return_counts=True)
    assert len(types)==65
    orig=read(PAIR/'protocol.json')['sha256']; fp=read(PRE/'inputs_used.json')
    # Freeze inputs against archival manifests, including exact chunk file names.
    wanted={int(r['uid']):{} for r in rows}
    binds={str(p.as_posix()):sha(p) for p in SOURCES}
    for folder,name in ((WEIGHT,'w'),(FULL,'full')):
        for path in sorted(folder.glob('chunk*.npz')):
            with np.load(path,allow_pickle=False) as z:
                slots=[(int(u),i) for i,u in enumerate(z['uid']) if int(u) in wanted]
            if slots:
                k=path.as_posix(); assert sha(path)==fp[k];binds[k]=fp[k]
            for u,i in slots:
                assert name not in wanted[u];wanted[u][name]=[k,i]
    for video in sorted({r['video'] for r in rows}):
        raw=Path(f'runs/z13_data/{video}.npz');meta=PAIR/'inputs'/f'{video}.json'
        for p,h in [(raw,orig[raw.as_posix()]),(meta,fp[meta.as_posix()])]:
            assert sha(p)==h;binds[p.as_posix()]=h
    cases=[dict(uid=int(r['uid']),video=r['video'],**wanted[int(r['uid'])]) for r in rows]
    assert all('w' in c and 'full' in c for c in cases)
    p=dict(time=datetime.now(timezone(timedelta(hours=3))).isoformat(),git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hypothesis='One 10% pulse to previous-frame state of a single type changes the frozen 34-direct-channel candidate score after one further real frame. Tests immediate sensitivity, not chronic deletion.',
        model='flow/0000/000 best, unchanged archive observed-exemplar coefficients, both complete candidate masks; no neighbor features, no GT in forward',
        selection='All 172 previously eligible evaluate26 historical training-source pairs. Round-robin one UID per video ordered lexicographically, then remaining UIDs. 23 videos represented. If gate fails, stop; partial results descriptive only.',
        intervention='Once after penultimate image, clone tensor nodes.activity, scale every column of ONE cell type by 0.90, run original full next frame (two internal steps). Same one-type 1.10 mirrored pulse as matched control. Each treated type includes its actual number of units; absolute injected energy not equalized.',
        controls='Untouched vs same-state clone identity; archive validation on all 34 normalized channel maps in both full masks and exact ordering for original 34 and T4/T5. Check no outside-mask effect at injection and base state immutability. No model training, label only in evaluator.',
        gates=dict(archive_normalized_map_max_abs=0.002,archive_score_max_abs=0.00002,identity_normalized_map_max_abs=0.00001,
                   archive_rank_exact=True,identity_score_abs=0.0000001,tie_abs=1e-10),
        outcome='Raw mean normalized 34-vectors per candidate for every condition saved. Paired difference in correct choices vs untouched, tie=0.5, by whole video. Bootstrap 20000 video blocks seed42, 95% descriptive and Bonferroni family65. Exploratory even if positive; not end-to-end/HOTA.',
        budget='Single guarded GPU, bounded small batches, no BoxEye, no video decode, finite 172. One complete UID timing for projection before continuing; no tuning on trial results.',types=types.tolist(),counts=n.tolist(),cases=cases,source_sha256=binds)
    OUT.mkdir(); (OUT/'raw').mkdir(); write(OUT/'protocol.json',p)
    print('registered',len(types),len(cases),len(binds),flush=True)


def run(obs):
    import numpy as np
    import torch
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    p=read(OUT/'protocol.json')
    assert not list((OUT/'raw').glob('*.npz'))
    for k,h in p['source_sha256'].items(): assert sha(Path(k))==h, k
    g=p['gates']; dim=34
    norm=read(NORM); mu=np.asarray(norm['mean'])[:dim];sd=np.asarray(norm['sd'])[:dim];assert np.all(sd>0)
    def take(state, frames_prev, frames_cur):
        return bb.step(state,frames_prev,frames_cur)
    def cloned(values):
        st=bb.net._initial_state(params,values.shape[0])
        assert list(st.nodes)==['activity'] and not list(st.edges)
        st.nodes.activity=values.clone()
        return bb.net._state_api(st)
    def rank(x):return int(x>1e-10)-int(x< -1e-10)
    def measures(features, masks):
        # features = [B,34,721]. Preserve archive's FP32 map after train normalization.
        arr=((features.detach().cpu().numpy().transpose(0,2,1).astype('float64')-mu)/sd).astype('float32')
        assert np.isfinite(arr).all()
        avg=np.stack([arr[:,m].astype('float64').mean(1) for m in masks],axis=1)
        return arr,avg
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        _,_,bb=build_arm('C',torch,None);device=bb.dev
        assert str(device).startswith('cuda') and not bb.net.training and not bb.net._state_hooks
        params=bb.net._param_api()
        types=bb.net.connectome.nodes.type[:].astype(str)
        assert np.array_equal(types,np.load(TYPES,allow_pickle=False).astype(str))
        ids=[torch.as_tensor(np.flatnonzero(types==n),dtype=torch.int64,device=device) for n in p['types']]
        assert [int(a.numel()) for a in ids]==p['counts'] and torch.cat(ids).unique().numel()==len(types)
        for number,c in enumerate(p['cases']):
            uid=c['uid']; started=time.perf_counter()
            meta=next(r for r in read(PAIR/'inputs'/(c['video']+'.json'))['rows'] if r['uid']==uid)
            with np.load(meta['retinal_source'],allow_pickle=False) as z:
                off=z['ep_offset'];a=meta['state_prefix_start'];b=meta['current_array_index']
                assert a==int(off[meta['episode_local']]) and b<off[meta['episode_local']+1] and b==meta['state_prefix_end']
                hx=torch.from_numpy(z['fr_hex'][a:b+1].astype('float32')).to(device)[None]
            with np.load(c['full'][0],allow_pickle=False) as z:
                i=c['full'][1];assert int(z['uid'][i])==uid
                ref=z['full'][i,:,:dim]
                masks=z['retinal_mask'][i,1:3]
                assert all(np.array_equal(np.flatnonzero(m),inds) for m,inds in zip(masks,meta['current_masks'],strict=True))
            with np.load(c['w'][0],allow_pickle=False) as z:
                i=c['w'][1];assert int(z['uid'][i])==uid
                weights=z['seed'][i,:dim].astype('float64')
            state=bb.init(hx[:,:1])
            for j in range(1,hx.shape[1]-1):
                state,_=take(state,hx[:,j-1],hx[:,j])
            original=state.nodes.activity.clone()
            prev,curr=hx[:,-2],hx[:,-1]
            _,f=take(state,prev,curr); actual,avg=measures(f,masks)
            refavg=np.stack([ref[m].astype('float64').mean(0) for m in masks])
            whole=np.array([abs((avg[0]-refavg)@weights).max(),abs((avg[0,:,4:12]-refavg[:,4:12])@weights[4:12]).max()])
            archive_diff=float(np.abs(actual[0]-ref).max())
            for slic in [slice(None),slice(4,12)]:
                ar=refavg[:,slic]@weights[slic]; z=avg[0,:,slic]@weights[slic]
                assert rank(float(ar[0]-ar[1]))==rank(float(z[0]-z[1])),('archive_choice',uid)
            assert archive_diff<g['archive_normalized_map_max_abs'] and whole.max()<g['archive_score_max_abs'],('archive',uid,archive_diff,whole)
            trials=[(-1,1.),*[(i,s) for s in [.9,1.1] for i in range(len(ids))]]
            allavg=[avg];iderr=None
            for start in range(0,len(trials),4):
                part=trials[start:start+4];B=len(part)
                values=original.expand(B,-1).clone()
                for k,(idx,scale) in enumerate(part):
                    if idx>=0:
                        indices=ids[idx]; mask=torch.ones_like(original[0],dtype=torch.bool);mask[indices]=False
                        old=original[0,indices];expected=old*scale;values[k,indices]=expected
                        assert torch.equal(values[k,indices],expected) and torch.equal(values[k,mask],original[0,mask])
                _,f=take(cloned(values),prev.expand(B,-1),curr.expand(B,-1))
                arr,av=measures(f,masks)
                if start==0:
                    iderr=float(np.abs(arr[0]-actual[0]).max())
                    assert iderr<=g['identity_normalized_map_max_abs'] and np.abs((av[0]-avg[0])@weights).max()<=g['identity_score_abs'],('identity',uid,iderr)
                allavg.append(av)
            assert torch.equal(original,state.nodes.activity)
            joined=np.concatenate(allavg)
            dest=OUT/'raw'/f'{uid}.npz';tmp=dest.with_suffix('.pending')
            with tmp.open('xb') as of:
                np.savez_compressed(of,uid=uid,means=joined,weights=weights,reference=refavg,archive_map_max=archive_diff,identity_map_max=iderr)
            os.replace(tmp,dest); torch.cuda.synchronize()
            secs=time.perf_counter()-started
            info=dict(complete=number+1,total=len(p['cases']),uid=uid,seconds=secs,guard=obs.read(),trajectory_inputs=len(hx[0]),archive_max=archive_diff)
            if number==0:write(OUT/'first_case.json',dict(**info,projected_remaining_s=secs*(len(p['cases'])-1)))
            write(OUT/'progress.json',info)
            if number%10==0:print(number+1,uid,round(secs,3),flush=True)
    write(OUT/'inference_completed.json',dict(count=len(p['cases']),raw_sha256={f.name:sha(f) for f in sorted((OUT/'raw').glob('*.npz'))},
                                              guard=obs.read(),protocol_sha256=sha(OUT/'protocol.json')))


if __name__=='__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--run' in sys.argv:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
        sys.path.insert(0,str(GUARD.parent));from system_memory_guard import guard_entrypoint
        run(guard_entrypoint(timeout=4800))
