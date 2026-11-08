"""Recover the26 discarded standard decoder channels, preserving ready C8/R134."""
import os
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import hashlib,json,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timedelta,timezone
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;BASE=ROOT/'runs/bbox_fit_maps_20261108'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'runs/flyvis_repairs_20261106/unification_20261107'))
from system_memory_guard import guard_entrypoint
guard_entrypoint(timeout=1200)
import numpy as np
from scripts.z13_target_model import receptor_features_rich
from tracking.flyvis_activity import OUTPUT_CELL_TYPES
OTHER=[0,1,2,3]+list(range(12,34))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify(p):
    for path,digest in p['sources'].items():assert sha(ROOT/path)==digest,path
def stamp():return datetime.now(timezone(timedelta(hours=3))).isoformat()
def prepare():
    master=read(BASE/'input_complete.json')['requests'];manifest=read(BASE/'features_complete.json')['files']
    recipes={};sources=[Path(__file__),BASE/'input_complete.json',BASE/'features_complete.json',
        ROOT/'scripts/z13_target_model.py',ROOT/'tracking/z13_readout.py',ROOT/'tracking/flyvis_activity.py',
        ROOT/'tracking/flyvis_adapter.py',ROOT/'tracking/flyvis_checkpoint.py',ROOT/'tracking/flyvis_motion.py',ROOT/'tracking/foveated_eyes.py',
        ROOT/'runs/flyvis/results/flow/0000/000/best_chkpt',ROOT/'runs/flyvis/results/flow/0000/000/_meta.yaml',
        ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json',ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json']
    for prefix in ['bbox_fit_maps_20261108','bbox_normalized_signals_20261108','singleton_verify_20261108','native_confirm_policy_20261108','identity_assignment_20261108']:
        base=ROOT/'runs'/prefix;path=base/'input_complete.json'
        if not path.exists():continue
        sources.append(path)
        for key,r in read(path).get('requests',{}).items():
            if key in master and key not in recipes and 'input_indices' in r:
                assert r['sequence']==master[key]['sequence'];recipes[key]=(base,r)
    data={};cache={};normal=read(ROOT/'runs/z13_target_center_dev_20261018/R1/normalization.json')
    mu=np.asarray(normal['mean'])[:34];sd=np.asarray(normal['sd'])[:34];errors={};origins={}
    for key,r in master.items():
        expected=BASE/'features'/(key+'.npz');assert sha(expected)==manifest[key];sources.append(expected)
        if key in recipes:
            base,recipe=recipes[key];path=base/'retinal'/(r['video']+'.npz');marker=path.with_suffix('.json')
            item=read(marker);assert sha(path)==item['npz_sha256'];sources.extend([path,marker])
            if path not in cache:
                with np.load(path,allow_pickle=False) as z:cache[path]={n:z[n].copy() for n in ['frames','crops','hex']}
            z=cache[path];idx=np.asarray(recipe['input_indices']);hx=z['hex'][idx]
            assert np.array_equal(z['frames'][idx],[v[0] for v in r['sequence']])
            assert np.array_equal(z['crops'][idx],[v[1] for v in r['sequence']])
            origins[key]=dict(kind='archived_input_recipe',path=path.relative_to(ROOT).as_posix(),indices=idx.tolist())
        else:
            path=ROOT/f'runs/z13_data/{r["video"]}.npz';sources.append(path)
            if path not in cache:
                with np.load(path,allow_pickle=False) as z:
                    cache[path]={(int(f),tuple(map(int,c))):h.copy() for f,c,h in zip(z['fr_frame'],z['fr_crop'],z['fr_hex'])}
            hx=np.stack([cache[path][(int(f),tuple(c))] for f,c in r['sequence']])
            origins[key]=dict(kind='archived_retinal_views',path=path.relative_to(ROOT).as_posix(),sequence=r['sequence'])
        data[key]=hx.copy()
        actual=((receptor_features_rich(hx[None].astype(float))[-1][0].T-mu)/sd).astype(np.float32)
        with np.load(expected,allow_pickle=False) as z:error=float(np.max(abs(actual-z['R1'])))
        errors[key]=error;assert error<=1e-4,(key,error)
    with (OUT/'inputs.npz').open('xb') as f:np.savez_compressed(f,**data)
    write('input_readiness.json',dict(passed=True,sequences=len(data),recipes=len(recipes),fallback_views=len(data)-len(recipes),
        r1_parity_max_error=max(errors.values()),r1_parity_errors=errors,origins=origins,inputs_sha256=sha(OUT/'inputs.npz')))
    sources += [OUT/'inputs.npz',OUT/'input_readiness.json']
    p=dict(registered=stamp(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        deadline='2026-11-09T07:40:00+03:00',requests=master,other_indices=OTHER,other_types=[OUTPUT_CELL_TYPES[i] for i in OTHER],
        question='Does discarding26 of34 standard decoder input cell maps lose information useful for instance association?',
        scope='Recover missing outputs only. All399 causal input sequences have old34 R1 parity. Replay is necessary because previous extractor discarded26 core outputs. Keep existing C8 and R134 unchanged.',
        justification='The34 types are the fixed standard flow decoder inputs, not a completeness scan of all65 cell types. Older34-channel spatial readers already existed; this preparation supports a different spatial patch comparison.',
        numerical_gate='For every newly replayed sequence require old normalized T4/T5 channels to match archived C8 <=1e-4 before saving only the other26. Original core weights unchanged. No labels or metric-based selection.',
        execution='One GPU lock, one-item sequences, idle CPU priority and50ms pause per sequence. Independent memory reserve guard timeout1200s. No video decoding, no core training. Finished per-sequence files reused after hash check.',
        intended_comparison='Freeze two readers (correlation and the fixed background filter), C8 versus C34 and equal-channel R134. Zero-padding C8 to34 is analytically unchanged. Select a reader only on fit20 and require an independent evaluation before native association integration.',
        acceptance='All399 missing-channel maps finite; old C8 parity<=1e-4; inputs and weights stable; no evaluation-label fitting.',
        sources={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(sources))})
    write('protocol.json',p)
    path=ROOT/'experiments/project_history.json';cat=read(path);assert not any(c['id']==OUT.name for c in cat['experiments'])
    cat['experiments'].append(dict(id=OUT.name,title='Восстановление26 отброшенных выходных карт полного декодера FlyVis',area='flyvis',state='running',started='2026-11-08',
        question=p['question'],hypothesis='Прежний восьмиканальный экспорт мог отбросить полезную пространственную информацию.',
        expected='Восстановить только отсутствующие26 карт и подтвердить совпадение готовых8; подготовить равный34-канальный контроль входа.',
        acceptance=[dict(criterion=p['acceptance'],met=None)],evidence=[dict(kind='file',ref=f'runs/{OUT.name}/protocol.json',claim='Постановка до восстановления')]))
    path.write_text(json.dumps(cat,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'dissertation/ЕДИНЫЙ_ПЛАН.md';text=path.read_text(encoding='utf8');now=datetime.now(timezone(timedelta(hours=3)))
    entry=f'**{now:%d.%m.%Y, %H:%M} MSK** — Подготовлено восстановление26 отброшенных стандартных выходных каналов.\n\n*Постановка.* Активная цель сохранения личности. После отрицательного гейта фонового фильтра проверяется потеря представления при экспорте8из34каналов. Входы всех399последовательностей восстановимы без видео:393рецепта и6сохранённых историй. R1проверяется по всем34готовым каналам доGPU. Во время восстановления требуется совпадение старых8до10⁻⁴; сохраняются только новые26. Это34входа штатного декодера, не полный перебор65типов; старые34-канальные опыты учтены. Карточка `{OUT.name}`, `runs/{OUT.name}/input_readiness.json`, `runs/{OUT.name}/protocol.json`.\n\n'
    path.write_text(text.replace('## Журнал изменений\n','## Журнал изменений\n\n'+entry,1),encoding='utf8')
    print('Ready',len(data),'sequences, R134 parity',max(errors.values()),flush=True)
def restore():
    p=read(OUT/'protocol.json');verify(p)
    if (OUT/'completed.json').exists():
        for path,digest in read(OUT/'completed.json')['files'].items():assert sha(ROOT/path)==digest
        return
    import psutil,torch
    from tracking.gpu_lock import GpuLock
    from scripts.z13_target_model import build_arm
    psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    normal=read(ROOT/'runs/z13_target_center_dev_20261018/C/normalization.json');mu=np.asarray(normal['mean'])[:34];sd=np.asarray(normal['sd'])[:34]
    folder=OUT/'maps';folder.mkdir(exist_ok=True);files={};max_error=0.;started=time.monotonic()
    with np.load(OUT/'inputs.npz',allow_pickle=False) as source,GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        _,_,bb=build_arm('C',torch,None);assert bb.dev.type=='cuda' and not bb.net.training
        weights={k:v.detach().clone() for k,v in bb.net.named_parameters()}
        for number,key in enumerate(p['requests']):
            dest=folder/(key+'.npz');receipt=folder/(key+'.json')
            if receipt.exists():
                r=read(receipt);assert sha(dest)==r['sha256'];max_error=max(max_error,r['old_c8_error']);files[dest.relative_to(ROOT).as_posix()]=r['sha256'];continue
            x=torch.from_numpy(source[key][None].astype(np.float32)).to(bb.dev);state=bb.init(x[:,:1])
            for t in range(1,x.shape[1]):state,f=bb.step(state,x[:,t-1],x[:,t])
            full=((f[0].cpu().numpy().T.astype(float)-mu)/sd).astype(np.float32)
            with np.load(BASE/'features'/(key+'.npz'),allow_pickle=False) as old:error=float(np.max(abs(full[:,4:12]-old['C'])))
            assert error<=1e-4,(key,error)
            other=full[:,OTHER];assert other.shape==(721,26) and np.isfinite(other).all()
            with dest.open('xb') as out:np.savez_compressed(out,C_OTHER=other)
            r=dict(sha256=sha(dest),old_c8_error=error,protocol_sha256=sha(OUT/'protocol.json'))
            with receipt.open('x',encoding='utf8') as out:json.dump(r,out,indent=2)
            files[dest.relative_to(ROOT).as_posix()]=r['sha256'];max_error=max(max_error,error)
            del x,state,f,full,other
            if number%10==0:print('restored',number+1,'/',len(p['requests']),'maxC8error',max_error,flush=True)
            time.sleep(.05)
        assert all(torch.equal(v,weights[k]) for k,v in bb.net.named_parameters())
    verify(p);write('completed.json',dict(passed=True,files=files,sequences=len(files),old_c8_max_error=max_error,weights_unchanged=True,
        source_hashes_stable=True,seconds=time.monotonic()-started));print('Complete',len(files),'old C8 error',max_error,flush=True)

{'prepare':prepare,'restore':restore}[sys.argv[1]]()
