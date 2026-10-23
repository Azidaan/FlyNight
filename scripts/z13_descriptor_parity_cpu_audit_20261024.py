"""Fixed-case CPU-only service audit, no training, no change to acceptance."""
from pathlib import Path
import os,sys,time
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts.z13_filter_descriptor_cache_20261023 import RUN as CACHE,SPLITS,read,write,sha,now,journal
RUN=CACHE/'parity_audit_cpu_20261024';T0=time.monotonic();EPISODE=7128;SPLIT='fit84';ARM='D'


def status(stage,**kw):
    p=read(RUN/'protocol.json')
    if (RUN/'stop.request').exists() or time.monotonic()-T0>=p['budget_seconds'] or now()>=datetime.fromisoformat(p['save_at']):raise TimeoutError('CPU audit bounded deadline')
    write(RUN/'status.json',dict(state='running',stage=stage,updated=now().isoformat(),elapsed_seconds=round(time.monotonic()-T0,2),**kw))


def prepare():
    assert not RUN.exists();auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    files=['scripts/z13_descriptor_parity_cpu_audit_20261024.py','scripts/z13_weekend_progress.ps1','scripts/z13_target_model.py','tracking/z13_filter_descriptor.py','tracking/z13_target_center.py','tracking/z13_candidates.py','runs/z13_filter_descriptor_cache_20261023/parity_failure.json']
    old=read(CACHE/'protocol.json');p=dict(id=old['id'],device='cpu',created=now().isoformat(),budget_seconds=300,save_at=auth['save_at'],hard_stop=auth['hard_stop'],stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),scope='Single preidentified D/fit84/7128 failed observed-frame3 group. Compare CPU repeat, original tensor prefix layout,2Dvs3D ridge evaluation and independently recomputed old CPU normalizer. Observed inputs0..3 only enter feature evaluation; labels never used for efficacy. No GPU,training,threshold/tolerance/data changes. Does not rescue failed acceptance by discarding candidates.',sha256={n:sha(ROOT/n) for n in files},original_protocol_sha256=sha(CACHE/'protocol.json'))
    write(RUN/'protocol.json',p);journal('короткийCPUразбор воспроизведения модели цели','Отдельный протокол runs/z13_filter_descriptor_cache_20261023/parity_audit_cpu_20261024/protocol.json. Предварительно выбран единственный уже провалившийся D/fit84/эпизод7128: центр19,113пикселя,балл0,250649 при неизменных допусках0,5/0,001. До300сCPU,никакогоCUDA/обучения. Проверяются порядок численных операций,раскладка наблюдаемых входов и нормировка;результат определяет служебное исправление. Старые источники/ошибка/готовые587блоков сохраняются. Нового научного результата пока нет.')


def worker():
    import numpy as np,torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_num_threads(2);torch.set_default_device('cpu')
    assert torch.ones(1,device='cpu').device.type=='cpu'
    from scripts import z13_target_model as old
    from scripts.z13_train import Episodes
    from tracking.z13_filter_descriptor import fit_descriptor
    from tracking.z13_target_center import observed_center_target,ridge_scores
    from tracking.z13_candidates import candidates
    from tracking.z13_readout import to_frame
    p=read(RUN/'protocol.json');assert sha(CACHE/'protocol.json')==p['original_protocol_sha256']
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h
    folder,part=SPLITS[SPLIT];found=[]
    for meta in sorted((folder/part/ARM).glob('cache_*.json')):
        m=read(meta)
        if EPISODE in m['episodes']:found.append((meta,m))
    assert len(found)==1;meta,m=found[0];source=meta.with_suffix('.npz');assert sha(source)==m['sha256'];ch=m['episodes'];refs={}
    with np.load(source) as z:
        length=int(z['t'].max())+1
        for j in np.flatnonzero(z['t']==3):refs[int(z['episode'][j])]={k:z[k][j].copy() for k in ('seed_features','centers','scores','count')}
    assert sorted(refs)==sorted(ch);data={};base=0;hashes={};videos=read(CACHE/'protocol.json')['videos'][SPLIT]
    for v in videos:
        path=ROOT/f'runs/z13_data/{v}.npz'
        with np.load(path) as z:
            off=z['ep_offset'];needed=[e for e in ch if base<=e<base+len(off)-1]
            if needed:
                hashes[path.relative_to(ROOT).as_posix()]=sha(path)
                for e in needed:
                    i=e-base;data[e]=(z['fr_hex'][off[i]:off[i]+4].astype(np.float32),z['ex_mask'][i].copy(),z['fr_crop'][off[i]+3].copy())
            base+=len(off)-1
    assert sorted(data)==sorted(ch);status('Сборка только CPU,фиксированная группа',done=0,total=4)
    ck=torch.load(ROOT/'runs/z13_spatial_v2_20261013/D/full/model.pt',map_location='cpu',weights_only=False);bank,geom,bb=old.build_arm(ARM,torch,ck);assert bb.dev.type=='cpu'
    normal=read(ROOT/f'runs/z13_target_center_dev_20261018/{ARM}/normalization.json');stored=(np.asarray(normal['mean']),np.asarray(normal['sd']));lam=read(ROOT/f'runs/z13_target_center_20261018/{ARM}/choice.json')['center']['lambda_value'];hx=np.stack([data[e][0] for e in ch]);dense=torch.as_tensor(hx,device='cpu');padded=torch.as_tensor(np.repeat(hx[:,-1:,:],length,axis=1),device='cpu');padded[:,:4]=dense
    features={};case_results={};maps={}
    for name,input in [('contiguous4',dense),('original_prefix_stride',padded[:,:4])]:
        status('Вычисление наблюдаемых4кадров,CPU',variant=name,done=len(features),total=4);features[name]=old.episode_features(torch,ARM,bb,input,geom.neighbors,{3})[3]
    status('Контроль штатной нормировки по200эпизодам прежних20train',done=2,total=4)
    stats=old.feature_stats(torch,ARM,bb,geom,Episodes(read(CACHE/'protocol.json')['videos']['fit20']))
    for name,f in features.items():
        for norm_name,(mu,sd) in [('stored',stored),('cpu_recomputed',stats)]:
            checks=[]
            for j,e in enumerate(ch):
                x=(f[j].T-mu)/sd;mask=data[e][1];y=observed_center_target(mask,geom.points,geom.spacing);w,m,b=fit_descriptor(x,y,lam)
                two=(x-m)@w+b;three=ridge_scores(x,y,x[None],[lam])[0,0]
                for shape,score in [('2d',two),('3d',three)]:
                    r=candidates(score,x,geom);n=len(r['indices']);ref=refs[e];centers=np.asarray([to_frame(c,data[e][2],bank.eye_size) for c in r['centers']]);same_count=n==int(ref['count']);dc=float(np.abs(centers-ref['centers'][:n]).max()) if same_count else None;ds=float(np.abs(r['scores']-ref['scores'][:n]).max()) if same_count else None
                    checks.append(dict(episode=e,shape=shape,seed_max_abs=float(np.abs(x[mask].mean(0)-ref['seed_features']).max()),center_max_px=dc,score_max_abs=ds,same_count=same_count,pass_original_candidate_tolerances=bool(same_count and dc<.5 and ds<.001),response_std=float(score.std()),top_margin=float(np.sort(score)[-1]-np.sort(score)[-2]),projection_shape_max_abs=float(np.abs(two-three).max()),indices=r['indices'].tolist()))
                    if e==EPISODE:maps[name+'_'+norm_name+'_'+shape]=score
            case_results[name+'_'+norm_name]=checks
    report=dict(episode=EPISODE,group=ch,source_cache=source.relative_to(ROOT).as_posix(),source_cache_sha256=sha(source),input_sha256=hashes,device=str(bb.dev),normalization_mean_max_abs=float(np.abs(stats[0]-stored[0]).max()),normalization_sd_max_abs=float(np.abs(stats[1]-stored[1]).max()),layout_feature_max_abs=float(np.abs(features['contiguous4']-features['original_prefix_stride']).max()),cases=case_results,efficacy_not_computed=True,original_tolerances_unchanged=True)
    np.savez_compressed(RUN/'observed_response_maps.npz',**maps);write(RUN/'report.json',report);write(RUN/'status.json',dict(state='completed',stage='Фиксированный CPUразбор сохранён',updated=now().isoformat(),done=4,total=4))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
