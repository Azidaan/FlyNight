"""Diagnose fixed secondary-peak discrepancy using the saved observed map only."""
from pathlib import Path
import os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from scripts.z13_descriptor_parity_cpu_audit_20261024 import RUN as OLD,CACHE,read,write,sha,now,journal
RUN=CACHE/'peak_audit_cpu_20261024';T0=time.monotonic()


def prepare():
    assert not RUN.exists();assert read(OLD/'controller_result.json')['exit_code']==0
    auth=read(ROOT/'runs/z13_weekend_20261024/authorization.json')
    names=['scripts/z13_descriptor_peak_audit_20261024.py','tracking/z13_candidates.py',(OLD/'report.json').relative_to(ROOT).as_posix(),(OLD/'observed_response_maps.npz').relative_to(ROOT).as_posix()]
    write(RUN/'protocol.json',dict(id='z13_filter_descriptor_v1',created=now().isoformat(),device='cpu',budget_seconds=90,save_at=auth['save_at'],hard_stop=auth['hard_stop'],stop_request_path=(RUN/'stop.request').relative_to(ROOT).as_posix(),scope='Same failing observed-frame3case, no new forward pass or fitting. Find original cached peak indices from their local weighted centers on the saved observed response map; inspect local-max and separation margins. No efficacy or changed criteria.',sha256={n:sha(ROOT/n) for n in names}))
    journal('проверка дискретного выбора вторичного пика','ПервыйCPUразбор завершён exit0 за78с: нормировка совпала точно,2D/3D и раскладка входа результата не меняют;7из8эпизодов воспроизводятся,в7128сохраняется отличие. Ещё одна короткая служебная проверка до90сCPU использует только уже сохранённую карту наблюдаемого кадра;муха повторно не прогоняется. Проверяет конкретно предикат локального максимума и расстояние подавления. Протокол peak_audit_cpu_20261024/protocol.json. Допуски не меняются, научный исход не вычисляется.')


def worker():
    import numpy as np,torch
    torch.cuda.is_available=lambda:False;torch.cuda.device_count=lambda:0;torch.set_num_threads(2);torch.set_default_device('cpu')
    from scripts import z13_target_model as M
    from tracking.z13_readout import to_frame
    from tracking.z13_candidates import candidates
    p=read(RUN/'protocol.json')
    for n,h in p['sha256'].items():assert sha(ROOT/n)==h
    write(RUN/'status.json',dict(state='running',stage='Геометрия и сохранённая карта',updated=now().isoformat()))
    old=read(OLD/'report.json');bank,geom,bb=M.build_arm('R1',torch,None);assert bb is None
    with np.load(OLD/'observed_response_maps.npz') as z:score=z['contiguous4_stored_2d']
    source=ROOT/old['source_cache'];assert sha(source)==old['source_cache_sha256']
    with np.load(source) as z:
        j=np.flatnonzero((z['episode']==7128)&(z['t']==3))[0];oldcenters=z['centers'][j];oldscore=z['scores'][j];n=int(z['count'][j])
    # Recover the crop from observed inputs. The video/episode offsets are fixed.
    base=0;crop=None
    for video in read(CACHE/'protocol.json')['videos']['fit84']:
        with np.load(ROOT/f'runs/z13_data/{video}.npz') as z:
            off=z['ep_offset']
            if base<=7128<base+len(off)-1:crop=z['fr_crop'][off[7128-base]+3].copy();break
            base+=len(off)-1
    assert crop is not None
    sd=float(score.std())+1e-9;centers=[]
    for i in range(len(score)):
        nb=np.unique(geom.neighbors[i]);w=np.exp((score[nb]-score[nb].max())/sd);center=(w[:,None]*geom.points[nb]).sum(0)/w.sum();centers.append(to_frame(center,crop,bank.eye_size))
    centers=np.asarray(centers);peaks=[]
    for k,c in enumerate(oldcenters[:n]):
        dist=np.linalg.norm(centers-c,axis=1);i=int(np.argmin(dist));nb=geom.neighbors[i];other=nb[nb!=i];local_margin=float(score[i]-score[other].max())
        peaks.append(dict(rank=k,index=i,reproduced_center=centers[i].tolist(),cached_center=c.tolist(),max_center_error=float(np.abs(centers[i]-c).max()),reproduced_score=float((score[i]-score.mean())/sd),cached_score=float(oldscore[k]),score_error=abs(float((score[i]-score.mean())/sd)-float(oldscore[k])),local_max_margin=local_margin,local_max_ok=bool(score[i]>=score[nb].max()),neighbors=other.tolist(),neighbors_margin=(score[i]-score[other]).tolist(),nearest_prior_distance=min(float(np.linalg.norm(geom.points[i]-geom.points[v['index']])) for v in peaks) if peaks else None,separation_threshold=2*geom.spacing))
    actual=candidates(score,np.zeros((len(score),68)),geom)
    write(RUN/'report.json',dict(source_audit_sha256=sha(OLD/'report.json'),actual_indices=actual['indices'].tolist(),original_peaks_on_same_map=peaks,all_original_peak_values_within_tolerance=all(x['max_center_error']<.5 and x['score_error']<.001 for x in peaks),hypothesis='A discrete peak eligibility predicate can flip under tiny numerical differences; verify local margins before attributing cause.',efficacy_not_computed=True))
    write(RUN/'status.json',dict(state='completed',stage='Предикаты исходных пиков проверены',updated=now().isoformat()))


if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    elif '--worker' in sys.argv:
        try:worker()
        except Exception as e:write(RUN/'status.json',dict(state='failed',stage=repr(e),updated=now().isoformat()));raise
    else:raise RuntimeError('Explicit mode required')
