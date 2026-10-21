"""Bounded R1 shape recovery; preserve old error and completed C/D/noop videos."""
from pathlib import Path
import gc
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import z13_center_hota_20261021 as science
from scripts import z13_center_hota_service_20261021 as service
RUN=science.RUN;CONTRACT=RUN/'recovery1_contract.json'


def prepare():
    if CONTRACT.exists():raise RuntimeError('Existing recovery contract')
    old=science.read(RUN/'probe_controller_result.json');assert old['worker_ended'] and old['exit_code']==1
    end=science.read(RUN/'controller.json');assert science.now().isoformat()<end['save_stop_at']
    files=['tracking/z13_ridge_localizer_v2.py','tests/test_z13_ridge_live_shape.py','scripts/z13_center_hota_recovery1_20261021.py']
    saved=[]
    for arm in ('C','D','noop'):
        video=science.read(RUN/'protocol.json')['probe_video'];saved.extend([f'runs/z13_center_hota_20261021/probe/{arm}/{video}/signatures.npz',f'runs/z13_center_hota_20261021/probe/{arm}/{video}/signatures.json'])
    science.write(CONTRACT,dict(created=science.now().isoformat(),original_service_contract_sha256=science.sha(RUN/'service_contract.json'),original_failure=old,original_save_stop_at=end['save_stop_at'],original_hard_stop_at=end['hard_stop_at'],change='Only remove the actual singleton retina channel for R1. C/D delegate unchanged; values,weights,normalization,lambda,threshold and data unchanged. Completed C/D/noop full videos reused and revalidated; old error/result/logs retained.',sha256={n:science.sha(ROOT/n) for n in files+saved},old_stderr_sha256=science.sha(RUN/'probe.stderr.log')))
    science.journal('проверяемое служебное восстановление формы R1','*Ошибка.* `runs/z13_center_hota_20261021/probe.stderr.log` и probe_controller_result.json: IndexError в R1, exit1. Живой BoxEye возвращает B,T,1,N, эпизодный загрузчик — B,T,N; дополнительная ось давала неверные оси EMA. Полные C/D и no-op видео уже сохранены и не повторяются.\n\n*Исправление.* Отдельный tracking/z13_ridge_localizer_v2.py убирает только ось размера1 у R1; для C/D вызовы исходных методов неизменны. Тест живой/эпизодной формы проверяет точное совпадение состояния, центров и баллов на6 последовательных шагах; неожиданные несколько каналов отклоняются. Сохранённые научные источники не переписаны. recovery1_contract.json связывает хеши служебной версии, старой ошибки и сохранённых видео. Повторяются только короткие проверки корректности, затем считается незавершённый R1/MASA/контроль сшивки; исходный срок пробы '+end['save_stop_at']+' не продлевается. Численного результата HOTA нет.')


def guard(p):
    contract=science.read(CONTRACT);assert science.sha(RUN/'service_contract.json')==contract['original_service_contract_sha256']
    for rel,digest in contract['sha256'].items():assert science.sha(ROOT/rel)==digest,rel
    if (RUN/'technical.json').exists() and ('--full-control' in sys.argv or '--stage' in sys.argv):assert science.read(RUN/'technical.json')['recovery_execution_contract_sha256']==science.sha(CONTRACT)
    return service.guard(p)


def probe():
    import numpy as np
    import torch
    from tracking.gpu_lock import GpuLock
    from tracking.z13_compiled_backend import enable
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    from scripts.run_bdd_video_rate_experiment import make_motion_estimator
    from scripts.build_fly_signatures import RenderOnlyBank
    from scripts.z13_train import Episodes
    torch.set_num_threads(2);p=science.read(RUN/'protocol.json');science.verify(p);data=Episodes(p['threshold_videos']);eps=sorted({l[0] for l in data.labels if not l[8]})[:8];maximum=[max(l[1] for l in data.labels if l[0]==e and not l[8]) for e in eps];repro={};timing={};video=p['probe_video']
    with GpuLock('flyvis_fovea_gpu'),torch.no_grad():
        for arm in p['arms']:
            science.status('Повторная сверка эпизодной/живой формы',variant=arm,done=len(repro),total=3)
            est=make_motion_estimator('flyvis_tiled',30.,.02);loc=RecoveredCenterRidgeLocalizer(RUN/arm/'config.json',RenderOnlyBank(est,max_batch=4));hx,ms=data.batch(torch,loc.dev,eps,maximum);mask=ms.cpu().numpy()
            eager=science.replay(loc,hx,mask,None,data,eps)
            if arm in ('C','D'):enable(loc)
            compiled=science.replay(loc,hx[:,:,None,:],mask,None,data,eps);saved={r['label']:r for r in science.read(ROOT/f'runs/z13_target_center_20261018/{arm}/center_rows.json')}
            dc=max(float(np.abs(r['center']-saved[i]['center']).max()) for i,r in compiled.items());ds=max(abs(r['score']-saved[i]['p_present']) for i,r in compiled.items());ce=max(float(np.abs(r['center']-eager[i]['center']).max()) for i,r in compiled.items());se=max(abs(r['score']-eager[i]['score']) for i,r in compiled.items());assert dc<.5 and ds<.001 and ce<.5 and se<.001
            assert all((r['score']>=loc.tau)==(saved[i]['p_present']>=loc.tau)==(eager[i]['score']>=loc.tau) for i,r in compiled.items());repro[arm]=dict(rows=len(compiled),max_center_px=dc,max_score=ds,compiler_and_shape_center_px=ce,compiler_and_shape_score=se,same_threshold_decisions=True)
            del loc,est,hx,ms;gc.collect();torch.cuda.empty_cache()
        science.write(RUN/'recovery1_reproduction.json',repro)
        with np.load(RUN/'probe/noop'/video/'signatures.npz') as a,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as b:assert np.array_equal(a['outputs'],b['outputs']) and int(a['frame_count'])==int(b['frame_count'])
        timing['noop_seconds']=science.read(RUN/'probe/noop'/video/'signatures.json')['seconds']
        for arm in p['arms']:
            path=RUN/'probe'/arm/video/'signatures.json'
            if not path.exists():science.status('Возобновление незавершённого полного видео',variant=arm,done=2,total=3);science.build(video,arm,RUN/'probe'/arm)
            meta=science.read(path)
            with np.load(path.with_suffix('.npz')) as actual,np.load(ROOT/'runs/fly_signatures'/video/'signatures.npz') as baseline:assert int(actual['frame_count'])==int(baseline['frame_count']) and np.isfinite(actual['outputs']).all()
            timing[arm]=dict(seconds=meta['seconds'],events=meta['z13_events'],reused_saved=arm in ('C','D'))
        from tracking.masa_r50 import build as masa_model
        from scripts.z13_shared_masa import extract_video
        science.status('MASA после восстановления R1',done=0,total=1);model=masa_model();paths={a:RUN/'probe'/a/video/'signatures.npz' for a in p['arms']};outputs={a:RUN/'probe/masa'/a/(video+'.npz') for a in p['arms']}
        timing['masa']=extract_video(model,ROOT/'data/bdd100k/videos/train'/(video+'.mov'),paths,outputs,check_first_frame=True);del model;gc.collect();torch.cuda.empty_cache()
    science.status('Проверка прежней сшивки35 после восстановления',done=0,total=35);science.environment_head();science.verify(p)
    science.write(RUN/'technical.json',dict(passed=True,reproduction=repro,no_intervention_outputs_bitwise_equal=True,baseline_stitch35_equal=True,timing=timing,rough_full_seconds=35*(sum(timing[a]['seconds'] for a in p['arms'])+timing['masa']['seconds']),caveat='One complete train video; C/D/noop saved inference time reused, initialization excluded. No HOTA; partial queue expected if window insufficient.',execution_service_contract_sha256=science.sha(RUN/'service_contract.json'),recovery_execution_contract_sha256=science.sha(CONTRACT),initial_whole_corpus_verified=service._probe_verified))


def main():
    if '--prepare-recovery' in sys.argv:prepare();return
    import tracking.z13_ridge_localizer as original
    from tracking.z13_ridge_localizer_v2 import RecoveredCenterRidgeLocalizer
    original.CenterRidgeLocalizer=RecoveredCenterRidgeLocalizer;science.verify=guard;science.probe=probe
    if '--full-control' in sys.argv:
        source=(ROOT/'scripts/z13_center_hota_controller_20261021.py').read_text(encoding='utf-8');old="str(ROOT/'scripts/z13_center_hota_20261021.py')";assert source.count(old)==1;source=source.replace(old,"str(ROOT/'scripts/z13_center_hota_recovery1_20261021.py')")
        assert source.count("RUN/'probe_controller_result.json'")==2;source=source.replace("RUN/'probe_controller_result.json'","RUN/'probe_recovery1_controller_result.json'")
        scope=dict(__name__='recovered_center_hota_controller',__file__=str(ROOT/'scripts/z13_center_hota_controller_20261021.py'));exec(compile(source,'recovered_center_hota_controller','exec'),scope);scope['main']();return
    science.main()


if __name__=='__main__':main()
