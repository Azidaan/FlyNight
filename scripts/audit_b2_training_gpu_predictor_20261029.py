"""Independent process GPU replay of all dev losses; CPU audit failure retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'b2_training_gpu_predictor_audit'
    p=service.read(folder/'protocol.json')
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    for f,h in p['sha256'].items():assert service.sha(ROOT/f)==h,f
    proof=service.read(RUN/'b2_predictor_device_profile_recovery1/report.json')
    assert proof['complete'] and proof['all_within_original_tolerance'] and len(proof['losses'])==60
    for name in ['b2_sparse_training_audit_recovery3','b2_predictor_device_profile_recovery1']:
        c=service.read(RUN/name/'controller_result.json')
        assert c['all_owned_ended'] and not any(service.matching(k,v) for k,v in c['owned'].items())
    code=source.read_text(encoding='utf-8')
    code=code.replace("os.environ['CUDA_VISIBLE_DEVICES']=''", "os.environ['CUDA_VISIBLE_DEVICES']='0'")
    code=code.replace("folder=RUN/'b2_sparse_training_audit'", "folder=RUN/'b2_training_gpu_predictor_audit'")
    code=code.replace("torch.set_default_device('cpu')", "torch.set_default_device('cuda')")
    code=code.replace("device='cpu'", "device='cuda'")
    code=code.replace("not torch.cuda.is_initialized()", "torch.cuda.is_initialized()")
    code=code.replace("models=load_branches(state,'cpu')", "models=load_branches(state,'cuda')")
    start="    detector=YOLO(str(ROOT/'weights/baseline_seed42/best.pt')).model.cpu().float().eval()\n    detector.fuse(verbose=False);detector.requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"
    replacement="""    yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
    first=service.read(ROOT/tp['entries'][0]['manifest'])
    warmcap=cv2.VideoCapture(str(ROOT/first['video_path']));ok,warm=warmcap.read();warmcap.release();assert ok
    for _ in range(2):yolo.predict(warm,imgsz=960,conf=.01,iou=.7,device='0',verbose=False)
    predictor=yolo.predictor;detector=predictor.model.model
    detector.eval().float().requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"""
    assert code.count(start)==1;code=code.replace(start,replacement)
    line="                tensor=torch.from_numpy(np.ascontiguousarray(value[:,:,::-1].transpose(2,0,1))[None]).to(device='cuda',dtype=torch.float32)/255"
    assert code.count(line)==1;code=code.replace(line,"                tensor=predictor.preprocess([image])")
    code=code.replace('if need>=available:', 'if False:  # Author cancelled stage time gates')
    code=code.replace('max_cpu_gpu_loss_difference=maximum','max_independent_gpu_loss_difference=maximum')
    code=code.replace('no_cuda_initialized=True','audit_device="cuda",cpu_gpu_equivalence_confirmed=False,original_cpu_audit_failed=True')
    # Never overwrite the original training-independent-audit path with another device.
    code=code.replace("service.write(source/'independent_audit.json',service.read(folder/'report.json'))", "service.write(source/'independent_gpu_audit.json',service.read(folder/'report.json'))")
    from tracking.gpu_lock import GpuLock
    with GpuLock('Local\\DissertationSingleGpu'):
        ns={'__name__':'b2_gpu_predictor_audit','__file__':str(source)}
        exec(compile(code,str(source),'exec'),ns);ns['main']()

if __name__=='__main__':main()
