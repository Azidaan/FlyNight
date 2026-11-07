"""Third discriminator: original predictor API on CPU, unchanged loss tolerance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'b2_predictor_cpu_profile'
    p=service.read(folder/'protocol.json')
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    assert service.sha(source)==p['source_sha256']
    from tracking.b2_cpu_loader_recovery3 import load_cpu_branches
    code=source.read_text(encoding='utf-8').replace("folder=RUN/'b2_sparse_training_audit'", "folder=RUN/'b2_predictor_cpu_profile'")
    code=code.replace("models=load_branches(state,'cpu')", "models=load_cpu_branches(state)")
    start="    detector=YOLO(str(ROOT/'weights/baseline_seed42/best.pt')).model.cpu().float().eval()\n    detector.fuse(verbose=False);detector.requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"
    replacement="""    yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
    first=service.read(ROOT/tp['entries'][0]['manifest'])
    cap=cv2.VideoCapture(str(ROOT/first['video_path']));ok,warm=cap.read();cap.release();assert ok
    for _ in range(2):yolo.predict(warm,imgsz=960,conf=.01,iou=.7,device='cpu',verbose=False)
    predictor=yolo.predictor;detector=predictor.model.model
    detector.eval().float().requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"""
    assert code.count(start)==1;code=code.replace(start,replacement)
    line="                tensor=torch.from_numpy(np.ascontiguousarray(value[:,:,::-1].transpose(2,0,1))[None]).to(device='cpu',dtype=torch.float32)/255"
    assert code.count(line)==1;code=code.replace(line,"                tensor=predictor.preprocess([image])")
    code=code.replace('completed=[];maximum=0.;compared=0','completed=[];maximum=0.;compared=0;diagnostics=[]')
    anchor="                            assert diff<=p['loss_absolute_tolerance'],(e['video'],frame,name,value,ref,diff)"
    assert code.count(anchor)==1
    code=code.replace(anchor,"                            diagnostics.append(dict(frame=frame,arm=name,value=value,reference=ref,difference=diff,within_original_tolerance=diff<=p['loss_absolute_tolerance']))")
    anchor='        if vi==0:\n';assert code.count(anchor)==1
    code=code.replace(anchor,"        if vi==0:\n            service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,scientific_verdict=None,device='cpu',losses=diagnostics,maximum_difference=maximum,all_within_original_tolerance=all(x['within_original_tolerance'] for x in diagnostics),original_tolerance=p['loss_absolute_tolerance'],no_cuda_initialized=not torch.cuda.is_initialized(),updated=service.now()))\n            return\n")
    ns={'__name__':'b2_cpu_predictor_diagnostic','__file__':str(source),'load_cpu_branches':load_cpu_branches}
    exec(compile(code,str(source),'exec'),ns);ns['main']()

if __name__=='__main__':main()
