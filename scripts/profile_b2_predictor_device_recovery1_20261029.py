"""Correct profile isolation before second device discriminator."""
from pathlib import Path
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from scripts import profile_b2_loss_device_20261029 as profile
    folder=profile.RUN/'b2_predictor_device_profile_recovery1'
    old_result=profile.service.read(profile.RUN/'b2_predictor_device_profile/controller_result.json')
    assert old_result['all_owned_ended']
    assert not any(profile.service.matching(k,v) for k,v in old_result['owned'].items())
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    original_read=Path.read_text
    def read_text(path,*args,**kwargs):
        text=original_read(path,*args,**kwargs)
        if path==source:
            text=text.replace("folder=RUN/'b2_sparse_training_audit'", "folder=RUN/'b2_predictor_device_profile_recovery1'")
            start="    detector=YOLO(str(ROOT/'weights/baseline_seed42/best.pt')).model.cpu().float().eval()\n    detector.fuse(verbose=False);detector.requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"
            replacement="""    yolo=YOLO(str(ROOT/'weights/baseline_seed42/best.pt'))
    first=service.read(ROOT/tp['entries'][0]['manifest'])
    warmcap=cv2.VideoCapture(str(ROOT/first['video_path']));ok,warm=warmcap.read();warmcap.release();assert ok
    for _ in range(2):yolo.predict(warm,imgsz=960,conf=.01,iou=.7,device='0',verbose=False)
    predictor=yolo.predictor;detector=predictor.model.model
    detector.eval().float().requires_grad_(False);detector.args=get_cfg(overrides=detector.args)"""
            assert text.count(start)==1;text=text.replace(start,replacement)
            line="                tensor=torch.from_numpy(np.ascontiguousarray(value[:,:,::-1].transpose(2,0,1))[None]).to(device='cpu',dtype=torch.float32)/255"
            assert text.count(line)==1;text=text.replace(line,"                tensor=predictor.preprocess([image])")
        return text
    # Only initial profile protocol lookup still names the old folder.
    old_read=profile.service.read
    def protocol_read(path):
        if path==profile.RUN/'b2_loss_device_profile/protocol.json':path=folder/'protocol.json'
        return old_read(path)
    profile.service.read=protocol_read
    Path.read_text=read_text
    try:profile.main()
    finally:Path.read_text=original_read

if __name__=='__main__':main()
