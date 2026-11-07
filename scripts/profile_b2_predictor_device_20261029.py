"""Second discriminator: original predictor setup/preprocessing, frozen weights."""
from pathlib import Path
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']='0'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    from scripts import profile_b2_loss_device_20261029 as profile
    old=profile.ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    original=Path.read_text
    def read_text(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        if path==old:
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
    # Redirect only this new execution; old source and old report stay intact.
    profile.RUN=profile.RUN  # service functions still use the same global run.
    original_read=profile.service.read
    original_write=profile.service.write
    def redirect(path):
        return Path(str(path).replace('b2_loss_device_profile','b2_predictor_device_profile'))
    profile.service.read=lambda path:original_read(redirect(path))
    profile.service.write=lambda path,*a,**kw:original_write(redirect(path),*a,**kw)
    Path.read_text=read_text
    try:profile.main()
    finally:
        Path.read_text=original_read_text

original_read_text=Path.read_text
if __name__=='__main__':main()
