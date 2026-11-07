import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    import torch
    from tracking.masa_detic_prepared import build,official_swin_classes
    torch.set_default_device('cpu');torch.set_num_threads(1)
    started=time.monotonic();out=ROOT/'runs/continuation_20261028/reid_candidates_preparation'
    model=build('cpu')
    # A small deterministic technical input; no val tuning or scientific claim.
    image=torch.linspace(-1,1,3*96*128).reshape(1,3,96,128)
    boxes=torch.tensor([[0,0,32,32],[5,10,90,80],[0,0,128,96]],dtype=torch.float32)
    with torch.inference_mode():
        features=model.features(image)
        embedding=model.embed_boxes(features,boxes,1.)
        assert embedding.shape==(3,256) and torch.isfinite(embedding).all()
        assert model.embed_boxes(features,boxes[:0],1.).shape==(0,256)
        repeated=model.embed_boxes(features,boxes,1.)
        assert torch.equal(embedding,repeated)
    assert not torch.cuda.is_initialized()
    report=dict(passed=True,scope='Strict checkpoint loading and deterministic small CPU forward; NOT full official parity or HOTA',
        shapes=[list(x.shape) for x in features],embedding_shape=list(embedding.shape),
        parameter_count=sum(p.numel() for p in model.parameters()),cuda_initialized=False,
        elapsed_seconds=time.monotonic()-started,full_reference_parity=False,scientific_verdict=None)
    (out/'cpu_forward_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
