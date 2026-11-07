"""CPU-only, restricted unpickling inspection; never permits arbitrary globals."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json, time

def main():
    import torch
    torch.set_default_device('cpu');torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/continuation_20261028/reid_candidates_preparation'
    asset=json.loads((out/'download_status.json').read_text())
    assert asset['state']=='complete'
    started=time.monotonic()
    try:
        checkpoint=torch.load(asset['path'],map_location='cpu',weights_only=True)
        state=checkpoint.get('state_dict',checkpoint)
        keys={k:list(v.shape) for k,v in state.items() if isinstance(v,torch.Tensor)}
        groups={name:sum(v.numel() for k,v in state.items() if isinstance(v,torch.Tensor) and k.startswith(name))
                for name in ('detector.backbone.','backbone.','masa_adapter.','track_head.')}
        report=dict(passed=True,restricted_weights_only=True,sha256=asset['sha256'],
                    cuda_initialized=torch.cuda.is_initialized(),parameter_groups=groups,tensor_shapes=keys,
                    elapsed_seconds=time.monotonic()-started,inference_executed=False,scientific_verdict=None)
        assert not report['cuda_initialized']
    except Exception as error:
        report=dict(passed=False,restricted_weights_only=True,error=repr(error),
                    unsafe_unpickling_not_attempted=True,inference_executed=False,scientific_verdict=None)
    (out/'checkpoint_inspection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='tensor_shapes'},ensure_ascii=False))

if __name__=='__main__':main()
