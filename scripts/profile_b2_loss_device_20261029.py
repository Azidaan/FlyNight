"""First complete dev video replay on original GPU; diagnostic, not audit acceptance."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    folder=RUN/'b2_loss_device_profile'
    p=service.read(folder/'protocol.json')
    source=ROOT/'scripts/audit_b2_sparse_training_20261029.py'
    assert service.sha(source)==p['source_sha256']
    old=RUN/'b2_sparse_training_audit_recovery3/controller_result.json'
    assert service.sha(old)==p['failed_cpu_audit_sha256']
    c=service.read(old)
    assert c['exit_code']==1 and c['all_owned_ended']
    assert not any(service.matching(k,v) for k,v in c['owned'].items())
    code=source.read_text(encoding='utf-8')
    code=code.replace("os.environ['CUDA_VISIBLE_DEVICES']=''", "os.environ['CUDA_VISIBLE_DEVICES']='0'")
    code=code.replace("folder=RUN/'b2_sparse_training_audit'", "folder=RUN/'b2_loss_device_profile'")
    code=code.replace("torch.set_default_device('cpu')", "torch.set_default_device('cuda')")
    code=code.replace("device='cpu'", "device='cuda'")
    code=code.replace("and not torch.cuda.is_initialized()", "and torch.cuda.is_initialized()")
    code=code.replace("models=load_branches(state,'cpu')", "models=load_branches(state,'cuda')")
    code=code.replace(".model.cpu().float().eval()", ".model.cuda().float().eval()")
    anchor="                            assert diff<=p['loss_absolute_tolerance'],(e['video'],frame,name,value,ref,diff)"
    assert code.count(anchor)==1
    code=code.replace(anchor,"                            diagnostics.append(dict(frame=frame,arm=name,value=value,reference=ref,difference=diff,within_original_tolerance=diff<=p['loss_absolute_tolerance']))")
    code=code.replace('completed=[];maximum=0.;compared=0', 'completed=[];maximum=0.;compared=0;diagnostics=[]')
    anchor="        if vi==0:\n"
    assert code.count(anchor)==1
    code=code.replace(anchor,"        if vi==0:\n            service.write(folder/'report.json',dict(complete=True,diagnostic_only=True,scientific_verdict=None,device='cuda',losses=diagnostics,maximum_difference=maximum,all_within_original_tolerance=all(x['within_original_tolerance'] for x in diagnostics),original_tolerance=p['loss_absolute_tolerance'],updated=service.now()))\n            return\n")
    from tracking.gpu_lock import GpuLock
    with GpuLock('Local\\DissertationSingleGpu'):
        ns={'__name__':'b2_device_diagnostic','__file__':str(source)}
        exec(compile(code,str(source),'exec'),ns);ns['main']()

if __name__=='__main__':main()
