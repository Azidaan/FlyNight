"""Prepare one real-frame GPU interface measurement; no tracker claim."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.continuation_service_20261028 import RUN,service

def main():
    old=ROOT/'scripts/check_masa_detic_bdd_input_20261029.py'
    script='scripts/check_masa_detic_gpu_input_20261029.py'
    code=old.read_text(encoding='utf-8')
    code=code.replace("os.environ['CUDA_VISIBLE_DEVICES'] = ''", "os.environ['CUDA_VISIBLE_DEVICES'] = '0'")
    code=code.replace("model = build('cpu')", "model = build('cuda')")
    code=code.replace('tensor, scale = preprocess_bdd(image)', "tensor, scale = preprocess_bdd(image, 'cuda')")
    code=code.replace('torch.as_tensor(boxes, dtype=torch.float32)', "torch.as_tensor(boxes, dtype=torch.float32, device='cuda')")
    code=code.replace('    started = time.monotonic()', '    torch.cuda.synchronize()\n    started = time.monotonic()')
    code=code.replace('    assert embedding.shape', '    torch.cuda.synchronize()\n    assert embedding.shape')
    code=code.replace('assert not torch.cuda.is_initialized()', 'assert torch.cuda.is_initialized()')
    code=code.replace('cuda_initialized=False', 'cuda_initialized=True')
    code=code.replace("target=out/'bdd_input_check.json'", "target=ROOT/'runs/continuation_20261028/masa_detic_gpu_input_profile/report.json'")
    code=code.replace("if __name__ == '__main__': main()", "if __name__ == '__main__':\n    from tracking.gpu_lock import GpuLock\n    with GpuLock('Local\\\\DissertationSingleGpu'): main()")
    compile(code,script,'exec');assert not (ROOT/script).exists();(ROOT/script).write_text(code,encoding='utf-8')
    folder=RUN/'masa_detic_gpu_input_profile';assert not folder.exists();folder.mkdir()
    service.write(folder/'protocol.json',dict(budget_seconds=86400,scope='Single existing train image, original trained model on GPU; synchronized time and finite embeddings only; no tracker/HOTA or CPU-GPU parity acceptance',sha256={s:service.sha(ROOT/s) for s in [script,'tracking/masa_detic_prepared.py','tracking/masa_detic_bdd_input.py']},scientific_verdict=None))
    service.write(RUN/'ready_jobs/153_masa_detic_gpu_input_profile.json',dict(name='153_masa_detic_gpu_input_profile',folder=folder.relative_to(ROOT).as_posix(),script=script,script_sha256=service.sha(ROOT/script),protocol_sha256=service.sha(folder/'protocol.json')))

if __name__=='__main__':main()
