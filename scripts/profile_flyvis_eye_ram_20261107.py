"""One-frame diagnostic only, with an OS-enforced private-memory ceiling."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from pathlib import Path
import ctypes as C
from ctypes import wintypes as W
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/flyvis_state_subgroups_20261107/memory_incident"
sys.path.insert(0,str(ROOT))
class BASIC(C.Structure):
    _fields_=[("PerProcessUserTimeLimit",C.c_int64),("PerJobUserTimeLimit",C.c_int64),("LimitFlags",W.DWORD),
              ("MinimumWorkingSetSize",C.c_size_t),("MaximumWorkingSetSize",C.c_size_t),("ActiveProcessLimit",W.DWORD),
              ("Affinity",C.c_size_t),("PriorityClass",W.DWORD),("SchedulingClass",W.DWORD)]
class IO(C.Structure):
    _fields_=[(k,C.c_uint64) for k in ("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]
class EXT(C.Structure):
    _fields_=[("BasicLimitInformation",BASIC),("IoInfo",IO),("ProcessMemoryLimit",C.c_size_t),("JobMemoryLimit",C.c_size_t),
              ("PeakProcessMemoryUsed",C.c_size_t),("PeakJobMemoryUsed",C.c_size_t)]
def main():
    import psutil
    assert psutil.virtual_memory().available>=10*1024**3, "Insufficient system headroom: no profile"
    kernel=C.WinDLL("kernel32",use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];kernel.CreateJobObjectW.restype=W.HANDLE
    kernel.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD];kernel.SetInformationJobObject.restype=W.BOOL
    kernel.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];kernel.AssignProcessToJobObject.restype=W.BOOL
    kernel.GetCurrentProcess.restype=W.HANDLE
    job=kernel.CreateJobObjectW(None,None);assert job,C.get_last_error()
    limit=EXT();limit.BasicLimitInformation.LimitFlags=0x200;limit.JobMemoryLimit=2*1024**3
    assert kernel.SetInformationJobObject(job,9,C.byref(limit),C.sizeof(limit)),C.get_last_error()
    assert kernel.AssignProcessToJobObject(job,kernel.GetCurrentProcess()),C.get_last_error()
    OUT.mkdir(exist_ok=True)
    (OUT/"one_frame_profile_started.json").write_text(json.dumps(dict(pid=os.getpid(),birth=psutil.Process().create_time(),job_memory_limit=limit.JobMemoryLimit,frames=1,shape=[391,391],stage="before_torch_import"))+"\n",encoding="utf-8")
    import torch
    torch.set_default_device("cpu");torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    from tracking.flyvis_adapter import activate_flyvis
    fv=activate_flyvis();fv.device=torch.device("cpu")
    from flyvis.datasets.rendering import BoxEye
    eye=BoxEye();eye.conv.double()
    before=psutil.Process().memory_info()._asdict();start=time.monotonic()
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True) as profile:
        result=eye(torch.zeros((1,1,391,391),dtype=torch.float64))
    assert result.shape==(1,1,1,721) and torch.isfinite(result).all() and not torch.cuda.is_initialized()
    events=sorted([dict(name=e.name,cpu_memory_usage=e.cpu_memory_usage,self_cpu_memory_usage=e.self_cpu_memory_usage,input_shapes=e.input_shapes)
                   for e in profile.events() if e.cpu_memory_usage>0],key=lambda e:e["cpu_memory_usage"],reverse=True)
    source=ROOT/"tools/flyvis-runtime/flyvis/datasets/rendering/eye.py"
    proof=dict(single_frame_only=True,job_memory_limit=limit.JobMemoryLimit,seconds=time.monotonic()-start,
        process_before=before,process_after=psutil.Process().memory_info()._asdict(),system_available=psutil.virtual_memory().available,
        positive_allocation_events=events,largest_single_allocation_bytes=events[0]["cpu_memory_usage"],
        predicted_103_frame_workspace_bytes=103*169*391*391*8,
        prediction_is_extrapolation_not_observed_peak=True,official_eye_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        method="Official BoxEye mean Conv2d, one391x391 frame CPU FP64; PyTorch CPU profiler. No core model, histories or scientific outputs recomputed.")
    (OUT/"one_frame_eye_memory.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(proof,ensure_ascii=False))
if __name__=="__main__":main()
