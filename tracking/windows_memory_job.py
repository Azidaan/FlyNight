"""Keep a Windows process and its descendants within a private-memory job cap."""
import ctypes as C
from ctypes import wintypes as W
import os

class Basic(C.Structure):
    _fields_=[("PerProcessUserTimeLimit",C.c_int64),("PerJobUserTimeLimit",C.c_int64),
        ("LimitFlags",W.DWORD),("MinimumWorkingSetSize",C.c_size_t),("MaximumWorkingSetSize",C.c_size_t),
        ("ActiveProcessLimit",W.DWORD),("Affinity",C.c_size_t),("PriorityClass",W.DWORD),("SchedulingClass",W.DWORD)]
class IO(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]
class Extended(C.Structure):
    _fields_=[("BasicLimitInformation",Basic),("IoInfo",IO),("ProcessMemoryLimit",C.c_size_t),
        ("JobMemoryLimit",C.c_size_t),("PeakProcessMemoryUsed",C.c_size_t),("PeakJobMemoryUsed",C.c_size_t)]

class MemoryJob:
    def __init__(self,limit_bytes):
        if os.name!="nt" or not isinstance(limit_bytes,int) or limit_bytes<=0:
            raise ValueError("A positive Windows job-memory limit is required")
        k=C.WinDLL("kernel32",use_last_error=True)
        k.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];k.CreateJobObjectW.restype=W.HANDLE
        k.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD];k.SetInformationJobObject.restype=W.BOOL
        k.QueryInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD,C.c_void_p];k.QueryInformationJobObject.restype=W.BOOL
        k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];k.AssignProcessToJobObject.restype=W.BOOL
        k.GetCurrentProcess.restype=W.HANDLE
        self.kernel=k;self.handle=k.CreateJobObjectW(None,None)
        if not self.handle:raise OSError(C.get_last_error(),"CreateJobObjectW")
        info=Extended();info.BasicLimitInformation.LimitFlags=0x200;info.JobMemoryLimit=limit_bytes
        if not k.SetInformationJobObject(self.handle,9,C.byref(info),C.sizeof(info)):raise OSError(C.get_last_error(),"SetInformationJobObject")
        if not k.AssignProcessToJobObject(self.handle,k.GetCurrentProcess()):raise OSError(C.get_last_error(),"AssignProcessToJobObject")
        self.limit_bytes=limit_bytes
        assert self.read()["limit_bytes"]==limit_bytes
        # Keep the handle alive for the process lifetime; no destructor that
        # would change the cap while a child is still running.
    def read(self):
        info=Extended()
        if not self.kernel.QueryInformationJobObject(self.handle,9,C.byref(info),C.sizeof(info),None):
            raise OSError(C.get_last_error(),"QueryInformationJobObject")
        return dict(limit_bytes=info.JobMemoryLimit,limit_active=bool(info.BasicLimitInformation.LimitFlags&0x200),
                    peak_job_bytes=info.PeakJobMemoryUsed,peak_process_bytes=info.PeakProcessMemoryUsed)
