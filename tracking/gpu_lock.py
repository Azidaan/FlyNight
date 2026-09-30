"""Межпроцессная блокировка участков работы с GPU (Windows named mutex).

30.09.2026: при нескольких CUDA-процессах одновременно на Windows процессы изредка навсегда
зависали в cuStreamSynchronize при простаивающем GPU (py-spy). Одиночный процесс не зависал
ни разу. Блокировка выстраивает обращения к GPU в очередь; CPU-часть работы (декодирование
видео, сетчатка, трекер) идёт параллельно. Если держатель мьютекса убит, Windows отдаёт
мьютекс следующему с кодом WAIT_ABANDONED — это считается успешным захватом.
"""

from __future__ import annotations

import ctypes
import os
import sys

WAIT_OBJECT_0, WAIT_ABANDONED, INFINITE = 0x0, 0x80, 0xFFFFFFFF


class GpuLock:
    def __init__(self, name: str | None) -> None:
        self.handle = None
        if name and sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
            self._k = kernel32
            self.handle = kernel32.CreateMutexW(None, False, name)
        self.waits = 0

    def __enter__(self):
        if self.handle:
            code = self._k.WaitForSingleObject(self.handle, INFINITE)
            if code not in (WAIT_OBJECT_0, WAIT_ABANDONED):
                raise OSError(f"WaitForSingleObject вернул {code:#x}")
            self.waits += 1
        return self

    def __exit__(self, *exc):
        if self.handle:
            self._k.ReleaseMutex(self.handle)
        return False


def from_env() -> GpuLock:
    return GpuLock(os.environ.get("FOVEA_GPU_LOCK"))
