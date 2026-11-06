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
import threading
import warnings

WAIT_OBJECT_0, WAIT_ABANDONED, WAIT_FAILED, INFINITE = 0x0, 0x80, 0xFFFFFFFF, 0xFFFFFFFF
CANONICAL_GPU_LOCK_NAME = r"Local\DissertationSingleGpu"
_KNOWN_GPU_LOCK_NAMES = frozenset((
    r"Local\DissertationSingleGpu",
    "flyvis_fovea_gpu",
    "FlyVis_z13_observed_memory_single_gpu",
))


def resolve_gpu_lock_name(name: str | None) -> str | None:
    """Map the three project aliases to one mutex; preserve custom/disabled names.

    Only newly loaded processes use this mapping. Existing old-code processes
    must finish or restart before cross-alias exclusion can be relied upon.
    """
    return CANONICAL_GPU_LOCK_NAME if name in _KNOWN_GPU_LOCK_NAMES else name


def _win_error(operation):
    error = ctypes.WinError(ctypes.get_last_error())
    raise OSError(f'{operation} failed: {error}') from error


class GpuLock:
    """Reusable, reentrant mutex; close an idle instance when its owner is done.

    Exiting a context releases ownership but retains the handle for EyeBank's
    next frame. Discarded idle instances close their handles automatically.
    Closing an acquired lock or one with waiting threads is rejected.
    """
    def __init__(self, name: str | None) -> None:
        self.handle = None
        self.name = name
        self.effective_name = resolve_gpu_lock_name(name)
        self.waits = 0
        self._guard = threading.Lock()
        self._depth = self._waiters = 0
        self._owner = None
        self._closed = False
        self._enabled = bool(name and sys.platform == 'win32')
        if self._enabled:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
            kernel32.ReleaseMutex.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle.restype = ctypes.c_int
            self._k = kernel32
            self.handle = kernel32.CreateMutexW(None, False, self.effective_name)
            if not self.handle:
                _win_error('CreateMutexW')

    def __enter__(self):
        if not self._enabled:
            return self
        with self._guard:
            if self._closed:
                raise RuntimeError('GpuLock is closed')
            self._waiters += 1
        try:
            code = self._k.WaitForSingleObject(self.handle, INFINITE)
        except BaseException:
            with self._guard:
                self._waiters -= 1
            raise
        with self._guard:
            self._waiters -= 1
            if code not in (WAIT_OBJECT_0, WAIT_ABANDONED):
                if code == WAIT_FAILED:
                    _win_error('WaitForSingleObject')
                raise OSError(f'WaitForSingleObject returned {code:#x}')
            owner = threading.get_ident()
            self._depth = self._depth + 1 if self._owner == owner and code != WAIT_ABANDONED else 1
            self._owner = owner
            self.waits += 1
        return self

    def __exit__(self, *exc):
        if not self._enabled:
            return False
        with self._guard:
            if self._owner != threading.get_ident() or not self._depth:
                raise RuntimeError('GpuLock must be released by its acquiring thread')
            if not self._k.ReleaseMutex(self.handle):
                _win_error('ReleaseMutex')
            self._depth -= 1
            if not self._depth:
                self._owner = None
        return False

    def close(self):
        if not self._enabled:
            return
        with self._guard:
            if self._depth or self._waiters:
                raise RuntimeError('Cannot close an acquired GpuLock or one with waiting threads')
            if self.handle:
                if not self._k.CloseHandle(self.handle):
                    _win_error('CloseHandle')
                self.handle = None
                self._closed = True

    def __del__(self, _warn=warnings.warn):
        if getattr(self, 'handle', None):
            try:
                self.close()
            except Exception as error:
                # Releasing another thread's mutex or closing a handle while a
                # wait is pending is unsafe. Keep that failure explicit.
                _warn(f'GpuLock handle was not closed: {error}', ResourceWarning)


def from_env() -> GpuLock:
    return GpuLock(os.environ.get("FOVEA_GPU_LOCK"))
