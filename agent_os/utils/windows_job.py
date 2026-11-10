# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker processes die with the daemon on Windows (batch 1110, item 14).

A hard daemon kill (``Stop-Process``, a crash, the installer) closes every
handle the daemon holds. On Windows a Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` turns that into a kill of every process
assigned to it, so a Claude Code / Codex / ACP worker in the middle of a turn
cannot outlive the daemon and race the restarted daemon's resumed work (the
2026-11-10 smoke saw an orphan ``claude.exe`` finish an import scan and write
the memory files 55 s after its daemon was gone). Children of a worker (its
shells) inherit the job, so the whole tree goes.

One job for the daemon's life, created on first use. Everything is best-effort
and never raises: off Windows, or when the OS refuses (a process already in a
job that forbids nesting), ``assign_worker`` returns False and the transport
carries on as before. ``ctypes`` only: no new dependency.
"""

from __future__ import annotations

import logging
import sys
import threading

logger = logging.getLogger(__name__)

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_lock = threading.Lock()
_job = None  # the HANDLE, held for the daemon's life (closing it is the kill)
_failed = False
_k32 = None


def _kernel32():
    global _k32
    if _k32 is None:
        import ctypes
        from ctypes import wintypes

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateJobObjectW.restype = wintypes.HANDLE
        k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k.SetInformationJobObject.restype = wintypes.BOOL
        k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        k.AssignProcessToJobObject.restype = wintypes.BOOL
        k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.IsProcessInJob.restype = wintypes.BOOL
        k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                     ctypes.POINTER(wintypes.BOOL)]
        k.CloseHandle.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        _k32 = k
    return _k32


def _limit_info():
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    return info


def _ensure_job():
    """The daemon's job handle, created once; None when it cannot be made."""
    global _job, _failed
    if _job is not None or _failed:
        return _job
    import ctypes
    k = _kernel32()
    job = k.CreateJobObjectW(None, None)
    if not job:
        _failed = True
        logger.warning("windows job: CreateJobObject failed (error %s); workers "
                       "will not be tied to the daemon", ctypes.get_last_error())
        return None
    info = _limit_info()
    if not k.SetInformationJobObject(job, JobObjectExtendedLimitInformation,
                                     ctypes.byref(info), ctypes.sizeof(info)):
        logger.warning("windows job: SetInformationJobObject failed (error %s)",
                       ctypes.get_last_error())
        k.CloseHandle(job)
        _failed = True
        return None
    _job = job
    return job


def assign_worker(pid: int | None) -> bool:
    """Tie the process ``pid`` to the daemon's kill-on-close job.

    True when it is now in the job. False off Windows, for a missing pid, or
    when the OS refuses; never raises.
    """
    if sys.platform != "win32" or not pid:
        return False
    try:
        with _lock:
            job = _ensure_job()
        if job is None:
            return False
        import ctypes
        k = _kernel32()
        h = k.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
        if not h:
            logger.info("windows job: cannot open pid %s (error %s)", pid,
                        ctypes.get_last_error())
            return False
        try:
            if k.AssignProcessToJobObject(job, h):
                return True
            logger.info("windows job: pid %s not assigned (error %s); it will not "
                        "die with the daemon", pid, ctypes.get_last_error())
            return False
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001 — a spawn must never fail on this
        logger.warning("windows job: assign failed for pid %s", pid, exc_info=True)
        return False


def in_daemon_job(pid: int) -> bool | None:
    """True/False when it can be told, None off Windows or on error (tests)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        with _lock:
            job = _ensure_job()
        if job is None:
            return None
        k = _kernel32()
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            out = wintypes.BOOL(0)
            if not k.IsProcessInJob(h, job, ctypes.byref(out)):
                return None
            return bool(out.value)
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001
        return None
