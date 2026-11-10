# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import ctypes
import ctypes.wintypes as wintypes
import logging
import os

import uuid

from agent_os.platform.proc_runner import (
    WindowsJob,
    keep_log,
    kill_process_tree,
    read_bounded,
)
from agent_os.platform.types import (
    CommandResult,
    ProcessHandle,
    SANDBOX_PASSWORD_KEY,
    SANDBOX_USERNAME,
    windows_worker_env,
)
from agent_os.platform.windows.credentials import CredentialStore

logger = logging.getLogger("agent_os.platform.windows.process")

# ---- Win32 constants ----

LOGON_WITH_PROFILE = 0x00000001
LOGON_NETCREDENTIALS_ONLY = 0x00000002
LOGON_NO_PROFILE = 0x00000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_UNICODE_ENVIRONMENT = 0x00000400
WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000
INFINITE = 0xFFFFFFFF
CREATE_NO_WINDOW = 0x08000000
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016
CTRL_BREAK_EVENT = 1
STILL_ACTIVE = 259

# ---- ctypes structures ----


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", STARTUPINFOW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class COORD(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


# ---- Win32 API references ----

advapi32 = ctypes.windll.advapi32
kernel32 = ctypes.windll.kernel32

_CreateProcessWithLogonW = advapi32.CreateProcessWithLogonW
_CreateProcessWithLogonW.restype = wintypes.BOOL
_CreateProcessWithLogonW.argtypes = [
    wintypes.LPCWSTR,   # lpUsername
    wintypes.LPCWSTR,   # lpDomain
    wintypes.LPCWSTR,   # lpPassword
    wintypes.DWORD,     # dwLogonFlags
    wintypes.LPCWSTR,   # lpApplicationName
    wintypes.LPWSTR,    # lpCommandLine
    wintypes.DWORD,     # dwCreationFlags
    ctypes.c_void_p,    # lpEnvironment
    wintypes.LPCWSTR,   # lpCurrentDirectory
    ctypes.POINTER(STARTUPINFOW),       # lpStartupInfo
    ctypes.POINTER(PROCESS_INFORMATION),  # lpProcessInformation
]

_WaitForSingleObject = kernel32.WaitForSingleObject
_WaitForSingleObject.restype = wintypes.DWORD
_WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]

_GetExitCodeProcess = kernel32.GetExitCodeProcess
_GetExitCodeProcess.restype = wintypes.BOOL
_GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]

_TerminateProcess = kernel32.TerminateProcess
_TerminateProcess.restype = wintypes.BOOL
_TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]

_CloseHandle = kernel32.CloseHandle
_CloseHandle.restype = wintypes.BOOL
_CloseHandle.argtypes = [wintypes.HANDLE]

_GenerateConsoleCtrlEvent = kernel32.GenerateConsoleCtrlEvent
_GenerateConsoleCtrlEvent.restype = wintypes.BOOL
_GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]

_GetLastError = ctypes.GetLastError

_CreatePipe = kernel32.CreatePipe
_CreatePipe.restype = wintypes.BOOL
_CreatePipe.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),  # hReadPipe
    ctypes.POINTER(wintypes.HANDLE),  # hWritePipe
    ctypes.c_void_p,                  # lpPipeAttributes
    wintypes.DWORD,                   # nSize
]

_CreateProcessW = kernel32.CreateProcessW
_CreateProcessW.restype = wintypes.BOOL
_CreateProcessW.argtypes = [
    wintypes.LPCWSTR,                         # lpApplicationName
    wintypes.LPWSTR,                          # lpCommandLine
    ctypes.c_void_p,                          # lpProcessAttributes
    ctypes.c_void_p,                          # lpThreadAttributes
    wintypes.BOOL,                            # bInheritHandles
    wintypes.DWORD,                           # dwCreationFlags
    ctypes.c_void_p,                          # lpEnvironment
    wintypes.LPCWSTR,                         # lpCurrentDirectory
    ctypes.POINTER(STARTUPINFOW),             # lpStartupInfo
    ctypes.POINTER(PROCESS_INFORMATION),      # lpProcessInformation
]

_CreatePseudoConsole = kernel32.CreatePseudoConsole
_CreatePseudoConsole.restype = wintypes.LONG  # HRESULT
_CreatePseudoConsole.argtypes = [
    COORD,                            # size (passed by value)
    wintypes.HANDLE,                  # hInput
    wintypes.HANDLE,                  # hOutput
    wintypes.DWORD,                   # dwFlags
    ctypes.POINTER(wintypes.HANDLE),  # phPC
]

_ClosePseudoConsole = kernel32.ClosePseudoConsole
_ClosePseudoConsole.restype = None
_ClosePseudoConsole.argtypes = [wintypes.HANDLE]

_ResizePseudoConsole = kernel32.ResizePseudoConsole
_ResizePseudoConsole.restype = wintypes.LONG  # HRESULT
_ResizePseudoConsole.argtypes = [wintypes.HANDLE, COORD]

_InitializeProcThreadAttributeList = kernel32.InitializeProcThreadAttributeList
_InitializeProcThreadAttributeList.restype = wintypes.BOOL
_InitializeProcThreadAttributeList.argtypes = [
    ctypes.c_void_p,                  # lpAttributeList
    wintypes.DWORD,                   # dwAttributeCount
    wintypes.DWORD,                   # dwFlags
    ctypes.POINTER(ctypes.c_size_t),  # lpSize
]

_UpdateProcThreadAttribute = kernel32.UpdateProcThreadAttribute
_UpdateProcThreadAttribute.restype = wintypes.BOOL
_UpdateProcThreadAttribute.argtypes = [
    ctypes.c_void_p,    # lpAttributeList
    wintypes.DWORD,     # dwFlags
    ctypes.c_size_t,    # Attribute
    ctypes.c_void_p,    # lpValue (pointer to attribute value)
    ctypes.c_size_t,    # cbSize
    ctypes.c_void_p,    # lpPreviousValue
    ctypes.c_void_p,    # lpReturnSize
]

_DeleteProcThreadAttributeList = kernel32.DeleteProcThreadAttributeList
_DeleteProcThreadAttributeList.restype = None
_DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]


_STILL_ACTIVE = 259


def _close_process_handles(handle: ProcessHandle) -> None:
    """Close a launched process's handles (a finished run used to leak them).

    Each stored value is cleared as it is closed, so the helper is
    idempotent: after ``CloseHandle`` the OS may reuse the number for an
    unrelated object (the daemon's kill-on-close job handle among them), and
    a later wait or second close on it would act on that object. The exit
    code is recorded first so ``wait()`` can still answer afterwards.
    """
    natives = handle._native_handles
    proc_h = natives.get("process_handle")
    if proc_h and "exit_code" not in natives:
        code = wintypes.DWORD()
        if _GetExitCodeProcess(proc_h, ctypes.byref(code)) and code.value != _STILL_ACTIVE:
            natives["exit_code"] = code.value
    for key in ("process_handle", "thread_handle"):
        native = natives.get(key)
        natives[key] = None
        if native:
            _CloseHandle(native)


def _launched_native_handles(process_h, thread_h, **extra) -> dict:
    """The ``_native_handles`` dict for a launched process.

    ``is_alive`` reads the handle from the dict at call time (not a captured
    number), so once the handle is closed it answers False without touching
    Win32. The lambda references the dict, not the ProcessHandle wrapper.
    """
    natives = {"process_handle": process_h, "thread_handle": thread_h, **extra}
    natives["is_alive"] = lambda: _process_alive(natives)
    return natives


def _process_alive(natives: dict) -> bool:
    proc_h = natives.get("process_handle")
    if not proc_h:
        return False
    return _WaitForSingleObject(proc_h, 0) == WAIT_TIMEOUT


# ---- ProcessLauncher ----


class ProcessLauncher:
    """Launches processes as the sandbox user via CreateProcessWithLogonW."""

    def __init__(self, credential_store: CredentialStore) -> None:
        self._credential_store = credential_store

    def launch(
        self,
        command: str,
        args: list[str],
        working_dir: str,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> ProcessHandle:
        """Launch a command as the sandbox user.

        Args:
            command: Executable path or name.
            args: Command arguments.
            working_dir: Working directory for the process.
            env_vars: Extra environment variables to set.
            inherit_env: If True, start from os.environ then overlay env_vars.

        Returns:
            ProcessHandle with pid and native handles.

        Raises:
            RuntimeError: If password is missing or process creation fails.
        """
        password = self._credential_store.retrieve(SANDBOX_PASSWORD_KEY)
        if password is None:
            raise RuntimeError(
                "Sandbox user password not found in credential store"
            )

        if not os.path.isdir(working_dir):
            raise RuntimeError(
                f"Working directory does not exist: {working_dir}"
            )

        cmd_line = self._build_command_line(command, args)
        cmd_line_buf = ctypes.create_unicode_buffer(cmd_line)

        env_block = self._build_env_block(env_vars, inherit_env)

        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(STARTUPINFOW)
        pi = PROCESS_INFORMATION()

        creation_flags = CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW

        logger.info("Launching process as %s: %s", SANDBOX_USERNAME, cmd_line)

        success = _CreateProcessWithLogonW(
            SANDBOX_USERNAME,
            ".",
            password,
            LOGON_NO_PROFILE,
            None,
            cmd_line_buf,
            creation_flags,
            env_block,
            working_dir,
            ctypes.byref(si),
            ctypes.byref(pi),
        )

        if not success:
            error_code = _GetLastError()

            # ERROR_DIRECTORY (267): sandbox user cannot resolve the working
            # directory despite valid ACLs — often caused by stale profile
            # state after password reset.  Retry with system temp as cwd.
            if error_code == 267:
                logger.warning(
                    "CreateProcessWithLogonW ERROR_DIRECTORY (267) for "
                    "working_dir=%s — retrying with system temp", working_dir
                )
                fallback_dir = os.path.join(
                    os.environ.get("SystemRoot", r"C:\Windows"), "Temp"
                )
                si = STARTUPINFOW()
                si.cb = ctypes.sizeof(STARTUPINFOW)
                pi = PROCESS_INFORMATION()
                success = _CreateProcessWithLogonW(
                    SANDBOX_USERNAME, ".", password, LOGON_NO_PROFILE,
                    None, cmd_line_buf, creation_flags, env_block,
                    fallback_dir, ctypes.byref(si), ctypes.byref(pi),
                )
                if not success:
                    fallback_err = _GetLastError()
                    raise RuntimeError(
                        f"CreateProcessWithLogonW failed with error code "
                        f"{error_code}, fallback also failed with {fallback_err}"
                    )
                logger.info(
                    "Fallback succeeded: launched in %s instead of %s",
                    fallback_dir, working_dir,
                )
            else:
                raise RuntimeError(
                    f"CreateProcessWithLogonW failed with error code {error_code}"
                )

        handle = ProcessHandle(
            pid=pi.dwProcessId,
            command=cmd_line,
            _native_handles=_launched_native_handles(pi.hProcess, pi.hThread),
        )

        logger.info("Process launched: pid=%d, command=%s", handle.pid, cmd_line)
        return handle

    def launch_with_pty(
        self,
        command: str,
        args: list[str],
        working_dir: str,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> ProcessHandle:
        """Launch a command as the CURRENT user with ConPTY attached.

        Unlike launch() which uses CreateProcessWithLogonW as sandbox user,
        this uses CreateProcessW as the current user with a pseudo-console.
        """
        import io
        import msvcrt

        if not os.path.isdir(working_dir):
            raise RuntimeError(f"Working directory does not exist: {working_dir}")

        cmd_line = self._build_command_line(command, args)
        cmd_line_buf = ctypes.create_unicode_buffer(cmd_line)

        # ConPTY runs as current user (not sandbox) — preserve user's TEMP/TMP
        # and set TERM so Node.js TUI libraries recognise the pseudo-terminal.
        if env_vars is None:
            env_vars = {}
        else:
            env_vars = dict(env_vars)  # don't mutate caller's dict
        env_vars.setdefault("TERM", "xterm-256color")
        env_vars.setdefault("TEMP", os.environ.get("TEMP", ""))
        env_vars.setdefault("TMP", os.environ.get("TMP", ""))

        env_block = self._build_env_block(env_vars, inherit_env)

        # Step 1: Create pipes
        pty_in_read = wintypes.HANDLE()
        pty_in_write = wintypes.HANDLE()
        pty_out_read = wintypes.HANDLE()
        pty_out_write = wintypes.HANDLE()

        if not _CreatePipe(ctypes.byref(pty_in_read), ctypes.byref(pty_in_write), None, 0):
            raise RuntimeError(f"CreatePipe (pty_in) failed: {_GetLastError()}")
        if not _CreatePipe(ctypes.byref(pty_out_read), ctypes.byref(pty_out_write), None, 0):
            _CloseHandle(pty_in_read)
            _CloseHandle(pty_in_write)
            raise RuntimeError(f"CreatePipe (pty_out) failed: {_GetLastError()}")

        # Step 2: Create pseudo console
        hpc = wintypes.HANDLE()
        size = COORD(80, 25)
        hr = _CreatePseudoConsole(size, pty_in_read, pty_out_write, 0, ctypes.byref(hpc))
        if hr != 0:
            _CloseHandle(pty_in_read)
            _CloseHandle(pty_in_write)
            _CloseHandle(pty_out_read)
            _CloseHandle(pty_out_write)
            raise RuntimeError(f"CreatePseudoConsole failed: HRESULT 0x{hr:08x}")

        # Step 3: Close pipe ends owned by ConPTY
        _CloseHandle(pty_in_read)
        _CloseHandle(pty_out_write)

        # Step 4: Build STARTUPINFOEXW with attribute list
        attr_size = ctypes.c_size_t(0)
        _InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(attr_size))
        attr_buf = (ctypes.c_byte * attr_size.value)()
        attr_list = ctypes.cast(attr_buf, ctypes.c_void_p)

        if not _InitializeProcThreadAttributeList(attr_list, 1, 0, ctypes.byref(attr_size)):
            _CloseHandle(pty_in_write)
            _CloseHandle(pty_out_read)
            _ClosePseudoConsole(hpc)
            raise RuntimeError(f"InitializeProcThreadAttributeList failed: {_GetLastError()}")

        # The Win32 API requires lpValue to be a *pointer to* the HPCON
        # handle, not the handle value itself.  UpdateProcThreadAttribute
        # reads cbSize bytes from the address lpValue points to.
        # Passing the HPCON directly causes it to read internal ConPTY
        # data instead of the handle, silently failing to attach the
        # pseudo console to the child process.
        hpc_ref = ctypes.c_void_p(hpc.value)
        if not _UpdateProcThreadAttribute(
            attr_list, 0,
            PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
            ctypes.byref(hpc_ref), ctypes.sizeof(ctypes.c_void_p),
            None, None,
        ):
            _DeleteProcThreadAttributeList(attr_list)
            _CloseHandle(pty_in_write)
            _CloseHandle(pty_out_read)
            _ClosePseudoConsole(hpc)
            raise RuntimeError(f"UpdateProcThreadAttribute failed: {_GetLastError()}")

        siex = STARTUPINFOEXW()
        siex.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
        siex.lpAttributeList = attr_list.value if isinstance(attr_list, ctypes.c_void_p) else attr_list

        pi = PROCESS_INFORMATION()

        creation_flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT

        logger.info("Launching process with ConPTY (current user): %s", cmd_line)

        # Step 5: CreateProcessW -- current user, no credentials
        #
        # ConPTY note: dwFlags MUST remain 0 — do NOT set STARTF_USESTDHANDLES.
        # PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE alone handles all three std handles.
        # Setting STARTF_USESTDHANDLES would override the pseudo console with
        # explicit pipe handles, making stdin a pipe (not TTY) to the child.
        #
        # Pass a pointer to the full STARTUPINFOEXW (not just the embedded
        # STARTUPINFOW sub-struct).  ctypes.byref(siex.StartupInfo) may return
        # a view whose buffer boundary stops at sizeof(STARTUPINFOW), hiding
        # lpAttributeList from CreateProcessW.  Casting through pointer()
        # guarantees the full structure is visible.
        siex_ptr = ctypes.cast(
            ctypes.pointer(siex), ctypes.POINTER(STARTUPINFOW)
        )
        success = _CreateProcessW(
            None,
            cmd_line_buf,
            None,
            None,
            False,
            creation_flags,
            env_block,
            working_dir,
            siex_ptr,
            ctypes.byref(pi),
        )

        _DeleteProcThreadAttributeList(attr_list)

        if not success:
            error_code = _GetLastError()
            _CloseHandle(pty_in_write)
            _CloseHandle(pty_out_read)
            _ClosePseudoConsole(hpc)
            raise RuntimeError(f"CreateProcessW failed with error code {error_code}")

        # Step 6: Wrap pipe handles as Python file objects
        # msvcrt.open_osfhandle transfers ownership -- closing FileIO closes the HANDLE
        stdin_fd = msvcrt.open_osfhandle(pty_in_write.value, 0)
        stdout_fd = msvcrt.open_osfhandle(pty_out_read.value, os.O_RDONLY)

        stdin_file = io.FileIO(stdin_fd, mode='wb', closefd=True)
        stdout_file = io.FileIO(stdout_fd, mode='rb', closefd=True)

        # Step 7: Build ProcessHandle
        handle = ProcessHandle(
            pid=pi.dwProcessId,
            command=cmd_line,
            stdin=stdin_file,
            stdout=stdout_file,
            stderr=None,  # ConPTY merges stderr into stdout
            _native_handles=_launched_native_handles(pi.hProcess, pi.hThread, hpc=hpc),
        )

        logger.info("ConPTY process launched: pid=%d, command=%s", handle.pid, cmd_line)
        return handle

    def is_running(self, handle: ProcessHandle) -> bool:
        """Check whether the process is still running (False once closed)."""
        return _process_alive(handle._native_handles)

    def terminate(self, handle: ProcessHandle, timeout_sec: int = 10) -> bool:
        """Terminate a process, trying graceful shutdown first.

        Sends CTRL_BREAK_EVENT and waits up to *timeout_sec* seconds.
        If the process does not exit, it is forcefully killed via
        TerminateProcess. Handles are closed before returning.

        Returns True when the process has been stopped. A second call (the
        handles are already closed) is a no-op.
        """
        if not handle._native_handles.get("process_handle"):
            return True
        if not self.is_running(handle):
            logger.info("Process pid=%d already exited, closing handles", handle.pid)
            _close_process_handles(handle)
            return True

        logger.info("Sending CTRL_BREAK_EVENT to pid=%d", handle.pid)
        _GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, handle.pid)

        result = _WaitForSingleObject(
            handle._native_handles["process_handle"], timeout_sec * 1000
        )

        if result == WAIT_TIMEOUT:
            logger.warning(
                "Process pid=%d did not exit gracefully, terminating", handle.pid
            )
            _TerminateProcess(handle._native_handles["process_handle"], 1)
            _WaitForSingleObject(handle._native_handles["process_handle"], 5000)

        _close_process_handles(handle)

        logger.info("Process pid=%d terminated and handles closed", handle.pid)
        return True

    def wait(self, handle: ProcessHandle, timeout_sec: int | None = None) -> int | None:
        """Wait for the process to exit and return its exit code.

        Args:
            handle: Process handle from launch().
            timeout_sec: Maximum seconds to wait (None = infinite).

        Returns:
            The process exit code. Once the handles are closed, the code
            recorded when they were closed, or None if none was recorded
            (never an invented success).

        Raises:
            TimeoutError: If the process does not exit within the timeout.
        """
        if not handle._native_handles.get("process_handle"):
            return handle._native_handles.get("exit_code")

        timeout_ms = INFINITE if timeout_sec is None else timeout_sec * 1000

        result = _WaitForSingleObject(handle._native_handles["process_handle"], timeout_ms)
        if result == WAIT_TIMEOUT:
            raise TimeoutError(
                f"Process pid={handle.pid} did not exit within {timeout_sec}s"
            )

        exit_code = wintypes.DWORD()
        _GetExitCodeProcess(handle._native_handles["process_handle"], ctypes.byref(exit_code))
        logger.info(
            "Process pid=%d exited with code %d", handle.pid, exit_code.value
        )
        return exit_code.value

    def run_and_capture(
        self,
        command: str,
        args: list[str],
        working_dir: str,
        env_vars: dict[str, str] | None = None,
        timeout_sec: int = 300,
        inherit_env: bool = True,
    ) -> CommandResult:
        """Launch a command as the sandbox user and capture stdout/stderr via temp files.

        Returns a CommandResult with exit_code, stdout, stderr, and timed_out flag.

        Spec 134 A1/W5: the process goes into a Job Object (no kill-on-close:
        a normal exit leaves whatever it started running), and a timeout
        kills the whole tree — job, ``cmd``, and the psutil-listed
        descendants — instead of only ``cmd``, which left the script running
        and writing the temp file. The files are read back bounded (head and
        tail), and a truncated one is kept under the workspace's
        shell-output folder.
        """
        from agent_os.agent.project_paths import ProjectPaths
        paths = ProjectPaths(working_dir)
        tmp_dir = paths.tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

        run_id = uuid.uuid4().hex[:12]
        stdout_path = os.path.join(tmp_dir, f"cmd_{run_id}_stdout.txt")
        stderr_path = os.path.join(tmp_dir, f"cmd_{run_id}_stderr.txt")

        # Build the original command string with args
        original_args_joined = " ".join(
            f'"{a}"' if " " in a else a for a in args
        )
        # Explicitly cd to the working directory before running the command.
        # CreateProcessWithLogonW with LOGON_NO_PROFILE unreliably sets cwd;
        # the ERROR_DIRECTORY (267) fallback may land us in C:\Windows\Temp.
        wrapped_inner = f'cd /d "{working_dir}" && {command} {original_args_joined} > "{stdout_path}" 2> "{stderr_path}"'

        timed_out = False
        exit_code = 1
        stdout_text = ""
        stderr_text = ""
        kept: set[str] = set()

        try:
            handle = self.launch(
                command="cmd",
                args=["/c", wrapped_inner],
                working_dir=working_dir,
                env_vars=env_vars,
                inherit_env=inherit_env,
            )
            job = WindowsJob.create(kill_on_close=False)
            if job is not None:
                job.assign(handle._native_handles["process_handle"])

            try:
                try:
                    exit_code = self.wait(handle, timeout_sec=timeout_sec)
                except TimeoutError:
                    timed_out = True
                    logger.info("pid=%d timed out after %ss; killing its tree",
                                handle.pid, timeout_sec)
                    kill_process_tree(handle.pid, job=job)
                    _TerminateProcess(handle._native_handles["process_handle"], 1)
                    _WaitForSingleObject(handle._native_handles["process_handle"], 5000)
            finally:
                if job is not None:
                    job.close()
                _close_process_handles(handle)

            # Read captured output, bounded.
            for path in (stdout_path, stderr_path):
                out = read_bounded(path)
                text = out.render(None)
                if out.truncated:
                    saved = keep_log(path, paths.shell_output_dir)
                    kept.add(path)
                    text = out.render(saved)
                if path == stdout_path:
                    stdout_text = text
                else:
                    stderr_text = text

        finally:
            # Clean up temp files (a survivor may still hold one; it stays).
            for path in (stdout_path, stderr_path):
                if path in kept:
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass

        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            timed_out=timed_out,
        )

    # ---- private helpers ----

    @staticmethod
    def _build_command_line(command: str, args: list[str]) -> str:
        """Build a single command-line string, quoting paths with spaces."""
        parts: list[str] = []
        if " " in command:
            parts.append(f'"{command}"')
        else:
            parts.append(command)
        for arg in args:
            if " " in arg:
                parts.append(f'"{arg}"')
            else:
                parts.append(arg)
        return " ".join(parts)

    @staticmethod
    def _build_env_block(
        env_vars: dict[str, str] | None,
        inherit_env: bool,
    ) -> ctypes.Array:
        """Build a Unicode environment block for CreateProcessWithLogonW.

        Format: KEY1=VALUE1\\0KEY2=VALUE2\\0\\0 (double-null terminated).
        """
        env: dict[str, str] = {}
        if inherit_env:
            env.update(os.environ)
        if env_vars:
            env.update(env_vars)

        # Strip CLAUDECODE env var to prevent "nested session" errors
        # when the daemon is launched from within a Claude Code session.
        env.pop("CLAUDECODE", None)

        # Spec 077 W2 — point the worker at a home it actually owns
        # (``C:\ProgramData\Orbital\worker``), replacing the C:\Temp /
        # C:\Windows\Temp scheme.
        #
        # Two things were broken before. TEMP/TMP aimed at the Windows public
        # temp directory, but APPDATA and LOCALAPPDATA were *inherited from the
        # daemon* and therefore pointed into the main user's profile — which is
        # owner-only, so every tool that writes a cache there (npm, pip, uv,
        # pnpm) failed on write. And CreateProcessWithLogonW with
        # LOGON_NO_PROFILE skips profile load, so without an explicit
        # HOMEDRIVE/HOMEPATH the worker cannot resolve the working directory
        # (ERROR_DIRECTORY 267).
        #
        # PATH is deliberately left inherited: it is what makes the W1
        # toolchain grants discoverable.
        if inherit_env:
            explicit = env_vars or {}
            for key, value in windows_worker_env().items():
                if key not in explicit:
                    env[key] = value

        parts = [f"{k}={v}" for k, v in sorted(env.items())]
        block = "\0".join(parts) + "\0\0"
        return ctypes.create_unicode_buffer(block)
