# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import io
import os
import subprocess
import sys
from typing import Literal

from agent_os.platform.base import PlatformProvider
from agent_os.platform.types import (
    CommandResult,
    FolderInfo,
    NetworkRules,
    PermissionResult,
    PlatformCapabilities,
    ProcessHandle,
    SetupResult,
)


class NullProvider(PlatformProvider):
    """Functional provider that runs processes without isolation (dev/testing)."""

    def __init__(self) -> None:
        self._processes: dict[str, tuple[ProcessHandle, subprocess.Popen]] = {}

    async def setup(self) -> SetupResult:
        return SetupResult(success=True)

    async def teardown(self) -> SetupResult:
        for project_id in list(self._processes):
            await self.stop_process(project_id)
        return SetupResult(success=True)

    def is_setup_complete(self) -> bool:
        return True

    def get_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform="null",
            isolation_method="none",
            setup_complete=True,
            setup_issues=[],
            supports_network_restriction=False,
            supports_folder_access=False,
            sandbox_username=None,
        )

    async def run_process(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        extra_env: dict[str, str] | None = None,
        use_pty: bool = False,
    ) -> ProcessHandle:
        # Stop any existing process for this project
        if project_id in self._processes:
            await self.stop_process(project_id)

        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        cmd = [command] + args
        use_real_pty = use_pty and sys.platform != "win32"

        if use_real_pty:
            import pty

            master_fd, slave_fd = pty.openpty()
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=working_dir,
                    env=env,
                )
            finally:
                os.close(slave_fd)

            stdout_stream: io.RawIOBase = os.fdopen(master_fd, "rb", buffering=0)  # type: ignore[assignment]
            stdin_stream: io.RawIOBase = os.fdopen(os.dup(master_fd), "wb", buffering=0)  # type: ignore[assignment]

            handle = ProcessHandle(
                pid=proc.pid,
                command=command,
                stdin=stdin_stream,
                stdout=stdout_stream,
                stderr=None,
                _native_handles={"is_alive": lambda: proc.poll() is None},
            )
        else:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            handle = ProcessHandle(
                pid=proc.pid,
                command=command,
                stdin=proc.stdin,
                stdout=proc.stdout,
                stderr=proc.stderr,
                _native_handles={"is_alive": lambda: proc.poll() is None},
            )

        self._processes[project_id] = (handle, proc)
        return handle

    async def run_command(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        timeout_sec: int = 300,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        cmd = [command] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=working_dir,
                env=env,
                timeout=timeout_sec,
            )
            return CommandResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(exit_code=1, stdout="", stderr="", timed_out=True)

    async def stop_process(self, project_id: str, timeout_sec: int = 10) -> bool:
        entry = self._processes.pop(project_id, None)
        if entry is None:
            return False
        handle, proc = entry
        for stream in (handle.stdin, handle.stdout, handle.stderr):
            if stream and not getattr(stream, "closed", True):
                try:
                    stream.close()
                except Exception:
                    pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
        return True

    def grant_folder_access(
        self, path: str, mode: Literal["read_only", "read_write"]
    ) -> PermissionResult:
        return PermissionResult(
            success=False, path=path, error="No platform provider available"
        )

    def revoke_folder_access(self, path: str) -> PermissionResult:
        return PermissionResult(
            success=False, path=path, error="No platform provider available"
        )

    def get_available_folders(self) -> list[FolderInfo]:
        return []

    def configure_network(self, project_id: str, rules: NetworkRules) -> None:
        pass

    def prevent_sleep(self, reason: str) -> object:
        return None

    def allow_sleep(self, handle: object) -> None:
        pass
