# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""MacOSPlatformProvider — Seatbelt-based process isolation for macOS."""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from agent_os.platform.base import PlatformProvider
from agent_os.platform.shared.network import NetworkProxy
from agent_os.platform.types import (
    CommandResult,
    DEFAULT_ALLOWLIST_DOMAINS,
    BlockedCallback,
    FolderInfo,
    NetworkRules,
    PermissionResult,
    PlatformCapabilities,
    ProcessHandle,
    SetupResult,
)
from agent_os.platform.macos.sandbox import generate_profile

logger = logging.getLogger("agent_os.platform.macos.provider")


class MacOSPlatformProvider(PlatformProvider):
    """macOS platform provider using Seatbelt (sandbox-exec) for process isolation."""

    def __init__(self, on_network_blocked: BlockedCallback | None = None) -> None:
        self._processes: dict[str, tuple[ProcessHandle, subprocess.Popen]] = {}
        self._proxies: dict[str, NetworkProxy] = {}
        self._pending_rules: dict[str, NetworkRules] = {}
        self._portal_paths: dict[str, str] = {}
        self._on_network_blocked = on_network_blocked
        self._caffeinate_proc: subprocess.Popen | None = None
        self._app_nap_activity = None
        self._workspace_base = str(
            Path.home() / "Library" / "Application Support" / "Orbital" / "workspace"
        )

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    async def setup(self) -> SetupResult:
        try:
            if not shutil.which("sandbox-exec"):
                return SetupResult(
                    success=False,
                    error="sandbox-exec not found — expected at /usr/bin/sandbox-exec",
                )
            os.makedirs(self._workspace_base, exist_ok=True)
            self._disable_app_nap()
            return SetupResult(success=True)
        except Exception as exc:
            logger.error("setup() failed: %s", exc)
            return SetupResult(success=False, error=str(exc))

    async def teardown(self) -> SetupResult:
        try:
            self._enable_app_nap()

            # Stop all proxies
            for project_id, proxy in list(self._proxies.items()):
                try:
                    await proxy.stop()
                except Exception:
                    logger.warning("Failed to stop proxy for project %s", project_id)
            self._proxies.clear()

            # Terminate all tracked processes
            for project_id, (handle, proc) in list(self._processes.items()):
                try:
                    for stream in (handle.stdin, handle.stdout, handle.stderr):
                        if stream and not getattr(stream, "closed", True):
                            try:
                                stream.close()
                            except Exception:
                                pass
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                except Exception:
                    logger.warning("Failed to terminate process for project %s", project_id)
            self._processes.clear()
            self._pending_rules.clear()

            return SetupResult(success=True)
        except Exception as exc:
            logger.error("teardown() failed: %s", exc)
            return SetupResult(success=False, error=str(exc))

    # ------------------------------------------------------------------
    # App Nap prevention
    # ------------------------------------------------------------------

    def _disable_app_nap(self):
        """Prevent macOS App Nap from throttling the daemon.

        App Nap aggressively throttles apps with no visible windows:
        timers deferred, network I/O deprioritized, CPU priority reduced.
        This uses Apple's NSProcessInfo API to declare latency-critical work.
        """
        if self._app_nap_activity is not None:
            return  # Already disabled

        try:
            from AppKit import NSProcessInfo

            # NSActivityUserInitiatedAllowingIdleSystemSleep = 0x00FFFFFF
            # Tells macOS this process is doing real work — do not throttle.
            self._app_nap_activity = (
                NSProcessInfo.processInfo()
                .beginActivityWithOptions_reason_(
                    0x00FFFFFF,
                    "Orbital daemon: maintaining agent connections and background tasks",
                )
            )
            logger.info("macOS App Nap disabled")
        except ImportError:
            logger.warning("AppKit not available — cannot disable App Nap")
        except Exception as e:
            logger.warning("Failed to disable App Nap: %s", e)

    def _enable_app_nap(self):
        """Re-enable App Nap (called during teardown)."""
        if self._app_nap_activity is None:
            return

        try:
            from AppKit import NSProcessInfo

            NSProcessInfo.processInfo().endActivity_(self._app_nap_activity)
            self._app_nap_activity = None
            logger.info("macOS App Nap re-enabled")
        except Exception as e:
            logger.warning("Failed to re-enable App Nap: %s", e)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def is_setup_complete(self) -> bool:
        return shutil.which("sandbox-exec") is not None

    def get_capabilities(self) -> PlatformCapabilities:
        setup_complete = self.is_setup_complete()
        issues: list[str] = []
        if not setup_complete:
            issues.append("sandbox-exec not found")
        return PlatformCapabilities(
            platform="macos",
            isolation_method="seatbelt",
            setup_complete=setup_complete,
            setup_issues=issues,
            supports_network_restriction=True,
            supports_folder_access=True,
            sandbox_username=None,  # Seatbelt doesn't use a separate user
        )

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

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

        # Ensure proxy is running for this project
        proxy = await self._ensure_proxy(project_id)

        # Generate Seatbelt profile
        profile = generate_profile(
            workspace_path=working_dir,
            portal_paths=self._portal_paths if self._portal_paths else None,
            network_proxy_port=proxy.port if proxy else None,
        )

        # Build environment with proxy settings
        env = os.environ.copy()
        if proxy:
            env["HTTP_PROXY"] = proxy.proxy_url
            env["HTTPS_PROXY"] = proxy.proxy_url
            env["NO_PROXY"] = "localhost,127.0.0.1"
        if extra_env:
            env.update(extra_env)

        cmd = ["sandbox-exec", "-p", profile, command] + args

        if use_pty:
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
        # Ensure proxy is running for this project
        proxy = await self._ensure_proxy(project_id)

        # Generate Seatbelt profile
        profile = generate_profile(
            workspace_path=working_dir,
            portal_paths=self._portal_paths if self._portal_paths else None,
            network_proxy_port=proxy.port if proxy else None,
        )

        # Build environment with proxy settings
        env = os.environ.copy()
        if proxy:
            env["HTTP_PROXY"] = proxy.proxy_url
            env["HTTPS_PROXY"] = proxy.proxy_url
            env["NO_PROXY"] = "localhost,127.0.0.1"
        if extra_env:
            env.update(extra_env)

        cmd = ["sandbox-exec", "-p", profile, command] + args

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

        # Close I/O streams
        for stream in (handle.stdin, handle.stdout, handle.stderr):
            if stream and not getattr(stream, "closed", True):
                try:
                    stream.close()
                except Exception:
                    pass

        # SIGTERM → wait → SIGKILL
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Stop the project's proxy
        proxy = self._proxies.pop(project_id, None)
        if proxy is not None:
            try:
                await proxy.stop()
            except Exception:
                logger.warning("Failed to stop proxy for project %s", project_id)

        return True

    # ------------------------------------------------------------------
    # Folder access
    # ------------------------------------------------------------------

    def grant_folder_access(
        self, path: str, mode: Literal["read_only", "read_write"]
    ) -> PermissionResult:
        """Grant folder access by adding to portal paths.

        Changes take effect on next process launch — Seatbelt profiles are
        compiled at process creation time.
        """
        self._portal_paths[path] = mode
        return PermissionResult(success=True, path=path)

    def revoke_folder_access(self, path: str) -> PermissionResult:
        """Revoke folder access by removing from portal paths.

        Changes take effect on next process launch — Seatbelt profiles are
        immutable after launch.
        """
        self._portal_paths.pop(path, None)
        return PermissionResult(success=True, path=path)

    def get_available_folders(self) -> list[FolderInfo]:
        # TODO: Validate TCC from .app bundle context during packaging task
        home = str(Path.home())
        well_known = [
            ("Desktop", "Desktop"),
            ("Documents", "Documents"),
            ("Downloads", "Downloads"),
            ("Developer", "Developer"),
            ("Movies", "Movies"),
            ("Music", "Music"),
            ("Pictures", "Pictures"),
            ("Public", "Public"),
        ]
        return [
            FolderInfo(
                path=os.path.join(home, dirname),
                display_name=display,
                accessible=True,
            )
            for dirname, display in well_known
        ]

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def configure_network(self, project_id: str, rules: NetworkRules) -> None:
        if project_id in self._proxies:
            self._proxies[project_id].set_rules(rules)
        # Store rules for deferred application when proxy starts
        self._pending_rules[project_id] = rules

    # ------------------------------------------------------------------
    # Sleep prevention
    # ------------------------------------------------------------------

    def prevent_sleep(self, reason: str) -> object:
        """Prevent system sleep using caffeinate."""
        self._caffeinate_proc = subprocess.Popen(["caffeinate", "-di"])
        logger.info("Sleep prevention enabled: %s", reason)
        return self._caffeinate_proc

    def allow_sleep(self, handle: object) -> None:
        """Re-allow system sleep by terminating the caffeinate process."""
        if handle is not None and hasattr(handle, "terminate"):
            handle.terminate()
            logger.info("Sleep prevention disabled")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_proxy(self, project_id: str) -> NetworkProxy:
        """Ensure a proxy is running for the given project, creating one if needed."""
        if project_id not in self._proxies:
            proxy = NetworkProxy(
                project_id=project_id,
                on_blocked=self._on_network_blocked,
            )

            # Apply pending rules if any, otherwise use default allowlist
            if project_id in self._pending_rules:
                proxy.set_rules(self._pending_rules[project_id])
            else:
                proxy.set_rules(NetworkRules(
                    mode="allowlist",
                    domains=list(DEFAULT_ALLOWLIST_DOMAINS),
                ))

            await proxy.start()
            self._proxies[project_id] = proxy

        return self._proxies[project_id]
