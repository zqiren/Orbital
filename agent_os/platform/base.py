# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Literal

from agent_os.platform.types import (
    CommandResult,
    FolderInfo,
    NetworkRules,
    PermissionResult,
    PlatformCapabilities,
    ProcessHandle,
    SetupResult,
)


class PlatformProvider(ABC):
    @abstractmethod
    async def setup(self) -> SetupResult: ...

    @abstractmethod
    async def teardown(self) -> SetupResult: ...

    @abstractmethod
    def is_setup_complete(self) -> bool: ...

    @abstractmethod
    def get_capabilities(self) -> PlatformCapabilities: ...

    @abstractmethod
    async def run_process(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        extra_env: dict[str, str] | None = None,
        use_pty: bool = False,
    ) -> ProcessHandle: ...

    @abstractmethod
    async def run_command(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        timeout_sec: int = 300,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult: ...

    @abstractmethod
    async def stop_process(self, project_id: str, timeout_sec: int = 10) -> bool:
        """Stop the agent process for a project. Returns True if process was running and stopped."""
        ...

    @abstractmethod
    def grant_folder_access(
        self, path: str, mode: Literal["read_only", "read_write"]
    ) -> PermissionResult: ...

    @abstractmethod
    def revoke_folder_access(self, path: str) -> PermissionResult: ...

    @abstractmethod
    def get_available_folders(self) -> list[FolderInfo]: ...

    @abstractmethod
    def configure_network(self, project_id: str, rules: NetworkRules) -> None: ...

    def prevent_sleep(self, reason: str) -> object:
        """Prevent the system from sleeping. Returns a handle for allow_sleep()."""
        return None

    def allow_sleep(self, handle: object) -> None:
        """Re-allow system sleep. Pass the handle from prevent_sleep()."""
