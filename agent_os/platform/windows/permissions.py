# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""C2: PermissionManager — Windows ACL management via icacls."""

import logging
import os
import re
import subprocess
from typing import Literal

from agent_os.platform.types import (
    AccessInfo,
    FolderInfo,
    PermissionResult,
    WORKSPACE_AGENT_DIR,
)

logger = logging.getLogger("agent_os.platform.windows.permissions")


class PermissionManager:
    """Manages file-system permissions for the sandbox user via icacls."""

    # Standard user folders returned by get_available_folders()
    _STANDARD_FOLDERS = [
        "Desktop",
        "Documents",
        "Downloads",
        "Pictures",
        "Videos",
        "Music",
    ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grant_access(
        self,
        username: str,
        path: str,
        mode: Literal["read_only", "read_write"],
    ) -> PermissionResult:
        """Grant *username* access to *path* with the specified *mode*."""
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")

        if mode == "read_only":
            perm = f"{username}:(OI)(CI)R"
        else:
            perm = f"{username}:(OI)(CI)F"

        result = self._run_icacls([resolved, "/grant", perm, "/T", "/Q"])
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error("icacls grant failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Granted %s access (%s) to %s", username, mode, resolved)
        return PermissionResult(success=True, path=resolved)

    def revoke_access(self, username: str, path: str) -> PermissionResult:
        """Revoke all access for *username* on *path*."""
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")

        result = self._run_icacls([resolved, "/remove", username, "/T", "/Q"])
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error("icacls revoke failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Revoked access for %s on %s", username, resolved)
        return PermissionResult(success=True, path=resolved)

    def check_access(self, username: str, path: str) -> AccessInfo:
        """Check what access *username* has on *path*."""
        resolved = self._resolve_path(path)
        if resolved is None:
            return AccessInfo(has_access=False, mode="none", path=path)

        result = self._run_icacls([resolved])
        if result.returncode != 0:
            logger.warning("icacls check failed on %s: %s", resolved, result.stderr.strip())
            return AccessInfo(has_access=False, mode="none", path=resolved)

        return _parse_icacls_output(result.stdout, username, resolved)

    def setup_workspace(self, username: str, workspace_path: str) -> PermissionResult:
        """Create the workspace directory structure and grant full control."""
        try:
            os.makedirs(workspace_path, exist_ok=True)
            agent_dir = os.path.join(workspace_path, WORKSPACE_AGENT_DIR)
            os.makedirs(agent_dir, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create workspace dirs at %s: %s", workspace_path, exc)
            return PermissionResult(
                success=False,
                path=workspace_path,
                error=f"Failed to create directories: {exc}",
            )

        grant_result = self.grant_access(username, workspace_path, "read_write")
        if not grant_result.success:
            return grant_result

        logger.info("Workspace set up for %s at %s", username, workspace_path)
        return PermissionResult(success=True, path=workspace_path)

    def get_available_folders(self) -> list[FolderInfo]:
        """Return standard user folders with accessibility info."""
        home = os.path.expanduser("~")
        folders: list[FolderInfo] = []

        for name in self._STANDARD_FOLDERS:
            folder_path = os.path.join(home, name)
            exists = os.path.isdir(folder_path)
            folders.append(
                FolderInfo(
                    path=folder_path,
                    display_name=name,
                    accessible=exists,
                    access_note=None if exists else "Folder does not exist",
                )
            )

        return folders

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path: str) -> str | None:
        """Resolve *path* to an absolute, symlink-free path.

        Returns ``None`` if the path does not exist.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return None
        return os.path.realpath(abs_path)

    @staticmethod
    def _run_icacls(args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run icacls with the given arguments."""
        cmd = ["icacls"] + args
        logger.debug("Running: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )


def _parse_icacls_output(output: str, username: str, path: str) -> AccessInfo:
    """Parse icacls output and determine the access level for *username*.

    icacls output example::

        C:\\Users\\dev\\project src\\main.py AgentOS-Worker:(OI)(CI)(F)
                                             BUILTIN\\Users:(OI)(CI)(RX)

    The username may appear with or without a domain prefix (e.g.
    ``DESKTOP-ABC\\AgentOS-Worker`` or just ``AgentOS-Worker``).
    """
    username_lower = username.lower()

    for line in output.splitlines():
        line_lower = line.lower()
        # Match "username:" accounting for optional DOMAIN\ prefix
        if username_lower + ":" not in line_lower:
            continue

        # Extract all permission flags in parentheses, e.g. (OI)(CI)(F)
        flags = re.findall(r"\(([^)]+)\)", line)
        flag_set = {f.upper() for f in flags}

        if "F" in flag_set:
            return AccessInfo(has_access=True, mode="read_write", path=path)
        if "R" in flag_set or "RX" in flag_set:
            return AccessInfo(has_access=True, mode="read_only", path=path)

        # Has an entry but no recognized read/write flag — treat as some access
        return AccessInfo(has_access=True, mode="read_only", path=path)

    return AccessInfo(has_access=False, mode="none", path=path)
