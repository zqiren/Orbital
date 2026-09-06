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
    windows_protected_control_files,
    windows_toolchain_roots,
    windows_worker_home,
)
from agent_os.utils.subprocess_flags import win_no_window_flags

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

    # ------------------------------------------------------------------
    # Spec 077 W1/W2/W3 — toolchain grants, worker home, deny ACEs
    # ------------------------------------------------------------------

    def deny_access(self, username: str, path: str) -> PermissionResult:
        """Add a deny-write ACE for *username* on *path* (spec 077 W3).

        Deny ACEs precede allows in Windows access evaluation, so the
        read-write grant on the enclosing workspace cannot override this. The
        inheritance flags ``(OI)(CI)`` are only valid on a directory, so a file
        target (``.git\\config``) gets the bare ``W`` right.

        Consequence, same as Claude Code documents: ``git remote add`` from
        inside the sandbox fails, because it writes ``.git\\config``. The agent
        passes the URL on the command line instead.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")

        flags = "(OI)(CI)W" if os.path.isdir(resolved) else "W"
        args = [resolved, "/deny", f"{username}:{flags}", "/Q"]
        if os.path.isdir(resolved):
            args.insert(-1, "/T")

        result = self._run_icacls(args)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error("icacls deny failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Denied write for %s on %s", username, resolved)
        return PermissionResult(success=True, path=resolved)

    def remove_deny(self, username: str, path: str) -> PermissionResult:
        """Drop the deny ACE added by :meth:`deny_access`.

        Teardown must call this: a deny ACE left on the user's own repository
        after uninstall outlives the account it names (spec 077 §8).
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")

        args = [resolved, "/remove:d", username, "/Q"]
        if os.path.isdir(resolved):
            args.insert(-1, "/T")

        result = self._run_icacls(args)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.warning("icacls remove:d failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)
        return PermissionResult(success=True, path=resolved)

    def has_deny(self, username: str, path: str) -> bool:
        """True when *username* already carries a deny ACE on *path*."""
        resolved = self._resolve_path(path)
        if resolved is None:
            return False
        result = self._run_icacls([resolved])
        if result.returncode != 0:
            return False
        return _has_deny_ace(result.stdout, username)

    def protect_control_files(self, username: str, root: str) -> list[PermissionResult]:
        """Apply the W3 deny ACEs to every control file that exists under *root*.

        Idempotent and cheap: an ``icacls`` query per candidate, and a write
        only when the ACE is missing. Windows ACLs need a real path — there is
        no Seatbelt-style pattern for a not-yet-existing file — which is why
        this is re-run after commands that could have created a repository.
        """
        results: list[PermissionResult] = []
        for target in windows_protected_control_files(root):
            if not os.path.exists(target):
                continue
            if self.has_deny(username, target):
                continue
            results.append(self.deny_access(username, target))
        return results

    def unprotect_control_files(self, username: str, root: str) -> list[PermissionResult]:
        """Reverse :meth:`protect_control_files` (teardown)."""
        results: list[PermissionResult] = []
        for target in windows_protected_control_files(root):
            if os.path.exists(target):
                results.append(self.remove_deny(username, target))
        return results

    def find_repository_roots(self, root: str, max_depth: int = 3) -> list[str]:
        """Directories under *root* that contain a ``.git`` folder.

        Depth-bounded and skipping dependency trees, so the post-command
        re-check stays cheap on a large workspace.
        """
        skip = {"node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".git"}
        found: list[str] = []
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            return found
        for dirpath, dirnames, _ in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if os.path.isdir(os.path.join(dirpath, ".git")):
                found.append(dirpath)
            if depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip]
        return found

    def grant_toolchain_roots(self, username: str) -> list[PermissionResult]:
        """Grant read+execute on each per-user toolchain root that exists (W1).

        The user's profile as a whole stays closed: a Windows profile holds
        browsers, mail and vaults, and a blanket grant plus deny ACEs for
        secrets would be both invasive and hard to reason about. Non-elevated
        is enough here — the user owns these folders.
        """
        results: list[PermissionResult] = []
        for root in windows_toolchain_roots():
            if not os.path.isdir(root):
                continue
            resolved = self._resolve_path(root)
            if resolved is None:
                continue
            result = self._run_icacls(
                [resolved, "/grant", f"{username}:(OI)(CI)RX", "/T", "/Q"]
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                logger.warning("toolchain grant failed on %s: %s", resolved, err)
                results.append(PermissionResult(success=False, path=resolved, error=err))
            else:
                results.append(PermissionResult(success=True, path=resolved))
        return results

    def revoke_toolchain_roots(self, username: str) -> list[PermissionResult]:
        """Reverse :meth:`grant_toolchain_roots` (teardown)."""
        results: list[PermissionResult] = []
        for root in windows_toolchain_roots():
            if os.path.isdir(root):
                results.append(self.revoke_access(username, root))
        return results

    def setup_worker_home(self, username: str) -> PermissionResult:
        """Create the worker's own home under ProgramData and hand it over (W2)."""
        home = windows_worker_home()
        try:
            for sub in ("", "Temp", os.path.join("AppData", "Roaming"),
                        os.path.join("AppData", "Local")):
                os.makedirs(os.path.join(home, sub) if sub else home, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create worker home at %s: %s", home, exc)
            return PermissionResult(
                success=False, path=home, error=f"Failed to create worker home: {exc}"
            )
        return self.grant_access(username, home, "read_write")

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
            # icacls emits ANSI-localized text on non-English Windows, which
            # is invalid UTF-8 — strict decoding would kill the pipe reader
            # thread and hand back stdout=None (see test_subprocess_encoding).
            errors="replace",
            creationflags=win_no_window_flags(),
        )


def _has_deny_ace(output: str, username: str) -> bool:
    """True when *username* has a ``(DENY)`` entry in ``icacls`` output.

    icacls renders one as ``AgentOS-Worker:(DENY)(OI)(CI)(W)`` (or
    ``DESKTOP-ABC\\AgentOS-Worker:(DENY)(W)`` for a file).
    """
    username_lower = username.lower()
    for line in output.splitlines():
        line_lower = line.lower()
        if username_lower + ":" in line_lower and "(deny)" in line_lower:
            return True
    return False


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
        # A deny ACE (spec 077 W3) is not an access grant — reading it as one
        # would report read_write on a path the worker cannot write.
        if "(deny)" in line_lower:
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
