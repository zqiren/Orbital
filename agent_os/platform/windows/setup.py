# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""C6: SetupOrchestrator — first-run elevated setup and teardown."""

import ctypes
import json
import logging
import os
import sys
import time

from agent_os.platform.types import SetupResult, SetupStatus
from agent_os.platform.windows.permissions import PermissionManager
from agent_os.platform.windows.sandbox import SandboxAccountManager

logger = logging.getLogger("agent_os.platform.windows.setup")

_SETUP_STATUS_FILENAME = "setup_status.json"
_ELEVATION_POLL_INTERVAL = 0.5  # seconds
_ELEVATION_TIMEOUT = 60  # seconds


class SetupOrchestrator:
    """Manages the one-time elevated setup flow for Agent OS isolation."""

    def __init__(
        self,
        account_manager: SandboxAccountManager,
        permission_manager: PermissionManager,
    ) -> None:
        self.account_manager = account_manager
        self.permission_manager = permission_manager

    def check_setup_status(self) -> SetupStatus:
        """Check current setup state. Does NOT require elevation."""
        issues: list[str] = []

        account_status = self.account_manager.validate_account()
        if not account_status.exists:
            issues.append("Sandbox user 'AgentOS-Worker' does not exist")
        elif not account_status.password_valid:
            issues.append("Sandbox user password is invalid or missing")

        default_workspace = self._get_default_workspace_path()
        workspace_ready = os.path.isdir(default_workspace)
        if not workspace_ready:
            issues.append("Default workspace directory does not exist")

        return SetupStatus(
            is_complete=len(issues) == 0,
            sandbox_user_exists=account_status.exists,
            sandbox_password_valid=account_status.password_valid if account_status.exists else False,
            workspace_ready=workspace_ready,
            issues=issues,
        )

    def run_setup(self) -> SetupResult:
        """Run full setup. Must be called from an elevated process."""
        # Step 1: Create sandbox account
        try:
            account_status = self.account_manager.ensure_account_exists()
            if not account_status.exists:
                error = account_status.error or "Failed to create sandbox account"
                logger.error("Setup step 1 (account creation) failed: %s", error)
                return SetupResult(success=False, error=error)
            logger.info("Setup step 1 complete: sandbox account ready")
        except Exception as exc:
            logger.error("Setup step 1 (account creation) raised: %s", exc)
            return SetupResult(success=False, error=str(exc))

        # Step 2: Create default workspace directory
        workspace_path = self._get_default_workspace_path()
        try:
            os.makedirs(workspace_path, exist_ok=True)
            logger.info("Setup step 2 complete: workspace directory at %s", workspace_path)
        except OSError as exc:
            logger.error("Setup step 2 (workspace creation) failed: %s", exc)
            return SetupResult(success=False, error=f"Failed to create workspace: {exc}")

        # Step 3: Set workspace permissions
        try:
            username = self.account_manager.get_username()
            perm_result = self.permission_manager.setup_workspace(username, workspace_path)
            if not perm_result.success:
                error = perm_result.error or "Failed to set workspace permissions"
                logger.error("Setup step 3 (permissions) failed: %s", error)
                return SetupResult(success=False, error=error)
            logger.info("Setup step 3 complete: workspace permissions set")
        except Exception as exc:
            logger.error("Setup step 3 (permissions) raised: %s", exc)
            return SetupResult(success=False, error=str(exc))

        logger.info("Setup completed successfully")
        return SetupResult(success=True)

    def run_teardown(self) -> SetupResult:
        """Reverse setup. Must be called from an elevated process."""
        # Step 1: Revoke ACL entries BEFORE deleting account (while SID is resolvable)
        try:
            username = self.account_manager.get_username()
            workspace_path = self._get_default_workspace_path()
            if os.path.isdir(workspace_path):
                self.permission_manager.revoke_access(username, workspace_path)
                logger.info("Teardown step 1 complete: ACL entries revoked for %s", username)
        except Exception as exc:
            # Non-fatal: log warning but continue with account deletion
            # If revoke fails, we still want to delete the account
            logger.warning("Teardown step 1 (ACL revoke) failed: %s — continuing with deletion", exc)

        # Step 2: Delete sandbox account
        try:
            self.account_manager.delete_account()
            logger.info("Teardown step 2 complete: sandbox account deleted")
        except Exception as exc:
            logger.error("Teardown step 2 (account deletion) failed: %s", exc)
            return SetupResult(success=False, error=str(exc))

        # Note: We intentionally do NOT delete the workspace directory.
        # It may contain user data. check_setup_status will report it as present
        # but the sandbox user no longer exists.
        logger.info("Teardown completed successfully")
        return SetupResult(success=True)

    @staticmethod
    def is_elevated() -> bool:
        """Check whether the current process has administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

    @staticmethod
    def request_elevation(script_path: str) -> int:
        """Launch a Python script with UAC elevation via ShellExecuteW.

        Returns 0 on successful launch, -1 on failure.
        The elevated script communicates its result via a status file.
        """
        status_dir = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "AgentOS"
        )
        status_path = os.path.join(status_dir, _SETUP_STATUS_FILENAME)

        # Write initial status
        os.makedirs(status_dir, exist_ok=True)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({"status": "running"}, f)

        # ShellExecuteW with 'runas' verb triggers UAC
        ret = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            f'"{script_path}"',
            None,
            1,  # SW_SHOWNORMAL
        )

        # ShellExecuteW returns >32 on success, <=32 on error
        if ret <= 32:
            logger.error("ShellExecuteW failed with code %d", ret)
            try:
                os.remove(status_path)
            except OSError:
                pass
            return -1

        logger.info("Elevated process launched, polling for status")

        # Poll the status file for completion
        deadline = time.monotonic() + _ELEVATION_TIMEOUT
        while time.monotonic() < deadline:
            time.sleep(_ELEVATION_POLL_INTERVAL)
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "complete":
                    logger.info(
                        "Elevated setup finished: success=%s error=%s",
                        data.get("success"),
                        data.get("error"),
                    )
                    try:
                        os.remove(status_path)
                    except OSError:
                        pass
                    return 0 if data.get("success") else -1
            except (OSError, json.JSONDecodeError):
                continue

        logger.error("Timed out waiting for elevated setup to complete")
        try:
            os.remove(status_path)
        except OSError:
            pass
        return -1

    @staticmethod
    def _get_default_workspace_path() -> str:
        """Return the default workspace path: %LOCALAPPDATA%/AgentOS/workspace."""
        return os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "AgentOS", "workspace"
        )
