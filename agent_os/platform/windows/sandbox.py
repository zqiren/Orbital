# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import ctypes
import ctypes.wintypes as wintypes
import logging
import secrets
import subprocess

from agent_os.platform.types import SANDBOX_USERNAME, SANDBOX_PASSWORD_KEY, AccountStatus
from agent_os.platform.windows.credentials import CredentialStore

logger = logging.getLogger("agent_os.platform.windows.sandbox")

# LogonUserW constants
LOGON32_LOGON_NETWORK = 3
LOGON32_PROVIDER_DEFAULT = 0

# Win32 Net API constants
USER_PRIV_USER = 1
UF_SCRIPT = 0x0001
UF_DONT_EXPIRE_PASSWD = 0x10000
UF_PASSWD_CANT_CHANGE = 0x0040
NERR_Success = 0
ERROR_ACCESS_DENIED = 5

netapi32 = ctypes.windll.netapi32


class _USER_INFO_1(ctypes.Structure):
    """Win32 USER_INFO_1 for NetUserAdd."""
    _fields_ = [
        ("usri1_name", wintypes.LPWSTR),
        ("usri1_password", wintypes.LPWSTR),
        ("usri1_password_age", wintypes.DWORD),
        ("usri1_priv", wintypes.DWORD),
        ("usri1_home_dir", wintypes.LPWSTR),
        ("usri1_comment", wintypes.LPWSTR),
        ("usri1_flags", wintypes.DWORD),
        ("usri1_script_path", wintypes.LPWSTR),
    ]


class _USER_INFO_1003(ctypes.Structure):
    """Win32 USER_INFO_1003 for password-only update."""
    _fields_ = [("usri1003_password", wintypes.LPWSTR)]


class SandboxAccountManager:
    """Manages the AgentOS-Worker Windows user account for sandbox isolation."""

    SANDBOX_USERNAME = SANDBOX_USERNAME

    def __init__(self, credential_store: CredentialStore) -> None:
        self._credential_store = credential_store

    def get_username(self) -> str:
        """Return the sandbox username constant."""
        return SANDBOX_USERNAME

    def ensure_account_exists(self) -> AccountStatus:
        """Create the sandbox account if it does not exist, then validate it.

        If the account exists but the stored password is invalid or missing,
        resets the password (requires admin).
        """
        is_admin = self._is_admin()

        if self._account_exists():
            logger.info("Sandbox account '%s' already exists", SANDBOX_USERNAME)
            status = self.validate_account()

            # If password is invalid and we're admin, reset it
            if not status.password_valid and is_admin:
                logger.info("Resetting password for existing sandbox account")
                password = secrets.token_urlsafe(32)
                if self._set_password(password):
                    self._credential_store.store(SANDBOX_PASSWORD_KEY, password)
                    logger.info("Password reset and stored successfully")
                    return self.validate_account()
                else:
                    return AccountStatus(
                        exists=True,
                        username=SANDBOX_USERNAME,
                        password_valid=False,
                        is_admin=is_admin,
                        error="Failed to reset password",
                    )

            return status

        # Account does not exist — need admin to create it
        if not is_admin:
            logger.warning("Cannot create sandbox account without elevation")
            return AccountStatus(
                exists=False,
                username=SANDBOX_USERNAME,
                password_valid=False,
                is_admin=False,
                error="Requires administrator privileges",
            )

        # Generate a secure password and create the user via Win32 API
        password = secrets.token_urlsafe(32)

        if not self._create_user(password):
            return AccountStatus(
                exists=False,
                username=SANDBOX_USERNAME,
                password_valid=False,
                is_admin=True,
                error="Failed to create account via NetUserAdd",
            )

        logger.info("Created sandbox account '%s'", SANDBOX_USERNAME)

        # Add to Users group (may already be a member — ignore errors)
        self._run_cmd(["net", "localgroup", "Users", SANDBOX_USERNAME, "/add"])

        # Store the password in the credential store
        self._credential_store.store(SANDBOX_PASSWORD_KEY, password)
        logger.info("Stored sandbox account password in credential store")

        # Validate the newly created account
        return self.validate_account()

    def validate_account(self) -> AccountStatus:
        """Check whether the sandbox account exists and the stored password is valid."""
        is_admin = self._is_admin()

        if not self._account_exists():
            logger.info("Sandbox account '%s' does not exist", SANDBOX_USERNAME)
            return AccountStatus(
                exists=False,
                username=SANDBOX_USERNAME,
                password_valid=False,
                is_admin=is_admin,
            )

        # Account exists — try to retrieve stored password
        password = self._credential_store.retrieve(SANDBOX_PASSWORD_KEY)
        if password is None:
            logger.warning("Sandbox account exists but no stored password found")
            return AccountStatus(
                exists=True,
                username=SANDBOX_USERNAME,
                password_valid=False,
                is_admin=is_admin,
                error="Password not found in credential store",
            )

        # Test authentication via LogonUserW
        password_valid = self._test_logon(SANDBOX_USERNAME, password)
        if password_valid:
            logger.info("Sandbox account password validated successfully")
        else:
            logger.warning("Sandbox account password validation failed")

        return AccountStatus(
            exists=True,
            username=SANDBOX_USERNAME,
            password_valid=password_valid,
            is_admin=is_admin,
        )

    def delete_account(self) -> None:
        """Delete the sandbox account and its stored credential. Requires elevation."""
        if not self._is_admin():
            raise RuntimeError("Requires administrator privileges to delete account")

        rc = netapi32.NetUserDel(None, SANDBOX_USERNAME)
        if rc != NERR_Success:
            logger.error("NetUserDel failed with code %d", rc)
            raise RuntimeError(f"Failed to delete account (NetUserDel error {rc})")

        logger.info("Deleted sandbox account '%s'", SANDBOX_USERNAME)

        self._credential_store.delete(SANDBOX_PASSWORD_KEY)
        logger.info("Removed sandbox password from credential store")

    # ------------------------------------------------------------------
    # Win32 API helpers
    # ------------------------------------------------------------------

    def _account_exists(self) -> bool:
        """Check if the sandbox user account exists via net user query."""
        returncode, _, _ = self._run_cmd(["net", "user", SANDBOX_USERNAME])
        return returncode == 0

    def _create_user(self, password: str) -> bool:
        """Create the sandbox user via NetUserAdd."""
        user_info = _USER_INFO_1()
        user_info.usri1_name = SANDBOX_USERNAME
        user_info.usri1_password = password
        user_info.usri1_password_age = 0
        user_info.usri1_priv = USER_PRIV_USER
        user_info.usri1_home_dir = None
        user_info.usri1_comment = None
        user_info.usri1_flags = UF_SCRIPT | UF_DONT_EXPIRE_PASSWD | UF_PASSWD_CANT_CHANGE
        user_info.usri1_script_path = None

        parm_err = wintypes.DWORD(0)
        rc = netapi32.NetUserAdd(None, 1, ctypes.byref(user_info), ctypes.byref(parm_err))
        if rc != NERR_Success:
            logger.error("NetUserAdd failed with code %d (parm_err=%d)", rc, parm_err.value)
            return False
        return True

    def _set_password(self, password: str) -> bool:
        """Reset the sandbox user password via NetUserSetInfo."""
        info = _USER_INFO_1003()
        info.usri1003_password = password
        parm_err = wintypes.DWORD(0)
        rc = netapi32.NetUserSetInfo(
            None, SANDBOX_USERNAME, 1003,
            ctypes.byref(info), ctypes.byref(parm_err),
        )
        if rc != NERR_Success:
            logger.error("NetUserSetInfo failed with code %d", rc)
            return False
        return True

    def _run_cmd(self, cmd: list[str]) -> tuple[int, str, str]:
        """Run a subprocess command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=15,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("Command timed out: %s", cmd)
            return -1, "", "Command timed out"

    @staticmethod
    def _is_admin() -> bool:
        """Check whether the current process is running with administrator privileges."""
        return ctypes.windll.shell32.IsUserAnAdmin() != 0

    @staticmethod
    def _test_logon(username: str, password: str) -> bool:
        """Test credentials via LogonUserW. Returns True if authentication succeeds."""
        token = wintypes.HANDLE()
        success = ctypes.windll.advapi32.LogonUserW(
            username,
            ".",
            password,
            LOGON32_LOGON_NETWORK,
            LOGON32_PROVIDER_DEFAULT,
            ctypes.byref(token),
        )
        if success:
            ctypes.windll.kernel32.CloseHandle(token)
            return True
        return False
