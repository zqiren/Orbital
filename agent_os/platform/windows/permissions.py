# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""C2: PermissionManager — Windows ACL management via icacls."""

import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
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
from agent_os.platform.windows import acl
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger("agent_os.platform.windows.permissions")

#: The write rights by name, i.e. generic-write minus ``SYNCHRONIZE`` (spec
#: 114 D1): write data / add file, append data / add subfolder, write
#: extended attributes, write attributes.
_WRITE_RIGHTS: tuple[str, ...] = ("WD", "AD", "WEA", "WA")
#: Delete, change permissions, take ownership. Windows lets a holder of
#: delete replace a file by renaming another over it (git writes
#: ``config.lock`` then renames it), and a holder of change-permissions lift
#: any deny — so a write-only deny does not protect a file (bench
#: 2026-10-19, issue 1).
_REPLACE_RIGHTS: tuple[str, ...] = ("DE", "WDAC", "WO")

W3Kind = Literal["repo_root", "git_dir", "hooks", "config"]

#: What each W3 target denies the worker. Reads (and ``SYNCHRONIZE``) are
#: never denied: the worker keeps using git. ``git_dir`` and ``repo_root``
#: are this-folder-only, so files git rewrites keep their own delete right.
W3_TARGET_RIGHTS: dict[str, tuple[str, ...]] = {
    "config": _WRITE_RIGHTS + _REPLACE_RIGHTS,
    "hooks": _WRITE_RIGHTS + _REPLACE_RIGHTS + ("DC",),
    # Not renamed away (which would let a fresh ``.git`` take its place),
    # no child replaced through the folder, no permission change.
    "git_dir": ("DE", "DC", "WDAC", "WO"),
    # ``.git`` cannot be replaced through its parent; no permission change.
    "repo_root": ("DC", "WDAC", "WO"),
}

#: ``OWNER RIGHTS``: an ACE for it replaces an owner's implicit rights.
OWNER_RIGHTS_SID = "S-1-3-4"
_FULL_CONTROL = 0x1F01FF
_MODIFY = 0x1301BF

#: What ``icacls ... R`` / ``F`` write, as file-specific rights.
_READ = 0x120089
_MODE_MASKS: dict[str, int] = {"read_only": _READ, "read_write": _FULL_CONTROL}

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_GENERIC_EXECUTE = 0x20000000
_GENERIC_ALL = 0x10000000


def _file_rights(mask: int) -> int:
    """*mask* with the generic bits mapped to file rights (FILE_GENERIC_*)."""
    out = mask & 0x00FFFFFF
    if mask & _GENERIC_ALL:
        out |= _FULL_CONTROL
    if mask & _GENERIC_READ:
        out |= _READ
    if mask & _GENERIC_WRITE:
        out |= 0x120116
    if mask & _GENERIC_EXECUTE:
        out |= 0x1200A0
    return out


#: Classification of what a target already holds for the worker.
DenyState = Literal["none", "current", "stale"]

_ACE_GROUP_RE = re.compile(r"\(([^()]*)\)")


class PermissionManager:
    """Manages file-system permissions for the sandbox user via icacls."""

    def __init__(self, icacls_timeout: float | None = None) -> None:
        # Spec 109 D7: a per-call bound on icacls, in seconds. ``None`` (the
        # daemon and the Folder-access UI) means no bound — a user may
        # legitimately grant a huge folder behind a visible spinner. The
        # installer paths (``SetupOrchestrator.run_setup/run_teardown``) set
        # it for their duration so an unknown slow DACL (network profile,
        # antivirus hooking icacls) can never wedge Inno's progress bar.
        self.icacls_timeout: float | None = icacls_timeout

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
            # Before the worker grant, so the inheritable user entry exists
            # by the time the worker creates anything (bench 2026-10-19).
            self.harden_writable_root(resolved)

        # Every session start re-grants its workspace, and every ``/grant``
        # re-propagates through the whole subtree even when the identical ACE
        # is already there (spec 109 V1). A root that already carries the
        # entry costs one DACL read; files created later inherit it.
        if self._has_root_grant(username, resolved, _MODE_MASKS[mode]):
            return PermissionResult(success=True, path=resolved)

        # Inheritable and NOT ``/T`` — the same rule as the toolchain roots
        # (spec 109 D1, verified V1): the root ACE reaches every inheriting
        # descendant, while ``/T`` also wrote an explicit ACE on each file
        # one at a time. On an imported D:\ project tree that walk held the
        # daemon for 1016 s (feedback row 4, 2026-11-07). A subtree with
        # inheritance turned off does not get the entry; that is one click in
        # Settings > Folder access on the subtree itself.
        # /C: one entry the user cannot re-permission (a worker-owned file on
        # a drive where the user holds no change-permissions) must not stop
        # the propagation over the rest of the tree.
        result = self._run_icacls([resolved, "/grant", perm, "/C", "/Q"])
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            if _processed_some(result.stdout):
                logger.warning("icacls grant for %s on %s skipped some entries: %s",
                               username, resolved, err)
            else:
                logger.error("icacls grant failed for %s on %s: %s", username, resolved, err)
                return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Granted %s access (%s) to %s", username, mode, resolved)
        return PermissionResult(success=True, path=resolved)

    def revoke_access(self, username: str, path: str) -> PermissionResult:
        """Revoke all access for *username* on *path*.

        Root-only, deliberately no ``/T`` (spec 109 D1): removing the
        inheritable ACE from the root removes the inherited copies through
        the same NTFS propagation that put them there. With ``/T`` this was
        a per-file walk over every toolchain root — the uninstall hang in
        issue #55. Explicit per-file entries written by pre-109 installs
        (which used ``/T`` on the grant) are not touched; once the account is
        deleted they are unresolvable-SID clutter that Windows ignores.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")

        result = self._run_icacls([resolved, "/remove", username, "/Q"])
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error("icacls revoke failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Revoked access for %s on %s", username, resolved)
        return PermissionResult(success=True, path=resolved)

    # ------------------------------------------------------------------
    # Spec 077 W1/W2/W3 — toolchain grants, worker home, deny ACEs
    # ------------------------------------------------------------------

    def deny_access(self, username: str, path: str,
                    kind: W3Kind | None = None) -> PermissionResult:
        """Add the W3 deny ACE for *username* on *path* (spec 077 W3, 114 D1).

        *kind* picks the rights (:data:`W3_TARGET_RIGHTS`); when omitted it is
        inferred from the name (``config`` / ``hooks``). Deny ACEs precede
        allows in Windows access evaluation, so the read-write grant on the
        enclosing workspace cannot override them.

        Rights are always named one by one, never the ``W`` shorthand: ``W``
        carries ``SYNCHRONIZE``, which every synchronous open asks for, so a
        ``W`` deny failed *reads* of ``.git\\config`` (issue #58).

        Only ``hooks`` is inheritable (``(OI)(CI)`` + ``/T``), so files created
        in it inherit the deny. The ``.git`` folder and the repository root are
        this-folder-only: an inheritable delete-deny there would reach every
        file git rewrites (index, refs, lock files) and break commits.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")
        kind = kind or _infer_kind(resolved)

        rights = "(" + ",".join(W3_TARGET_RIGHTS[kind]) + ")"
        flags = f"(OI)(CI){rights}" if kind == "hooks" else rights
        args = [resolved, "/deny", f"{username}:{flags}", "/Q"]
        if kind == "hooks":
            args.insert(-1, "/T")

        result = self._run_icacls(args)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.error("icacls deny failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)

        logger.info("Denied %s rights for %s on %s", kind, username, resolved)
        return PermissionResult(success=True, path=resolved)

    def remove_deny(self, username: str, path: str,
                    kind: W3Kind | None = None) -> PermissionResult:
        """Drop the deny ACE added by :meth:`deny_access`.

        Teardown must call this: a deny ACE left on the user's own repository
        after uninstall outlives the account it names (spec 077 §8). Recursive
        only for ``hooks``: a recursive ``/remove:d`` on the repository root
        would strip every deny in the workspace, nested repositories included.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return PermissionResult(success=False, path=path, error="Path does not exist")
        kind = kind or _infer_kind(resolved)

        args = [resolved, "/remove:d", username, "/Q"]
        if kind == "hooks":
            args.insert(-1, "/T")

        result = self._run_icacls(args)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.warning("icacls remove:d failed for %s on %s: %s", username, resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)
        return PermissionResult(success=True, path=resolved)

    def deny_state(self, username: str, path: str,
                   kind: W3Kind | None = None) -> DenyState:
        """What *path* already carries for *username*.

        Read from the ACE masks (:func:`acl.read_dacl`), never from ``icacls``
        text, which prints different masks identically (``0x116`` and
        ``0x100116`` are both ``(W)`` on Windows 10). ``"current"`` — exactly
        the deny for this *kind*; ``"stale"`` — any other explicit deny for
        the account (v0.16.0's ``W``, spec 114's write-only mask); ``"none"``
        — no deny, or the DACL could not be read.
        """
        resolved = self._resolve_path(path)
        if resolved is None:
            return "none"
        kind = kind or _infer_kind(resolved)
        sid = self._sid_for(username)
        aces = acl.read_dacl(resolved)
        if sid is None or aces is None:
            return "none"
        return classify_deny_aces(aces, sid, kind)

    def has_deny(self, username: str, path: str, kind: W3Kind | None = None) -> bool:
        """True when *username* carries the **current** W3 deny on *path*."""
        return self.deny_state(username, path, kind=kind) == "current"

    def protect_control_files(self, username: str, root: str) -> list[PermissionResult]:
        """Apply the W3 denies to every target of the repository at *root*.

        Idempotent and cheap: a DACL read per target, a write only when the
        deny is missing or stale. Windows ACLs need a real path — there is no
        Seatbelt-style pattern for a not-yet-existing file — which is why this
        is re-run after commands that could have created a repository.
        """
        results: list[PermissionResult] = []
        removed = _remove_planted_commondir(root)
        if removed is not None:
            results.append(removed)
        for target, kind in w3_targets(root):
            state = self.deny_state(username, target, kind=kind)
            if state == "current":
                continue
            if state == "stale":
                # An older deny shape heals here, on the worker's next command.
                # ``/remove:d`` clears every deny for the account on this
                # object, so the old ACE and the new one never coexist.
                logger.info("Rewriting stale W3 deny ACE for %s on %s (%s)",
                            username, target, kind)
                self.remove_deny(username, target, kind=kind)
            results.append(self.deny_access(username, target, kind=kind))
        return results

    def unprotect_control_files(self, username: str, root: str) -> list[PermissionResult]:
        """Reverse :meth:`protect_control_files` (teardown)."""
        return [self.remove_deny(username, target, kind=kind)
                for target, kind in w3_targets(root)]

    def harden_writable_root(self, root: str) -> PermissionResult | None:
        """Make sure the daemon can always re-permission what the worker creates.

        Adds, once, two inheritable entries on a writable root:

        * the current user, full control. Files the worker creates are owned by
          the worker, and without an inherited grant the user can lack
          change-permissions on them (seen on a ``D:\\`` workspace), so W3
          could not be applied to a repository the worker initialised;
        * ``OWNER RIGHTS`` (``S-1-3-4``), modify. An owner otherwise holds
          change-permissions implicitly, whatever the DACL denies, so the
          worker could lift a deny on a repository it created.

        Returns ``None`` when both are already present (the common case).
        """
        resolved = self._resolve_path(root)
        if resolved is None:
            return PermissionResult(success=False, path=root, error="Path does not exist")
        if self._root_hardening_present(resolved):
            return None
        user = _current_user_principal()
        args = [resolved, "/grant", f"{user}:(OI)(CI)F",
                f"*{OWNER_RIGHTS_SID}:(OI)(CI)M", "/Q"]
        result = self._run_icacls(args)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            logger.warning("root hardening failed on %s: %s", resolved, err)
            return PermissionResult(success=False, path=resolved, error=err)
        logger.info("Hardened writable root %s (user and OWNER RIGHTS entries)", resolved)
        return PermissionResult(success=True, path=resolved)

    def _root_hardening_present(self, resolved: str) -> bool:
        aces = acl.read_dacl(resolved)
        principal = _current_user_principal()
        user_sid = principal[1:] if principal.startswith("*") else self._sid_for(principal)
        if aces is None or user_sid is None:
            return False
        inherit = acl.OBJECT_INHERIT_ACE | acl.CONTAINER_INHERIT_ACE

        def has(sid: str, mask: int) -> bool:
            return any(
                a.type == acl.ACCESS_ALLOWED_ACE_TYPE and a.sid == sid
                and not a.flags & acl.INHERITED_ACE
                and a.flags & inherit == inherit and a.mask & mask == mask
                for a in aces
            )

        return has(user_sid, _FULL_CONTROL) and has(OWNER_RIGHTS_SID, _MODIFY)

    def _has_root_grant(self, username: str, resolved: str, need: int) -> bool:
        """True when *resolved* itself carries an explicit, inheritable allow
        for *username* that covers *need*. ``False`` whenever unsure."""
        sid = self._sid_for(username)
        aces = acl.read_dacl(resolved)
        if sid is None or aces is None:
            return False
        inherit = acl.OBJECT_INHERIT_ACE | acl.CONTAINER_INHERIT_ACE
        return any(
            a.type == acl.ACCESS_ALLOWED_ACE_TYPE and a.sid == sid
            and not a.flags & (acl.INHERITED_ACE | acl.INHERIT_ONLY_ACE
                                | acl.NO_PROPAGATE_INHERIT_ACE)
            and a.flags & inherit == inherit
            and _file_rights(a.mask) & need == need
            for a in aces
        )

    def _sid_for(self, username: str) -> str | None:
        cache = self.__dict__.setdefault("_sid_cache", {})
        if username not in cache:
            cache[username] = acl.account_sid(username)
        return cache[username]

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

        The ACE is **inheritable** (``(OI)(CI)``) and ``/T`` is deliberately
        absent (spec 109 D1). Setting an inheritable ACE on the root makes
        Windows propagate it to the existing subtree as *inherited* entries;
        ``/T`` additionally wrote an *explicit* ACE on every descendant, one
        file at a time, which on a dev machine (``%LOCALAPPDATA%\\Programs``,
        ``.cargo``, ``.rustup``, ``npm``: 10^5-10^6 files) pinned a CPU for
        minutes behind the installer's frozen "Configuring agent sandbox"
        bar (issue #55). Descendants whose DACL has inheritance disabled
        (protected) do not receive the ACE — V1 found one, ``Microsoft VS
        Code``, whose protected DACL grants ``BUILTIN\\Users:(RX)`` anyway,
        so the worker reads it regardless. Acceptable for a best-effort
        grant; anything missed is one click in Settings > Folder access.

        The propagation is still a walk of the subtree, just a much cheaper
        one, and it happens on EVERY ``/grant`` — re-granting an ACE that is
        already there costs the same as the first grant. So a root that
        already carries the explicit inheritable RX entry is skipped after
        one single-object ``icacls`` query; files created later inherit at
        creation time without any walk.

        Measured (spec 109 V1, 2026-10-10, Windows 10 22H2 laptop,
        ``%LOCALAPPDATA%\\Programs`` with 104,018 entries, NTFS SSD): with
        ``/T``: 182.3 s grant / 182.0 s remove; without: 23.5 s grant
        (19.2-19.3 s warm, identical for a repeat grant) / 19.6 s remove. A
        deep file (``Python\\Python313\\Lib\\encodings\\utf_8.py``) showed
        ``AgentOS-Worker:(I)(RX)`` after the ``/T``-less grant and lost it
        after the root-only ``/remove``.
        """
        results: list[PermissionResult] = []
        for root in windows_toolchain_roots():
            if not os.path.isdir(root):
                continue
            resolved = self._resolve_path(root)
            if resolved is None:
                continue
            query = self._run_icacls([resolved])
            if query.returncode == 0 and _has_inheritable_rx_grant(query.stdout, username):
                results.append(PermissionResult(success=True, path=resolved))
                continue
            result = self._run_icacls(
                [resolved, "/grant", f"{username}:(OI)(CI)RX", "/Q"]
            )
            if result.returncode != 0:
                err = result.stderr.strip() or result.stdout.strip()
                logger.warning("toolchain grant failed on %s: %s", resolved, err)
                results.append(PermissionResult(success=False, path=resolved, error=err))
            else:
                results.append(PermissionResult(success=True, path=resolved))
        return results

    def revoke_toolchain_roots(self, username: str) -> list[PermissionResult]:
        """Reverse :meth:`grant_toolchain_roots` (teardown).

        A root-only ``/remove`` still walks the whole subtree to re-propagate
        inheritance — and it does so even when the account has no entry on
        the root (spec 109 V1 follow-up, measured on a dev machine: npm 34.8
        s, ``%LOCALAPPDATA%\\Programs`` 19.3 s, pnpm 4.9 s; 59 s in total
        with or without the ACE present). This runs inside one hidden
        uninstaller step whose progress bar does not move, so: a root with no
        explicit entry for *username* is skipped after a single-object
        query, and the remaining roots are revoked concurrently. The roots
        never nest, so the concurrent propagations touch disjoint trees.
        Measured on the same machine: 49.7 s concurrent vs 58.8 s sequential
        with the ACE on all five roots (disk-bound, so not the ideal max of
        the roots), 0.04 s vs 59 s with no ACE to remove.
        """
        roots = [r for r in windows_toolchain_roots() if os.path.isdir(r)]
        if not roots:
            return []
        with ThreadPoolExecutor(max_workers=len(roots)) as pool:
            return list(pool.map(lambda r: self._revoke_toolchain_root(username, r), roots))

    def _revoke_toolchain_root(self, username: str, root: str) -> PermissionResult:
        resolved = self._resolve_path(root)
        if resolved is None:
            return PermissionResult(success=False, path=root, error="Path does not exist")
        query = self._run_icacls([resolved])
        if query.returncode == 0 and not _has_explicit_entry(query.stdout, username):
            return PermissionResult(success=True, path=resolved)
        return self.revoke_access(username, resolved)

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

    def _run_icacls(
        self, args: list[str], timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run icacls with the given arguments.

        *timeout* (seconds) overrides :attr:`icacls_timeout` for this call;
        ``None`` falls back to the attribute. On ``TimeoutExpired`` the child
        is already killed by ``subprocess.run``; a synthetic
        ``CompletedProcess`` with ``returncode=-1`` and a ``"timed out"``
        stderr comes back, so every caller's existing ``returncode != 0``
        branch reports a failed PermissionResult instead of raising (spec
        109 D7). A timed-out call can leave propagation partial; every call
        on the installer path is idempotent and re-runs at daemon start.
        """
        cmd = ["icacls"] + args
        bound = timeout if timeout is not None else self.icacls_timeout
        logger.debug("Running: %s", " ".join(cmd))
        try:
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
                timeout=bound,
            )
        except subprocess.TimeoutExpired:
            target = args[0] if args else "?"
            logger.warning(
                "icacls timed out after %ss on %s; child killed, continuing",
                bound, target,
            )
            return subprocess.CompletedProcess(
                args=cmd, returncode=-1, stdout="",
                stderr=f"timed out after {bound}s",
            )


def w3_targets(root: str) -> list[tuple[str, W3Kind]]:
    """The W3 targets of the repository at *root*, or ``[]`` without one.

    Only a real ``.git`` folder counts: a ``.git`` *file* is a worktree or
    submodule pointer whose control files live elsewhere.
    """
    git_dir = os.path.join(root, ".git")
    if not os.path.isdir(git_dir):
        return []
    targets: list[tuple[str, W3Kind]] = [(root, "repo_root"), (git_dir, "git_dir")]
    for entry in windows_protected_control_files(root):
        name = os.path.basename(entry.replace("\\", "/"))
        if name == "commondir":
            continue  # never kept and denied: removed (_remove_planted_commondir)
        path = os.path.join(git_dir, name)
        if os.path.exists(path):
            targets.append((path, "hooks" if name == "hooks" else "config"))
    return targets


def _remove_planted_commondir(root: str) -> PermissionResult | None:
    """Delete a top-level ``.git/commondir`` under *root*, if one exists.

    It redirects where git reads the repository's config from, so a worker
    that plants one (and a config of its own) makes the user's next
    ``git status`` run a program it chose, with ``.git\\config`` untouched.
    A deny cannot be placed on a file that does not exist yet, so this pass
    — run at start and after every command — removes it. Git only writes
    ``commondir`` inside ``.git\\worktrees\\<name>\\``, never at the top.
    """
    path = os.path.join(root, ".git", "commondir")
    if not os.path.isdir(os.path.join(root, ".git")) or not os.path.lexists(path):
        return None
    try:
        os.remove(path)
    except OSError as exc:
        logger.warning("could not remove planted %s: %s", path, exc)
        return PermissionResult(success=False, path=path, error=str(exc))
    logger.warning("Removed a planted %s (it redirects git's config)", path)
    return PermissionResult(success=True, path=path)


def classify_deny_aces(aces: list[acl.Ace], sid: str, kind: str) -> DenyState:
    """``current`` / ``stale`` / ``none`` for the worker's denies on a target.

    Only explicit (non-inherited) deny ACEs for *sid* count. ``current``
    needs the exact mask and inheritance for *kind*; any other explicit deny
    is ``stale`` and outranks a current one — the rewrite must run.
    """
    expected_mask = acl.rights_mask(W3_TARGET_RIGHTS[kind])
    inherit = acl.OBJECT_INHERIT_ACE | acl.CONTAINER_INHERIT_ACE
    expected_flags = inherit if kind == "hooks" else 0
    state: DenyState = "none"
    for ace in aces:
        if ace.type != acl.ACCESS_DENIED_ACE_TYPE or ace.sid != sid:
            continue
        if ace.flags & acl.INHERITED_ACE:
            continue
        flags = ace.flags & (inherit | acl.INHERIT_ONLY_ACE | acl.NO_PROPAGATE_INHERIT_ACE)
        if ace.mask == expected_mask and flags == expected_flags:
            if state == "none":
                state = "current"
        else:
            return "stale"
    return state


_PROCESSED_RE = re.compile(
    r"Successfully processed (\d+) files|已成功处理\s*(\d+)\s*个文件", re.IGNORECASE)


def _processed_some(stdout: str) -> bool:
    """True when an icacls summary line reports at least one processed file."""
    m = _PROCESSED_RE.search(stdout or "")
    return bool(m and int(m.group(1) or m.group(2)) > 0)


def _infer_kind(path: str) -> W3Kind:
    name = os.path.basename(os.path.normpath(path)).lower()
    if name == "hooks":
        return "hooks"
    if name == ".git":
        return "git_dir"
    if name == "config":
        return "config"
    return "repo_root"


def _current_user_principal() -> str:
    """The account the daemon runs as, in a form icacls accepts."""
    domain = os.environ.get("USERDOMAIN", "")
    user = os.environ.get("USERNAME") or ""
    if not user:
        import getpass
        user = getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def _has_explicit_entry(output: str, username: str) -> bool:
    """True when *username* has any non-inherited entry (allow or deny) in
    ``icacls`` output — something a root-only ``/remove`` would take off."""
    username_lower = username.lower()
    for line in output.splitlines():
        if username_lower + ":" not in line.lower():
            continue
        flags = {f.upper() for f in re.findall(r"\(([^)]+)\)", line)}
        if "I" not in flags:
            return True
    return False


def _has_inheritable_rx_grant(output: str, username: str) -> bool:
    """True when *username* has the explicit ``(OI)(CI)`` read+execute (or
    wider) allow entry that :meth:`PermissionManager.grant_toolchain_roots`
    writes, e.g. ``DESKTOP-ABC\\AgentOS-Worker:(OI)(CI)(RX)``.

    An inherited ``(I)`` copy does not count: it would vanish if the parent
    lost its entry, and the root is where the grant belongs.
    """
    username_lower = username.lower()
    for line in output.splitlines():
        line_lower = line.lower()
        if username_lower + ":" not in line_lower:
            continue
        flags = {f.upper() for f in re.findall(r"\(([^)]+)\)", line)}
        if "DENY" in flags or "I" in flags:
            continue
        if {"OI", "CI"} <= flags and flags & {"RX", "M", "F"}:
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
