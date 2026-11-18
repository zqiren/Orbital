# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""macOS install step of the in-app upgrade (core: ``agent_os.updater``).

prepare
    Mount the downloaded, hash-checked DMG read-only (``hdiutil attach
    -nobrowse``), copy ``Orbital.app`` with ``ditto`` (keeps the ~2,500
    symlinks as symlinks; ``--noqtn`` because our own gate below replaces
    Gatekeeper's first-launch check) into a staging folder BESIDE the
    installed app, so every later move is a same-volume rename. Then the
    gate, all of which must pass before anything is touched:

    * bundle id and version: ``com.orbital.desktop``, exactly the offered
      version, newer than the running one;
    * no symlink inside resolves outside the bundle;
    * ``codesign --verify --deep --strict``;
    * the same Team ID as the running app (a Developer ID signature);
    * ``spctl -a -t exec``: Gatekeeper accepts it (notarized Developer ID).

handoff
    A small ``/bin/sh`` helper (:data:`HELPER_SCRIPT`), started in its own
    session so it outlives us, waits for this process to exit, moves the old
    bundle aside, moves the new one into place, ``open``s it, and waits for
    the new version to report in (``booted-<version>``, written at daemon
    start by :meth:`MacInstaller.on_startup`). Only then is the old bundle
    deleted. On any failure it puts the old bundle back and opens it, and
    writes ``result.json`` so the old app can say the update failed.

The DMG is used instead of a new .app zip: it is already published on both
GitHub and the China mirror with its sha256 (no release-process change), and
a zip of a 500 MB bundle is no smaller than the compressed DMG.

Not attempted (the pill keeps the manual download): an app running
translocated or from a read-only volume (a DMG), a folder or bundle this user
cannot write or does not own, or a running app without a Developer ID Team ID
(local and main-branch CI builds are ad-hoc signed).
"""

from __future__ import annotations

import asyncio
import logging
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from agent_os.update_check import is_newer
from agent_os.updater import UpdateError, same_version
from agent_os.version import get_version

logger = logging.getLogger(__name__)

BUNDLE_ID = "com.orbital.desktop"
HELPER_NAME = "apply-update.sh"
HELPER_LOG = "apply-update.log"
STAGING_PREFIX = ".Orbital-update-"
BOOTED_PREFIX = "booted-"
APP_NAME_IN_DMG = "Orbital.app"

Runner = Callable[..., subprocess.CompletedProcess]


def _run(cmd: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    from agent_os.utils.subprocess_flags import win_no_window_flags

    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                          creationflags=win_no_window_flags())


def running_bundle(executable: str | None = None) -> Path | None:
    """``…/X.app`` for ``…/X.app/Contents/MacOS/<exe>``, else None."""
    exe = Path(executable or sys.executable)
    try:
        exe = exe.resolve()
    except OSError:
        return None
    if len(exe.parents) < 3:
        return None
    macos, contents, bundle = exe.parent, exe.parents[1], exe.parents[2]
    if macos.name == "MacOS" and contents.name == "Contents" and bundle.suffix == ".app":
        return bundle
    return None


def bundle_info(bundle: Path) -> dict:
    try:
        with open(bundle / "Contents" / "Info.plist", "rb") as fh:
            data = plistlib.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def signing_identity(bundle: Path, run: Runner = _run) -> dict:
    """{"identifier", "team_id", "authority": [...]} from ``codesign -dv``.
    ``team_id`` is "" for ad-hoc or unsigned code."""
    out = {"identifier": "", "team_id": "", "authority": []}
    try:
        proc = run(["/usr/bin/codesign", "-dv", "--verbose=2", str(bundle)], timeout=60)
    except Exception:
        return out
    for line in (proc.stderr or "").splitlines() + (proc.stdout or "").splitlines():
        key, _, value = line.partition("=")
        if key == "Identifier":
            out["identifier"] = value.strip()
        elif key == "TeamIdentifier":
            team = value.strip()
            out["team_id"] = "" if team in ("", "not set") else team
        elif key == "Authority":
            out["authority"].append(value.strip())
    return out


def location_problem(bundle: Path) -> str | None:
    """Why the installed bundle cannot be swapped in place (None = it can)."""
    path = str(bundle)
    if "/AppTranslocation/" in path:
        return "translocated"
    try:
        if os.statvfs(path).f_flag & os.ST_RDONLY:
            return "read-only"
    except OSError:
        return "unreadable"
    parent = bundle.parent
    if not os.access(parent, os.W_OK | os.X_OK):
        return "folder-not-writable"
    try:
        if os.stat(bundle).st_uid != os.getuid():
            return "not-owner"
    except OSError:
        return "unreadable"
    if not os.access(bundle, os.W_OK):
        return "not-writable"
    return None


def symlink_escaping(root: Path) -> str | None:
    """The first symlink under ``root`` that resolves outside it, if any."""
    base = os.path.realpath(root)
    prefix = base.rstrip(os.sep) + os.sep
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                target = os.path.realpath(path)
                if target != base and not target.startswith(prefix):
                    return path
    return None


def verify_bundle(candidate: Path, *, expected_version: str, current_version: str,
                  reference: dict, run: Runner = _run) -> None:
    """The macOS gate. Raises ``UpdateError("verify", why)``."""
    def fail(why: str) -> None:
        raise UpdateError("verify", why)

    if not candidate.is_dir() or candidate.is_symlink():
        fail("no app bundle")
    info = bundle_info(candidate)
    if info.get("CFBundleIdentifier") != BUNDLE_ID or \
            (reference.get("identifier") and reference["identifier"] != BUNDLE_ID):
        fail(f"bundle id {info.get('CFBundleIdentifier')!r}")
    version = str(info.get("CFBundleShortVersionString") or "")
    if not same_version(version, expected_version):
        fail(f"bundle is {version!r}, offered {expected_version}")
    if not is_newer(version, current_version):
        fail(f"{version} is not newer than {current_version}")
    escaping = symlink_escaping(candidate)
    if escaping:
        fail(f"symlink leaves the bundle: {escaping}")
    proc = run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(candidate)],
               timeout=300)
    if proc.returncode != 0:
        fail("codesign --verify failed: " + (proc.stderr or "").strip()[-300:])
    ident = signing_identity(candidate, run)
    want_team = reference.get("team_id") or ""
    if not want_team or ident.get("team_id") != want_team:
        fail(f"team id {ident.get('team_id')!r}, want {want_team!r}")
    if ident.get("identifier") != BUNDLE_ID:
        fail(f"signed identifier {ident.get('identifier')!r}")
    proc = run(["/usr/sbin/spctl", "-a", "-t", "exec", "-vv", str(candidate)], timeout=300)
    if proc.returncode != 0:
        fail("Gatekeeper rejected it: " + ((proc.stderr or "") + (proc.stdout or "")).strip()[-300:])


# The helper runs with the system shell only: the frozen app has no Python to
# lend it, and it must keep running after this process is gone.
HELPER_SCRIPT = r'''#!/bin/sh
# Orbital in-app update helper (macOS). Written and started by
# agent_os/update_macos.py; runs detached, after the app it replaces exits.
#   $1 old pid   $2 installed .app   $3 new .app (staged beside it)
#   $4 backup path for the old .app   $5 cache dir   $6 new version
set -u
OLD_PID="$1"; APP="$2"; NEW="$3"; BACKUP="$4"; CACHE="$5"; VERSION="$6"
OPEN="${ORBITAL_UPDATE_OPEN:-/usr/bin/open}"
WAIT_EXIT="${ORBITAL_UPDATE_WAIT_EXIT:-240}"     # half-seconds
WAIT_BOOT="${ORBITAL_UPDATE_WAIT_BOOT:-360}"     # half-seconds
STAGE="$(dirname "$NEW")"
BOOTED="$CACHE/booted-$VERSION"
RESULT="$CACHE/result.json"

log() { echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"; }
result() {
  printf '{"result": "%s", "version": "%s", "at": %s}\n' "$1" "$VERSION" "$(date +%s)" \
    > "$RESULT.tmp" && mv -f "$RESULT.tmp" "$RESULT"
}
# PIDs whose executable lives inside $APP (the new app after the swap).
app_pids() {
  ps -axo pid=,comm= | awk -v p="$APP/Contents/" 'index($0, p) { print $1 }'
}
end_app() {
  pids="$(app_pids)"
  [ -z "$pids" ] && return 0
  kill $pids 2>/dev/null
  i=0
  while [ -n "$(app_pids)" ] && [ "$i" -lt 20 ]; do sleep 0.5; i=$((i + 1)); done
  pids="$(app_pids)"
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null
  sleep 0.5
}
give_up() {  # the old app is still in place: clean up, reopen it
  log "giving up: $1"
  result "$2"
  rm -rf "$STAGE"
  [ "$3" = reopen ] && "$OPEN" "$APP"
  exit 1
}

log "waiting for pid $OLD_PID to exit"
i=0
while kill -0 "$OLD_PID" 2>/dev/null; do
  i=$((i + 1))
  [ "$i" -gt "$WAIT_EXIT" ] && give_up "the old app did not exit" aborted no
  sleep 0.5
done

[ -d "$NEW" ] || give_up "staged app missing" failed reopen
rm -f "$BOOTED"
if ! mv "$APP" "$BACKUP"; then give_up "could not move the old app aside" failed reopen; fi
if ! mv "$NEW" "$APP"; then
  if mv "$BACKUP" "$APP"; then give_up "could not move the new app into place" failed reopen; fi
  # Never delete $STAGE here: it holds the only copy of the old app.
  log "RESTORE FAILED: the old app is at $BACKUP"
  result failed
  exit 2
fi
log "swapped; opening $APP"
if "$OPEN" "$APP"; then
  i=0
  while [ ! -e "$BOOTED" ] && [ "$i" -lt "$WAIT_BOOT" ]; do sleep 0.5; i=$((i + 1)); done
fi
if [ -e "$BOOTED" ]; then
  log "v$VERSION is up; removing the old app"
  rm -f "$BOOTED"
  rm -rf "$STAGE"
  exit 0
fi

log "v$VERSION did not come up; rolling back"
end_app
if mv "$APP" "$STAGE/failed.app" && mv "$BACKUP" "$APP"; then
  result rolled_back
  "$OPEN" "$APP"
  rm -rf "$STAGE"
  exit 1
fi
log "ROLLBACK FAILED: the old app is at $BACKUP"
result failed
exit 2
'''


class MacInstaller:
    def __init__(self, bundle: Path | None = None, *, run: Runner = _run,
                 popen: Callable[..., object] = subprocess.Popen,
                 current_version: Callable[[], str] = get_version,
                 open_cmd: str | None = None, retry_delay: float = 2.0) -> None:
        self._retry_delay = retry_delay
        self._bundle = bundle if bundle is not None else running_bundle()
        self._run = run
        self._popen = popen
        self._current = current_version
        self._open_cmd = open_cmd  # tests / rehearsal only
        self._reference: dict | None = None

    @property
    def bundle(self) -> Path | None:
        return self._bundle

    def reference_identity(self) -> dict:
        if self._reference is None:
            self._reference = signing_identity(self._bundle, self._run) if self._bundle \
                else {"identifier": "", "team_id": "", "authority": []}
        return self._reference

    def unsupported_reason(self) -> str | None:
        if self._bundle is None:
            return "not-a-bundle"
        problem = location_problem(self._bundle)
        if problem:
            return problem
        if not self.reference_identity().get("team_id"):
            return "unsigned"
        return None

    def staging_root(self, version: str) -> Path:
        assert self._bundle is not None
        return self._bundle.parent / f"{STAGING_PREFIX}{version}"

    async def prepare(self, downloaded: Path, version: str) -> Path:
        return await asyncio.to_thread(self._prepare_sync, downloaded, version)

    def _prepare_sync(self, dmg: Path, version: str) -> Path:
        if self._bundle is None:
            raise UpdateError("unsupported", "not-a-bundle")
        stage = self.staging_root(version)
        shutil.rmtree(stage, ignore_errors=True)
        try:
            stage.mkdir(mode=0o700)
        except OSError as exc:
            raise UpdateError("permission", f"cannot write beside the app: {exc}") from None
        staged = stage / self._bundle.name
        mnt = Path(tempfile.mkdtemp(prefix="mnt-", dir=str(dmg.parent)))
        try:
            # hdiutil fails transiently ("Resource busy") often enough that
            # build-macos.sh retries it too.
            for attempt in range(3):
                proc = self._run(["/usr/bin/hdiutil", "attach", "-nobrowse", "-readonly",
                                  "-noautoopen", "-noverify", "-mountpoint", str(mnt),
                                  str(dmg)], timeout=180)
                if proc.returncode == 0:
                    break
                if attempt < 2:
                    time.sleep(self._retry_delay)
            if proc.returncode != 0:
                raise UpdateError("verify", "hdiutil attach failed: " + (proc.stderr or "")[-200:])
            try:
                src = mnt / APP_NAME_IN_DMG
                if src.is_symlink() or not src.is_dir():
                    raise UpdateError("verify", "no Orbital.app in the disk image")
                proc = self._run(["/usr/bin/ditto", "--noqtn", str(src), str(staged)],
                                 timeout=900)
                if proc.returncode != 0:
                    raise UpdateError("install", "copy failed: " + (proc.stderr or "")[-200:])
            finally:
                detach = self._run(["/usr/bin/hdiutil", "detach", str(mnt)], timeout=120)
                if detach.returncode != 0:
                    self._run(["/usr/bin/hdiutil", "detach", "-force", str(mnt)], timeout=120)
            verify_bundle(staged, expected_version=version, current_version=self._current(),
                          reference=self.reference_identity(), run=self._run)
            return staged
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        finally:
            try:
                mnt.rmdir()
            except OSError:
                pass

    async def handoff(self, staged: Path, version: str, cache: Path) -> str:
        if self._bundle is None:
            raise UpdateError("unsupported", "not-a-bundle")
        if not Path(staged).is_dir():
            raise UpdateError("install", "staged app missing")
        helper = cache / HELPER_NAME
        helper.write_text(HELPER_SCRIPT, encoding="utf-8")
        os.chmod(helper, 0o700)
        _remove(cache / f"{BOOTED_PREFIX}{version}")
        backup = Path(staged).parent / "previous.app"
        env = dict(os.environ)
        if self._open_cmd:
            env["ORBITAL_UPDATE_OPEN"] = self._open_cmd
        else:
            env.pop("ORBITAL_UPDATE_OPEN", None)
        log = open(cache / HELPER_LOG, "ab")
        try:
            self._popen(
                ["/bin/sh", str(helper), str(os.getpid()), str(self._bundle), str(staged),
                 str(backup), str(cache), version],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd="/", env=env, close_fds=True, start_new_session=True)
        except OSError as exc:
            raise UpdateError("install", f"could not start the helper: {exc}") from None
        finally:
            log.close()
        return "go"

    def on_startup(self, cache: Path, pending: dict, current: str) -> None:
        """The new version is up: tell the waiting helper."""
        if same_version(pending.get("target"), current):
            marker = cache / f"{BOOTED_PREFIX}{current}"
            try:
                marker.write_text(str(os.getpid()), encoding="utf-8")
            except OSError:
                logger.warning("could not write the update boot marker", exc_info=True)

    def cleanup(self, cache: Path) -> None:
        """No install in flight: sweep staging folders, markers, mounts."""
        if self._bundle is not None:
            try:
                for child in self._bundle.parent.iterdir():
                    if child.name.startswith(STAGING_PREFIX) and child.is_dir() \
                            and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
            except OSError:
                pass
        if cache.is_dir():
            for child in cache.iterdir():
                if child.name.startswith(BOOTED_PREFIX) or child.name == HELPER_NAME:
                    _remove(child)
                elif child.name.startswith("mnt-") and child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("could not remove %s", path, exc_info=True)
