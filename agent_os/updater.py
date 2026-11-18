# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-app upgrade: one click downloads, verifies, installs and relaunches.

The notify tier (``update_check``) learns that a newer version exists. This
module turns the pill click into the upgrade, with one core for both
platforms and a small platform step at the end:

    idle -> downloading -> verifying -> ready -> installing -> (app exits)
                 \\            \\          \\           \\
                  `------------`----------`-----------`--> failed

* **Feed.** The installer and its sha256 come from what release already
  publishes: the China mirror's ``latest.json`` (``scripts/sync-release-to-
  oss.sh``) and the GitHub release asset digest (``releases/tags/v<ver>``).
  The UI language picks the first source (the same rule the manual download
  uses); the other one is the fallback on any error. A source is only used
  when it serves exactly the version the pill offered, and only upgrades are
  ever offered (``is_newer``).
* **Download.** HTTP Range into ``<cache>/Orbital-<ver>.<ext>.part``, resumed
  across clicks and restarts (``If-Range`` guards a file replaced in place),
  size-checked, then sha256-checked before it is renamed into place. A wrong
  hash deletes the file. Stale parts and older downloads are swept.
* **Install.** ``update_macos`` / ``update_windows``. Both run only after the
  normal quit path agreed (``confirm_quit``: asks only while work runs) and
  both leave the old app working on any failure.
* **Inert in dev.** Installing needs the desktop shell's quit hooks
  (``register_shell``, set by ``agent_os/desktop/main.py``) AND a frozen
  build. A dev daemon has neither, so it reports ``can_install: false`` and
  the pill keeps the manual download. Tests inject both.

Integrity, honestly: the sha256 travels beside the file (same bucket, same
release), so it catches corruption and truncated or swapped-in-place files,
not a compromised bucket or release. On macOS the real gate is the code
signature: Developer ID, our Team ID, our bundle id, and Gatekeeper's
notarization assessment (``update_macos.verify_bundle``). Windows installers
are unsigned, so there the hash is the only gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from agent_os.update_check import CHINA_DOWNLOAD_BASE, _parse_version, is_newer
from agent_os.version import get_version

logger = logging.getLogger(__name__)

IDLE = "idle"
DOWNLOADING = "downloading"
VERIFYING = "verifying"
READY = "ready"
INSTALLING = "installing"
FAILED = "failed"
ACTIVE_STATES = frozenset({DOWNLOADING, VERIFYING, INSTALLING})

OSS_LATEST_URL = f"{CHINA_DOWNLOAD_BASE}/latest.json"
GITHUB_TAG_API = "https://api.github.com/repos/zqiren/Orbital/releases/tags/v{version}"
GITHUB_DOWNLOAD_PREFIX = "https://github.com/zqiren/Orbital/releases/download/"

#: A pending/result record older than this is ignored at startup (the user
#: relaunched days later; it no longer describes this launch).
_RECORD_MAX_AGE_S = 6 * 60 * 60
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Artifact:
    feed_key: str          # latest.json files.<key>
    stable_name: str       # release-aliases.sh / OSS name
    versioned: re.Pattern  # the versioned release asset
    suffix: str
    # Free space needed beyond the download itself, as a multiple of its
    # size: macOS unpacks the .app (~1.3x the DMG) beside the installed one.
    extra_space: float


ARTIFACTS = {
    "darwin": Artifact("mac", "Orbital-macOS.dmg",
                       re.compile(r"^Orbital-(\d+\.\d+\.\d+)-macOS\.dmg$"), ".dmg", 1.5),
    "win32": Artifact("win", "Orbital-Setup.exe",
                      re.compile(r"^Orbital-Setup-(\d+\.\d+\.\d+)\.exe$"), ".exe", 0.2),
}


class UpdateError(Exception):
    """A failed step. ``code`` is what the UI and logs see:

    feed, download, hash, space, verify, permission, install, launch,
    not_newer, unsupported.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class FeedEntry:
    version: str
    url: str
    sha256: str
    size: int
    source: str  # "china" | "global"


def normalize_version(value: Any) -> str:
    return str(value or "").strip().lstrip("vV")


def same_version(a: Any, b: Any) -> bool:
    pa = _parse_version(normalize_version(a))
    return pa is not None and pa == _parse_version(normalize_version(b))


def default_cache_dir(platform: str | None = None) -> Path | None:
    """App-owned download cache, outside the user's data dir and the app."""
    override = os.environ.get("AGENT_OS_UPDATE_CACHE_DIR")
    if override:
        return Path(override)
    platform = platform or sys.platform
    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / "Orbital" / "updates"
    if platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Orbital" / "updates"
    return None


# ---------------------------------------------------------------------------
# Feed resolution
# ---------------------------------------------------------------------------

FetchJson = Callable[[str], Awaitable[dict]]


async def _httpx_fetch_json(url: str) -> dict:
    import httpx

    headers = {"Accept": "application/json", "User-Agent": "Orbital-Updater"}
    if url.startswith("https://api.github.com/"):
        headers["Accept"] = "application/vnd.github+json"
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _valid_sha(value: Any) -> str:
    sha = str(value or "").strip().lower()
    if not _SHA_RE.match(sha):
        raise UpdateError("feed", "no sha256 for the installer")
    return sha


def _valid_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise UpdateError("feed", "no size for the installer") from None
    if size <= 0:
        raise UpdateError("feed", "no size for the installer")
    return size


async def resolve_oss(offered: str, artifact: Artifact, fetch_json: FetchJson) -> FeedEntry:
    """The China mirror. Its sync is a manual release step, so it can lag:
    anything but exactly the offered version is refused (and the caller falls
    back to GitHub)."""
    try:
        data = await fetch_json(OSS_LATEST_URL)
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("feed", f"mirror unreachable: {type(exc).__name__}") from None
    if not same_version(data.get("version"), offered):
        raise UpdateError("feed", f"mirror serves {data.get('version')!r}, not {offered}")
    entry = ((data.get("files") or {}).get(artifact.feed_key)) or {}
    name = entry.get("name")
    if not isinstance(name, str) or not (
        name == artifact.stable_name
        or ((m := artifact.versioned.match(name)) and same_version(m.group(1), offered))
    ):
        raise UpdateError("feed", f"unexpected mirror file {name!r}")
    return FeedEntry(normalize_version(offered), f"{CHINA_DOWNLOAD_BASE}/{name}",
                     _valid_sha(entry.get("sha256")), _valid_size(entry.get("size")), "china")


async def resolve_github(offered: str, artifact: Artifact, fetch_json: FetchJson) -> FeedEntry:
    """The GitHub release for exactly the offered tag (never ``latest``,
    which may already have moved on), with the asset's sha256 digest."""
    version = normalize_version(offered)
    try:
        data = await fetch_json(GITHUB_TAG_API.format(version=version))
    except UpdateError:
        raise
    except Exception as exc:
        raise UpdateError("feed", f"GitHub unreachable: {type(exc).__name__}") from None
    if not same_version(data.get("tag_name"), offered):
        raise UpdateError("feed", f"GitHub tag {data.get('tag_name')!r} is not {offered}")
    if data.get("draft") or data.get("prerelease"):
        raise UpdateError("feed", "not a published release")
    assets = [a for a in (data.get("assets") or []) if isinstance(a, dict)]
    chosen = next((a for a in assets if a.get("name") == artifact.stable_name), None)
    if chosen is None:
        chosen = next((a for a in assets
                       if (m := artifact.versioned.match(str(a.get("name") or "")))
                       and same_version(m.group(1), offered)), None)
    if chosen is None:
        raise UpdateError("feed", "no installer on the release")
    url = str(chosen.get("browser_download_url") or "")
    if not url.startswith(GITHUB_DOWNLOAD_PREFIX):
        raise UpdateError("feed", "unexpected download host")
    digest = str(chosen.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise UpdateError("feed", "release asset has no sha256 digest")
    return FeedEntry(version, url, _valid_sha(digest[len("sha256:"):]),
                     _valid_size(chosen.get("size")), "global")


_RESOLVERS = {"china": resolve_oss, "global": resolve_github}


def source_order(region: str | None) -> list[str]:
    """The UI language's source first (same rule as the manual download:
    zh -> the China mirror), the other one as the fallback."""
    return ["china", "global"] if region == "china" else ["global", "china"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _download_paths(dest_dir: Path, artifact: Artifact, version: str) -> tuple[Path, Path, Path]:
    final = dest_dir / f"Orbital-{normalize_version(version)}{artifact.suffix}"
    part = final.with_name(final.name + ".part")
    meta = final.with_name(final.name + ".part.json")
    return final, part, meta


def _unlink(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("could not remove %s", p, exc_info=True)


def clean_cache(dest_dir: Path, keep_version: str | None) -> None:
    """Remove every download (finished or partial) except ``keep_version``'s.
    Only our own file names are touched."""
    if not dest_dir.is_dir():
        return
    pattern = re.compile(r"^Orbital-(\d+\.\d+\.\d+)\.(dmg|exe)(\.part(\.json)?)?$")
    for child in dest_dir.iterdir():
        m = pattern.match(child.name)
        if m and not (keep_version and same_version(m.group(1), keep_version)):
            _unlink(child)


def _default_client_factory():
    import httpx

    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=15.0), follow_redirects=True,
        headers={"User-Agent": "Orbital-Updater"})


async def download(entry: FeedEntry, dest_dir: Path, artifact: Artifact, *,
                   on_progress: Callable[[int, int], None] | None = None,
                   client_factory: Callable[[], Any] | None = None) -> Path:
    """Fetch ``entry`` into ``dest_dir``, resuming a matching ``.part``.

    Returns the finished file, whose size and sha256 match the feed. Raises
    ``UpdateError``: ``download`` (network, HTTP, size; the part is kept for a
    resume unless it can no longer match), ``hash`` (the file is deleted) or
    ``space``.
    """
    import httpx

    progress = on_progress or (lambda _r, _t: None)
    client_factory = client_factory or _default_client_factory
    dest_dir.mkdir(parents=True, exist_ok=True)
    final, part, meta_path = _download_paths(dest_dir, artifact, entry.version)

    if final.exists():
        if final.stat().st_size == entry.size and \
                await asyncio.to_thread(sha256_file, final) == entry.sha256:
            progress(entry.size, entry.size)
            return final
        _unlink(final)

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    offset = part.stat().st_size if part.exists() else 0
    if offset and (meta.get("sha256") != entry.sha256 or offset > entry.size):
        _unlink(part, meta_path)
        offset = 0

    needed = (entry.size - offset) + int(entry.size * artifact.extra_space)
    try:
        free = shutil.disk_usage(dest_dir).free
    except OSError:
        free = needed
    if free < needed:
        raise UpdateError("space", f"{free} bytes free, {needed} needed")

    if offset < entry.size:
        headers: dict[str, str] = {}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if meta.get("validator"):
                headers["If-Range"] = meta["validator"]
        try:
            async with client_factory() as client:
                async with client.stream("GET", entry.url, headers=headers) as resp:
                    if resp.status_code == 206 and offset:
                        crange = resp.headers.get("content-range", "")
                        if not crange.startswith(f"bytes {offset}-"):
                            _unlink(part, meta_path)
                            raise UpdateError("download", f"bad Content-Range {crange!r}")
                        mode = "ab"
                    elif resp.status_code == 200:
                        mode, offset = "wb", 0
                    else:
                        if resp.status_code == 416:
                            _unlink(part, meta_path)
                        raise UpdateError("download", f"HTTP {resp.status_code}")
                    length = resp.headers.get("content-length")
                    if length is not None and offset + int(length) != entry.size:
                        _unlink(part, meta_path)
                        raise UpdateError("download", "served size differs from the feed")
                    validator = resp.headers.get("etag") or ""
                    if not validator or validator.startswith("W/"):
                        validator = resp.headers.get("last-modified") or ""
                    meta_path.write_text(json.dumps(
                        {"url": entry.url, "sha256": entry.sha256, "size": entry.size,
                         "validator": validator}), encoding="utf-8")
                    received = offset
                    progress(received, entry.size)
                    with open(part, mode) as fh:
                        # Network-sized chunks: an interrupted download keeps
                        # every byte that arrived (no 256 KiB re-chunk buffer).
                        async for chunk in resp.aiter_bytes():
                            received += len(chunk)
                            if received > entry.size:
                                fh.close()
                                _unlink(part, meta_path)
                                raise UpdateError("download", "more bytes than the feed size")
                            fh.write(chunk)
                            progress(received, entry.size)
        except UpdateError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise UpdateError("download", type(exc).__name__) from None

    size = part.stat().st_size if part.exists() else 0
    if size != entry.size:
        raise UpdateError("download", f"incomplete ({size}/{entry.size})")
    digest = await asyncio.to_thread(sha256_file, part)
    if digest != entry.sha256:
        _unlink(part, meta_path)
        raise UpdateError("hash", f"sha256 {digest} != {entry.sha256}")
    os.replace(part, final)
    _unlink(meta_path)
    return final


# ---------------------------------------------------------------------------
# Hooks from the desktop shell and the daemon
# ---------------------------------------------------------------------------

_shell: dict[str, Callable[[], Any]] = {}
_work_probe: tuple[asyncio.AbstractEventLoop, Callable[[], int]] | None = None


def register_shell(*, confirm_quit: Callable[[], bool], exit_app: Callable[[], Any]) -> None:
    """Called by the desktop shell once its daemon is up. ``confirm_quit``
    is the normal quit question (asks only while work runs); ``exit_app`` is
    the normal way out (ends jobs, flushes telemetry, releases the PID file,
    ``os._exit``). Without them the updater never installs."""
    _shell["confirm_quit"] = confirm_quit
    _shell["exit_app"] = exit_app


def clear_shell() -> None:
    _shell.clear()


def shell_registered() -> bool:
    return "confirm_quit" in _shell and "exit_app" in _shell


def register_work_probe(fn: Callable[[], int],
                        loop: asyncio.AbstractEventLoop | None = None) -> None:
    """``fn`` counts busy projects (agent mid-turn or worker running). It
    reads daemon state that lives on the event loop, so it always runs there."""
    global _work_probe
    _work_probe = (loop or asyncio.get_running_loop(), fn)


def clear_work_probe() -> None:
    global _work_probe
    _work_probe = None


def busy_agent_count(timeout: float = 2.0) -> int:
    """Projects with an agent mid-turn or a worker running, from any thread.
    0 when unknown: a broken probe must never trap the user in the app."""
    probe = _work_probe
    if probe is None:
        return 0
    loop, fn = probe
    try:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            return max(0, int(fn()))
        if loop.is_closed() or not loop.is_running():
            return 0

        async def _call() -> int:
            return int(fn())

        return max(0, asyncio.run_coroutine_threadsafe(_call(), loop).result(timeout))
    except Exception:
        logger.debug("busy probe failed", exc_info=True)
        return 0


def count_busy_projects(project_store, agent_manager, sub_agent_manager=None) -> int:
    """Projects where quitting would cut work off: the manager mid-turn in
    any session, or a worker turn / background job running."""
    n = 0
    for project in list(project_store.list_projects()):
        pid = project.get("project_id") if isinstance(project, dict) else None
        if not pid:
            continue
        busy = False
        try:
            busy = bool(agent_manager.is_running(pid))
        except Exception:
            pass
        if not busy and sub_agent_manager is not None:
            try:
                busy = bool(sub_agent_manager.has_running_sub_agents(pid))
            except Exception:
                pass
        n += int(busy)
    return n


# ---------------------------------------------------------------------------
# Platform step
# ---------------------------------------------------------------------------

class PlatformInstaller(Protocol):
    def unsupported_reason(self) -> str | None:
        """Why this install cannot update itself (None = it can). Cheap."""

    async def prepare(self, downloaded: Path, version: str) -> Any:
        """Unpack + verify. Raises UpdateError(verify|permission|install)."""

    async def handoff(self, prepared: Any, version: str, cache: Path) -> str:
        """Start the out-of-process installer. ``"go"`` = exit now;
        ``"cancelled"`` = the user backed out (stay, old app keeps running).
        Raises UpdateError(install)."""

    def on_startup(self, cache: Path, pending: dict, current: str) -> None:
        """Called at daemon start while an install record exists."""

    def cleanup(self, cache: Path) -> None:
        """Sweep leftovers of a finished or abandoned install."""


def platform_installer(platform: str | None = None) -> PlatformInstaller | None:
    platform = platform or sys.platform
    if platform == "darwin":
        from agent_os import update_macos

        return update_macos.MacInstaller()
    if platform == "win32":
        from agent_os import update_windows

        return update_windows.WindowsInstaller()
    return None


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

PENDING_FILE = "pending.json"
RESULT_FILE = "result.json"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


class Updater:
    def __init__(
        self,
        ws_manager=None,
        *,
        offered: Callable[[], dict] | None = None,
        platform: str | None = None,
        cache_dir: Path | None = None,
        installer: PlatformInstaller | None = None,
        fetch_json: FetchJson | None = None,
        client_factory: Callable[[], Any] | None = None,
        frozen: bool | None = None,
        current_version: Callable[[], str] | None = None,
    ) -> None:
        self._ws = ws_manager
        self._offered = offered or (lambda: {})
        self._platform = platform or sys.platform
        self._artifact = ARTIFACTS.get(self._platform)
        self._cache = cache_dir if cache_dir is not None else default_cache_dir(self._platform)
        self._installer = installer if installer is not None else (
            platform_installer(self._platform) if self._artifact else None)
        self._fetch_json = fetch_json or _httpx_fetch_json
        self._client_factory = client_factory
        self._frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
        self._current = current_version or get_version
        self._task: asyncio.Task | None = None
        self._prepared: tuple[str, Any] | None = None
        self._last_pct = -1
        self._dev_forced: dict | None = None
        self.state: dict = {"state": IDLE, "version": None, "progress": 0,
                            "error": None, "source": None}

    # -- reporting --------------------------------------------------------

    def unsupported_reason(self) -> str | None:
        if self._artifact is None or self._installer is None or self._cache is None:
            return "platform"
        if not self._frozen:
            return "dev"
        if not shell_registered():
            return "no-shell"
        try:
            return self._installer.unsupported_reason()
        except Exception:
            logger.debug("unsupported_reason failed", exc_info=True)
            return "check-failed"

    def snapshot(self) -> dict:
        if self._dev_forced is not None:
            return dict(self._dev_forced)
        return {**self.state, "can_install": self.unsupported_reason() is None}

    def force_dev_state(self, state: dict | None) -> dict:
        """UI review only (the route refuses in a frozen build)."""
        self._dev_forced = None if state is None else {
            "state": state.get("state", IDLE), "version": state.get("version"),
            "progress": int(state.get("progress") or 0), "error": state.get("error"),
            "source": state.get("source"), "can_install": bool(state.get("can_install", True)),
        }
        return self.snapshot()

    def _set(self, state: str, **fields) -> None:
        self.state.update(state=state, **fields)
        if state != DOWNLOADING:
            self._last_pct = -1
        self._broadcast()

    def _broadcast(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.broadcast_global({"type": "update.state", **self.snapshot()})
        except Exception:
            logger.debug("update.state broadcast failed", exc_info=True)

    def _on_progress(self, received: int, total: int) -> None:
        pct = int(received * 100 / total) if total else 0
        self.state["progress"] = min(100, pct)
        if pct != self._last_pct:
            self._last_pct = pct
            self._broadcast()

    # -- start ------------------------------------------------------------

    def _offered_version(self) -> str:
        status = self._offered() or {}
        latest = status.get("latest")
        if not status.get("update_available") or not latest:
            raise UpdateError("not_newer", "no update offered")
        if not is_newer(str(latest), self._current()):
            raise UpdateError("not_newer", f"{latest} is not newer than {self._current()}")
        return normalize_version(latest)

    async def start(self, region: str | None = None) -> dict:
        """The pill click. Idempotent while a run is in flight."""
        if self.state["state"] in ACTIVE_STATES and self._task and not self._task.done():
            return self.snapshot()
        reason = self.unsupported_reason()
        if reason is not None:
            raise UpdateError("unsupported", reason)
        offered = self._offered_version()
        self._task = asyncio.create_task(self._run(offered, region), name="updater")
        await asyncio.sleep(0)
        return self.snapshot()

    async def _run(self, offered: str, region: str | None) -> None:
        try:
            if self._prepared is not None and same_version(self._prepared[0], offered) \
                    and self.state["state"] == READY:
                await self._install(offered, self._prepared[1])
                return
            self._prepared = None
            self._set(DOWNLOADING, version=offered, progress=0, error=None, source=None)
            path = await self._download_with_fallback(offered, region)
            self._set(VERIFYING, progress=100)
            try:
                prepared = await self._installer.prepare(path, offered)
            except UpdateError as exc:
                if exc.code == "verify":
                    _unlink(path)  # a refused installer never passes; free the space
                raise
            self._prepared = (offered, prepared)
            self._set(READY)
            await self._install(offered, prepared)
        except UpdateError as exc:
            logger.warning("update to %s failed: %s", offered, exc)
            self._fail(exc.code)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("update to %s failed", offered)
            self._fail("install")

    def _fail(self, code: str) -> None:
        self._prepared = None
        self._set(FAILED, error=code)

    async def _download_with_fallback(self, offered: str, region: str | None) -> Path:
        assert self._cache is not None and self._artifact is not None
        await asyncio.to_thread(clean_cache, self._cache, offered)
        errors: list[UpdateError] = []
        for source in source_order(region):
            try:
                entry = await _RESOLVERS[source](offered, self._artifact, self._fetch_json)
                self.state["source"] = source
                return await download(entry, self._cache, self._artifact,
                                      on_progress=self._on_progress,
                                      client_factory=self._client_factory)
            except UpdateError as exc:
                logger.info("update source %s failed: %s", source, exc)
                errors.append(exc)
                if exc.code == "space":
                    break  # the other source needs the same space
        # Report the most telling failure: a bad hash or no space beats a
        # mirror that simply was not in sync.
        for code in ("space", "hash", "download"):
            for exc in errors:
                if exc.code == code:
                    raise exc
        raise errors[-1] if errors else UpdateError("feed")

    async def _install(self, version: str, prepared: Any) -> None:
        confirm = _shell.get("confirm_quit")
        exit_app = _shell.get("exit_app")
        if confirm is None or exit_app is None:
            raise UpdateError("unsupported", "no-shell")
        if not await asyncio.to_thread(confirm):
            self._set(READY)  # the user kept working; the pill offers a restart
            return
        self._set(INSTALLING)
        assert self._cache is not None
        _write_json(self._cache / PENDING_FILE, {
            "target": version, "previous": self._current(), "at": time.time()})
        _unlink(self._cache / RESULT_FILE)
        try:
            outcome = await self._installer.handoff(prepared, version, self._cache)
        except BaseException:
            _unlink(self._cache / PENDING_FILE)
            raise
        if outcome != "go":
            _unlink(self._cache / PENDING_FILE)
            self._set(READY)
            return
        await asyncio.sleep(0.3)  # let the "installing" state reach the pill
        await asyncio.to_thread(exit_app)

    # -- daemon start -----------------------------------------------------

    def on_startup(self) -> None:
        """Reconcile what a previous install left behind. Never raises. A dev
        daemon shares the cache folder with the installed app, so it never
        touches it."""
        if self._cache is None or self._installer is None or not self._frozen:
            return
        try:
            current = self._current()
            pending_path, result_path = self._cache / PENDING_FILE, self._cache / RESULT_FILE
            pending = _read_json(pending_path)
            fresh = bool(pending) and \
                time.time() - float(pending.get("at") or 0) < _RECORD_MAX_AGE_S
            if fresh:
                self._installer.on_startup(self._cache, pending, current)
                result = _read_json(result_path)
                if same_version(pending.get("target"), current):
                    # The new version is up: the install worked.
                    _unlink(pending_path, result_path)
                    clean_cache(self._cache, None)
                elif same_version(pending.get("previous"), current) and (
                        result is not None or self._platform == "win32"):
                    # The old version is back: the macOS helper rolled back
                    # (and said so in result.json), or Windows Setup aborted
                    # and relaunched what was installed.
                    rolled_back = (result or {}).get("result") == "rolled_back"
                    self.state.update(state=FAILED, version=pending.get("target"),
                                      error="launch" if rolled_back else "install")
                    _unlink(pending_path, result_path)
                elif not same_version(pending.get("previous"), current):
                    _unlink(pending_path, result_path)
                # else: old version, no verdict yet: a helper may still be
                # working; leave its records alone.
            elif pending is not None:
                _unlink(pending_path, result_path)
            if not pending_path.exists():
                self._installer.cleanup(self._cache)
            # Downloads of a version we already run (or older) are dead weight.
            if self._cache.is_dir():
                for child in self._cache.iterdir():
                    m = re.match(r"^Orbital-(\d+\.\d+\.\d+)\.", child.name)
                    if m and not is_newer(m.group(1), current):
                        _unlink(child)
            # Warm the platform check here, off the event loop: on macOS its
            # first call runs ``codesign -dv`` on the running bundle.
            self.unsupported_reason()
        except Exception:
            logger.debug("updater startup reconcile failed", exc_info=True)


_updater: Updater | None = None


def configure(ws_manager, **kwargs) -> Updater:
    global _updater
    _updater = Updater(ws_manager, **kwargs)
    return _updater


def get_updater() -> Updater | None:
    return _updater


__all__ = [
    "ARTIFACTS", "FeedEntry", "UpdateError", "Updater", "busy_agent_count",
    "clean_cache", "configure", "count_busy_projects", "default_cache_dir",
    "download", "get_updater", "register_shell", "register_work_probe",
    "resolve_github", "resolve_oss", "source_order",
]
