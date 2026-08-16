# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""On-demand installation of Orbital-managed sub-agents.

Most agents are bring-your-own: the user installs the CLI, Orbital finds it on
PATH. An agent that declares ``setup.orbital_install.platforms`` is the other
kind — Orbital owns its install, dropping a pinned npm composition into
``<data_dir>/agents/<slug>/`` on request. Nothing enters the ``.dmg``/``.exe``:
the native binaries in that tree never pass through Developer-ID signing or
notarization, and an 84 MB payload has no business in an installer.

Two properties are load-bearing:

* **``npm ci``, never ``npm install``.** The committed lockfile is the version
  freeze. Upstream's dist-tags are inverted (``latest`` resolves to an ancient
  ``0.0.1-rc.1``) and every intra-tree dependency is a ``^`` range, so an
  unpinned resolution installs a different generation without saying so.
* **The subprocess runs on the main loop.** ``broadcast_global`` is not
  thread-safe (``api/ws.py:141``) and silently drops payloads from a foreign
  thread, so progress streaming must never be moved to a thread executor.

``.install-complete`` is written only after a clean run. Its absence means the
tree is partial, and re-running repairs: assets are re-copied (refreshing a
stale template) and ``npm ci`` deletes ``node_modules`` before rebuilding it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from agent_os.daemon_v2.models import detect_os
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

# Written into the agent's install dir once, on success.
MARKER_FILENAME = ".install-complete"

# ``dsh`` calls ``process.loadEnvFile``, added in Node 20.12. Upstream declares
# no ``engines`` field, so this floor is ours to enforce — and to explain.
MIN_NODE_VERSION = (20, 12)

# Required in the bundled assets dir; their absence is a broken build, not a
# user problem. The manifest's ``runtime.config_template`` joins them when
# declared. Everything else in that directory is staged too — see
# ``_stage_assets`` — these are just the ones worth a clear error.
_BASE_ASSETS = ("package.json", "package-lock.json")

# Never copied out of the assets dir: npm builds the install tree itself, and
# a marker only ever means something in the target.
_NEVER_STAGED = frozenset({"node_modules", MARKER_FILENAME, "__pycache__"})


class InstallError(Exception):
    """The install could not be completed. The message reaches the user."""


class InstallUnsupported(InstallError):
    """This agent cannot be Orbital-installed on this platform (or at all)."""


class InstallInProgress(InstallError):
    """A job for this slug is already running."""


class SubAgentInstaller:
    """Runs and tracks one install job per agent slug.

    The job registry is in-memory and per-daemon: a restart mid-install leaves
    no ``.install-complete`` behind, which is exactly the "partial, re-run"
    signal the next status read needs.
    """

    def __init__(self, setup_engine=None, ws_manager=None) -> None:
        self._setup_engine = setup_engine
        self._ws_manager = ws_manager
        self._jobs: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, slug: str, manifest) -> str:
        """Schedule an install job and return its id immediately.

        Raises ``InstallUnsupported`` (this platform / not Orbital-installable)
        or ``InstallInProgress`` (duplicate). Everything that can only fail
        once npm is running reaches the caller as a ``sub_agent_install_failed``
        event instead.
        """
        self._require_supported(manifest)

        existing = self._jobs.get(slug)
        if existing is not None and existing["state"] == "installing":
            raise InstallInProgress(
                f"an install of '{slug}' is already running "
                f"(job {existing['job_id']})"
            )

        job_id = uuid4().hex[:12]
        self._jobs[slug] = {"job_id": job_id, "state": "installing"}
        self._tasks[slug] = asyncio.create_task(
            self._run(slug, manifest, job_id))
        return job_id

    def get_state(self, slug: str) -> dict | None:
        """The live job record for ``slug``, or None if it never ran here."""
        job = self._jobs.get(slug)
        return dict(job) if job is not None else None

    def supports(self, manifest) -> bool:
        """Whether Orbital can install this agent on the current platform.

        False for every bring-your-own agent (no ``orbital_install`` block) and
        for a declared agent on a platform it left out.
        """
        return detect_os() in (manifest.setup.orbital_install.platforms or [])

    def entry_for(self, manifest, *, installed: bool) -> dict:
        """The ``install`` block of a ``GET /settings/sub-agents`` entry.

        Refresh-safe by construction: derived from the live job registry, the
        marker file, and the binary probe — never from WS state a reloading
        client may have missed.

        ``platforms`` echoes the manifest so a client can tell the two kinds of
        ``supported: false`` apart: an empty list means the agent is
        bring-your-own everywhere, while a non-empty one that excludes this
        host means "Orbital installs this, just not here" — the only case that
        warrants a "not yet supported on this platform" line.
        """
        slug = manifest.slug
        platforms = list(manifest.setup.orbital_install.platforms or [])
        supported = self.supports(manifest)
        job = self._jobs.get(slug)
        job_id = job["job_id"] if job is not None else None

        if job is not None and job["state"] == "installing":
            return {"supported": supported, "platforms": platforms,
                    "state": "installing", "job_id": job_id}

        # For an Orbital-managed agent the marker is the completion record: a
        # binary can exist in a tree whose npm run was interrupted.
        complete = not platforms or self._marker(slug) is not None
        if installed and complete:
            state = "installed"
        elif job is not None and job["state"] == "failed":
            state = "failed"
        else:
            state = "not_installed"

        entry = {"supported": supported, "platforms": platforms,
                 "state": state}
        if job_id is not None:
            entry["job_id"] = job_id
        return entry

    async def wait(self, slug: str) -> None:
        """Await the in-flight job for ``slug``; no-op when there is none."""
        task = self._tasks.get(slug)
        if task is not None:
            await task

    # ------------------------------------------------------------------
    # The job
    # ------------------------------------------------------------------

    async def _run(self, slug: str, manifest, job_id: str) -> None:
        try:
            node, npm = await self._preflight(manifest)
            target = self._stage_assets(slug, manifest)
            await self._npm_ci(npm, target, slug, job_id)
            version = _pinned_version(os.path.join(target, "package.json"))
            _write_marker(target, version)
        except InstallError as exc:
            self._finish_failed(slug, job_id, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a job must never leave the UI hanging
            logger.exception("install job failed for %s", slug)
            self._finish_failed(slug, job_id, f"{type(exc).__name__}: {exc}")
            return
        self._finish_done(slug, job_id, version)

    async def _preflight(self, manifest) -> tuple[str, str]:
        """Resolve node + npm and enforce the Node floor.

        The packaged app merges the login-shell PATH at the desktop entrypoint
        (``desktop/main.py``), and a dev daemon inherits its invoking shell — a
        bare ``uvicorn`` started from a GUI context gets neither, which is the
        one case where a user has Node installed and we still can't see it. The
        message has to name that, or the report reads as "Orbital can't find
        the Node I just installed".
        """
        requirement = (
            f"Installing {manifest.name} needs Node.js "
            f"{MIN_NODE_VERSION[0]}.{MIN_NODE_VERSION[1]}+ and npm"
        )
        node = shutil.which("node")
        if not node:
            raise InstallError(
                f"{requirement}, but no 'node' was found on the daemon's PATH. "
                f"Install Node.js and restart Orbital so it picks up your "
                f"shell PATH."
            )
        npm = shutil.which("npm")
        if not npm:
            raise InstallError(
                f"{requirement}, but no 'npm' was found on the daemon's PATH "
                f"(node resolved to {node}). Install npm and restart Orbital "
                f"so it picks up your shell PATH."
            )

        raw = await _run_capture(node, "--version")
        parsed = _parse_node_version(raw)
        if parsed is None:
            raise InstallError(
                f"{requirement}, but '{node} --version' reported "
                f"{raw.strip()!r}, which is not a version this can read."
            )
        version, found = parsed
        if version < MIN_NODE_VERSION:
            raise InstallError(
                f"{requirement}, but {node} is {found}. Upgrade Node.js (or "
                f"put a newer one first on the daemon's PATH) and try again."
            )
        return node, npm

    def _stage_assets(self, slug: str, manifest) -> str:
        """Copy the bundled composition assets into the agent's install dir.

        The whole bundled tree is mirrored, not a fixed file list: a
        composition may carry local packages beside its manifest (a plugin the
        template loads by path, or a ``file:`` dependency ``npm ci`` resolves),
        and staging only the three named files would leave ``npm ci`` pointing
        at a directory that never arrived. ``_BASE_ASSETS`` plus the manifest's
        template are still *required* — their absence is a broken build, and
        saying so beats an inscrutable npm error.

        Re-copied on every run, so a re-install after a partial one refreshes a
        template that may have been half-written or shipped by an older build.
        """
        source = _resolve_assets_dir(slug)
        if source is None:
            raise InstallError(
                f"the bundled install assets for '{slug}' are missing from "
                f"this build; reinstall Orbital"
            )
        target = self._target_dir(slug)
        os.makedirs(target, exist_ok=True)

        required = list(_BASE_ASSETS)
        if manifest.runtime.config_template:
            required.append(manifest.runtime.config_template)
        for name in required:
            if not os.path.isfile(os.path.join(source, name)):
                raise InstallError(
                    f"bundled asset '{name}' is missing for '{slug}'; "
                    f"reinstall Orbital"
                )

        for name in sorted(os.listdir(source)):
            # A dev who ran npm in the source tree must not have 84 MB of it
            # copied over the install npm is about to build.
            if name in _NEVER_STAGED:
                continue
            src = os.path.join(source, name)
            dst = os.path.join(target, name)
            if os.path.isdir(src):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
            else:
                shutil.copyfile(src, dst)

        # A stale marker would claim the tree is complete while npm is still
        # rebuilding it. Drop it now; it is rewritten only on success.
        marker = os.path.join(target, MARKER_FILENAME)
        if os.path.exists(marker):
            os.remove(marker)
        return target

    async def _npm_ci(self, npm: str, target: str, slug: str,
                      job_id: str) -> None:
        """Run ``npm ci --omit=dev``, streaming its output line by line.

        NEVER ``npm install``: the lockfile is the freeze, and upstream's
        inverted dist-tags make an unpinned resolution silently wrong.
        """
        argv = [npm, "ci", "--omit=dev"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                creationflags=win_no_window_flags(),
            )
        except OSError as exc:
            raise InstallError(f"could not run '{' '.join(argv)}': {exc}")

        tail: list[str] = []
        if proc.stdout is not None:
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-5]
                self._broadcast({
                    "type": "sub_agent_install_progress",
                    "slug": slug,
                    "job_id": job_id,
                    "line": line,
                })

        rc = await proc.wait()
        if rc != 0:
            detail = " / ".join(tail) if tail else "no output"
            raise InstallError(
                f"npm ci failed with exit code {rc}: {detail}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_supported(self, manifest) -> None:
        platforms = list(manifest.setup.orbital_install.platforms or [])
        if not platforms:
            raise InstallUnsupported(
                f"'{manifest.slug}' is not installed by Orbital — install its "
                f"CLI yourself and Orbital will detect it"
            )
        current = detect_os()
        if current not in platforms:
            raise InstallUnsupported(
                f"installing '{manifest.slug}' is not supported on {current} "
                f"yet (supported: {', '.join(platforms)})"
            )

    def _target_dir(self, slug: str) -> str:
        data_dir = getattr(self._setup_engine, "data_dir", None)
        if not data_dir:
            raise InstallError(
                "the setup engine has no data dir; cannot choose an install "
                "location"
            )
        return os.path.join(data_dir, "agents", slug)

    def _marker(self, slug: str) -> dict | None:
        """The parsed ``.install-complete``, or None when absent/unreadable."""
        try:
            path = os.path.join(self._target_dir(slug), MARKER_FILENAME)
        except InstallError:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _finish_done(self, slug: str, job_id: str, version: str) -> None:
        self._jobs[slug] = {"job_id": job_id, "state": "installed"}
        engine = self._setup_engine
        # The binary the auto_detect probe is about to find did not exist when
        # this agent was last checked, and both caches outlive that fact — the
        # resolved-path one for the daemon's whole lifetime.
        if engine is not None:
            if hasattr(engine, "invalidate_cache"):
                engine.invalidate_cache()
            if hasattr(engine, "invalidate_resolved_path"):
                engine.invalidate_resolved_path(slug)
        self._broadcast({
            "type": "sub_agent_install_done",
            "slug": slug,
            "job_id": job_id,
            "version": version,
        })

    def _finish_failed(self, slug: str, job_id: str, error: str) -> None:
        self._jobs[slug] = {"job_id": job_id, "state": "failed",
                            "error": error}
        logger.warning("sub-agent install failed for %s: %s", slug, error)
        self._broadcast({
            "type": "sub_agent_install_failed",
            "slug": slug,
            "job_id": job_id,
            "error": error,
        })

    def _broadcast(self, payload: dict) -> None:
        if self._ws_manager is None:
            return
        try:
            self._ws_manager.broadcast_global(payload)
        except Exception:  # noqa: BLE001 - a dead client never fails an install
            logger.warning("failed to broadcast %s", payload.get("type"),
                           exc_info=True)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _resolve_assets_dir(slug: str) -> str | None:
    """Locate the bundled ``agent_os/agents/assets/<slug>/`` directory.

    Same frozen-path shape as the manifest directory (``api/app.py``): in a
    PyInstaller bundle the data files live under ``sys._MEIPASS``, not next to
    ``__file__``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass and getattr(sys, "frozen", False):
        candidate = Path(meipass) / "agent_os" / "agents" / "assets" / slug
    else:
        candidate = Path(__file__).resolve().parent / "assets" / slug
    return str(candidate) if candidate.is_dir() else None


async def _run_capture(*argv: str) -> str:
    """Run a short command and return its merged output."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=win_no_window_flags(),
        )
        out, _ = await proc.communicate()
    except OSError as exc:
        raise InstallError(f"could not run '{' '.join(argv)}': {exc}")
    return (out or b"").decode("utf-8", errors="replace")


def _parse_node_version(raw: str) -> tuple[tuple[int, int], str] | None:
    """``v24.3.0`` → ``((24, 3), "24.3.0")``.

    The comparable pair enforces the floor; the display string is what the
    failure message quotes back, so a user who runs ``node --version`` sees
    the same text Orbital did. None when the output isn't a version at all.
    """
    match = re.search(r"(\d+)\.(\d+)(?:\.\S+)?", raw or "")
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2))), match.group(0)


def _pinned_version(package_json_path: str) -> str:
    """The composition's pinned generation, for the completion marker.

    Every upstream dependency is pinned to one exact version by design, so the
    single distinct value IS the generation. Non-registry specs (``file:`` /
    ``link:`` shims vendored beside the composition) carry no generation and
    are ignored. If the remaining pins ever diverge, fall back to the
    composition package's own version rather than picking a winner.
    """
    try:
        with open(package_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return ""
    deps = data.get("dependencies") or {}
    versions = {
        v for v in deps.values()
        if isinstance(v, str) and v[:1].isdigit()
    }
    if len(versions) == 1:
        return versions.pop()
    return str(data.get("version", ""))


def _write_marker(target: str, version: str) -> None:
    """Record a completed install. Absence means partial — re-run repairs."""
    with open(os.path.join(target, MARKER_FILENAME), "w",
              encoding="utf-8") as f:
        json.dump({"version": version}, f)
