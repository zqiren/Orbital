# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setup engine — auto-detect, dependency check, and credential resolution.

Reads agent manifests from the registry, probes the system for installed
binaries and satisfied dependencies, and reports what is available.
"""

import concurrent.futures
import copy
import dataclasses
import glob
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import time

from agent_os.agents.manifest import AgentManifest
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_types import AgentSetupStatus, SetupAction
from agent_os.daemon_v2.models import detect_os
from agent_os.telemetry.diagnostics import classify_stderr, error_class_of
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)


# Cache TTL for SetupEngine.check_all() results in seconds. The check_all path
# spawns one subprocess per agent (auth-status check, install probes, etc.),
# which can take several seconds on a cold call. The cache keeps the response
# snappy for back-to-back requests; explicit invalidation clears it after any
# action that could change the result (install, login, refresh).
CHECK_ALL_CACHE_TTL_SECONDS = 60

# Spec 136: the sweep runs every agent's probes at once on a pool of its own
# (never asyncio's default executor, which the routes' ``to_thread`` work
# shares), so a cold sweep costs the slowest agent instead of the sum.
SWEEP_MAX_WORKERS = 8

# The one placeholder in the manifest system, expanded in ``setup.auto_detect``
# entries only. Agents Orbital installs itself land under the daemon's data dir,
# whose location is a runtime fact (dev daemons, the packaged app, and tests all
# differ) and so cannot be written into the read-only bundled manifest.
# ``runtime.command`` and ``runtime.args`` stay strict passthrough.
ORBITAL_DATA_DIR_TOKEN = "${ORBITAL_DATA_DIR}"


class StaleOkView:
    """A ``check_all()`` view that serves an expired result instead of
    re-probing (spec 101), for code that takes an engine and calls plain
    ``check_all()`` on the event loop (``resolve_visible_sub_agent_slugs``)."""

    def __init__(self, engine) -> None:
        self._engine = engine

    def check_all(self):
        return self._engine.check_all(allow_stale=True)


# What an agent's dispatch prints when its sign-in no longer works. Claude
# Code: "Failed to authenticate. API Error: 401 … authentication_error …
# OAuth token has expired", "OAuth session expired", "… could not be
# refreshed", "Invalid API key · Please run /login". Matched against error
# text only, never against a successful turn's output.
_AUTH_FAILURE_RE = re.compile(
    r"failed to authenticate"
    r"|authentication_error"
    r"|oauth (?:session|token) (?:has )?expired"
    r"|could not be refreshed"
    r"|please run /login",
    re.IGNORECASE,
)


def is_auth_failure_text(text: str | None) -> bool:
    """Whether a failed dispatch's error text says the sign-in is no good."""
    return bool(text) and _AUTH_FAILURE_RE.search(text) is not None


def _json_field(data, path: str):
    """Value at ``path`` in ``data``: a top-level key, or a dotted path.

    ``providers.managed:kimi-code.type`` walks nested dicts one segment at a
    time (segments may contain anything but a dot). Missing or non-dict
    intermediates yield "" so the caller's string comparison simply fails.
    """
    current = data
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return ""
        current = current[segment]
    return current


def _natural_key(path: str) -> list:
    """Sort key that orders ``v9`` < ``v20`` < ``v22`` (lexical would not)."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", path)]


def _mtime(path: str) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def _expand_auto_detect_entry(expanded: str) -> list[str]:
    """Spec 115 D3: an ``auto_detect`` entry containing ``*`` is a glob —
    ``$HOME/.nvm/versions/node/*/bin/codex``. Spec 144: matches come
    newest-modified first, because a folder name (the Codex app's
    ``bin/<hash>/``) says nothing about age; equal times fall back to natural
    name order, so nvm's highest version still wins a tie. Plain entries are
    returned as they are."""
    if "*" not in expanded:
        return [expanded]
    return sorted(glob.glob(expanded),
                  key=lambda p: (_mtime(p), _natural_key(p)), reverse=True)


_daemon_path_lock = threading.Lock()


def _append_to_daemon_path(directory: str) -> None:
    """Append *directory* to ``os.environ["PATH"]`` once (spec 115 D3 rule).

    Appended, never prepended: the user's own PATH order stays authoritative
    for anything it already resolves. Locked: the parallel sweep resolves
    several agents at once, and two read-modify-writes would lose one."""
    if not directory:
        return
    with _daemon_path_lock:
        entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        if directory in entries:
            return
        entries.append(directory)
        os.environ["PATH"] = os.pathsep.join(entries)


class _SweepMemo:
    """Answers shared by every agent in one sweep (spec 136 B.2): ten
    manifests ask ``node --version``, and it runs once. Single-flight per
    key, so agents probed in parallel wait for the first ask."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[tuple, concurrent.futures.Future] = {}

    def get(self, key: tuple, compute):
        with self._lock:
            future = self._futures.get(key)
            owner = future is None
            if owner:
                future = concurrent.futures.Future()
                self._futures[key] = future
        if owner:
            try:
                future.set_result(compute())
            except BaseException as exc:  # noqa: BLE001 — handed to every waiter
                future.set_exception(exc)
        return future.result()


def _status_digest(results) -> str:
    """A stable digest of a sweep, to tell a changed result from a repeat."""
    def plain(obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return getattr(obj, "__dict__", str(obj))
    payload = json.dumps([plain(r) for r in results], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def agent_store_path(store: dict, env: dict | None = None) -> str | None:
    """The file an agent keeps its own credentials in, per the manifest's
    ``agent_store`` — ``$<home_env>/<basename>`` when that variable is set,
    else ``file`` with ``~`` expanded."""
    raw = (store or {}).get("file")
    if not raw:
        return None
    env = os.environ if env is None else env
    home_env = store.get("home_env")
    if home_env and env.get(home_env):
        return os.path.join(env[home_env], os.path.basename(raw))
    return os.path.expanduser(raw)


def agent_store_has(store: dict, env: dict | None = None) -> bool:
    """Whether the agent's own credential store holds a non-empty value at
    ``store["key"]`` (a dotted path into its YAML document).

    Presence only: the value is never returned or logged. Unreadable,
    malformed or absent → False, and the Orbital-held key / env decide.
    """
    path = agent_store_path(store, env)
    key = (store or {}).get("key")
    if not path or not key or not os.path.isfile(path):
        return False
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            node = yaml.safe_load(f)
    except Exception:  # noqa: BLE001 — never surface a parse error with secrets in it
        logger.warning("agent credential store unreadable: %s", path)
        return False
    for part in key.split("."):
        if not isinstance(node, dict):
            return False
        node = node.get(part)
    return isinstance(node, str) and bool(node.strip())


def spawn_fingerprint(config_dict: dict, params: dict | None) -> str:
    """A digest of everything a spawn is launched with.

    Binary, argv, the env Orbital injects (credentials, locate-card PATH) and
    the agent's Settings params — including UI-only ones such as dsh's model,
    which never reach argv. A live worker whose recorded digest differs from
    the current one was spawned with settings that have since changed (a key
    saved 2026-10-26 never reached the running dsh process). Only the digest
    is kept, so no secret is held beside the adapter.
    """
    payload = json.dumps({
        "command": config_dict.get("command"),
        "args": list(config_dict.get("args") or []),
        "env": dict(sorted((config_dict.get("env") or {}).items())),
        "params": dict(sorted((params or {}).items())),
        # What a rendered file copies from the user's own setup (dsh's
        # providers in DSH Desktop): editing them there rebuilds the worker.
        "composition_inputs": config_dict.get("composition_inputs"),
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SetupEngine:
    """Probes the system for agent readiness based on manifest metadata."""

    def __init__(
        self,
        registry: AgentRegistry,
        credential_store=None,
        sub_agent_config_store=None,
        data_dir: str = "orbital-data",
        card_store=None,
    ) -> None:
        self._registry = registry
        self._credential_store = credential_store
        self._sub_agent_config_store = sub_agent_config_store
        # The user's credential cards (SettingsStore). Read-only here: a
        # ``model_provider`` credential borrows ONE card key for the provider
        # of the agent's configured model.
        self._card_store = card_store
        # Absolutised once: auto_detect probes and the composition renderer
        # both need a path that survives a cwd change mid-daemon.
        self._data_dir = os.path.abspath(data_dir)
        self._resolved_paths: dict[str, str] = {}  # slug -> resolved binary path
        # The locate card (user-chosen location). ``_override_hits`` holds
        # the slugs whose resolved path came from the override;
        # ``_override_missing`` the override path when its file is gone —
        # reported, never silently replaced by a PATH hit (the stale-path
        # bug hid behind exactly that fallback).
        self._override_hits: set[str] = set()
        self._override_missing: dict[str, str] = {}
        # Where each resolved binary came from: {"kind": "override" |
        # "desktop_app" | "path" | "auto_detect" | "check_command", plus
        # "label"/"app_path"/"app_version" for a desktop app}. The card says
        # which copy Orbital drives.
        self._resolved_sources: dict[str, dict] = {}
        # ACP handshake verdicts keyed by the binary's identity (path, mtime,
        # size, app version): re-run only when the binary changes, so a
        # 60 s status sweep never re-spawns every agent.
        self._handshakes: dict[tuple, dict] = {}
        # In-flight handshakes, one per identity. A sweep starts every ACP
        # agent's handshake up front so they overlap with the other agents'
        # version and auth probes instead of adding ~1-3 s each in series
        # (spec 136: the sweep is already the slow part of Settings).
        self._handshake_futures: dict[tuple, "concurrent.futures.Future"] = {}
        self._handshake_lock = threading.Lock()
        self._sweep_local = threading.local()
        self._handshake_pool: "concurrent.futures.ThreadPoolExecutor | None" = None
        # In-memory cache of the last check_all() result. Tuple of
        # (results, expires_at_monotonic). None when empty or invalidated.
        self._check_all_cache: tuple[list[AgentSetupStatus], float] | None = None
        # Single-flight guard for the probe sweep (spec 101): concurrent cold
        # callers — threaded routes, a stale-serve refresh — wait for ONE
        # sweep instead of each spawning ~13 CLI subprocesses.
        self._probe_lock = threading.Lock()
        # Bumped by invalidate_cache(). A sweep that started before an
        # invalidation may have probed an agent before the install/login
        # finished, so its result is returned to its caller but not cached.
        self._cache_generation = 0
        self._refresh_thread: threading.Thread | None = None
        self._sweep_pool: concurrent.futures.ThreadPoolExecutor | None = None
        # Spec 136 B.3: ``--version`` output keyed by the binary's identity
        # (realpath, mtime, size, app version, the check command and the
        # locate card's PATH). A CLI's version only changes when its file
        # does, so a sweep after the first skips these subprocesses. Kept for
        # the daemon's lifetime; Refresh drops it (``forget_versions``).
        self._version_cache: dict[tuple, str] = {}
        # Spec 144: whether a candidate binary starts, keyed like the version
        # cache, so a sweep launches each copy once per change of its file.
        self._start_checks: dict[tuple, bool] = {}
        # Spec 136 B.4: slugs whose binary was not found, with when that
        # answer expires (the sweep TTL). Read by sweeps only, so a missing
        # agent stops spawning its check command through a shell every sweep
        # while a single-agent check (locate, install done) looks afresh.
        self._resolved_misses: dict[str, float] = {}
        # Spec 136 decision 2: the last answer each ``oauth_cli`` probe gave,
        # kept when a later probe times out (Cursor's ``agent status`` hangs
        # several times a day; a timeout used to read as signed out).
        self._auth_verdicts: dict[tuple[str, str], bool] = {}
        # Told when a sweep (or ``refresh_agent``) caches a result that
        # differs from the last one; the app broadcasts it on the WS.
        self._status_listener = None
        self._published_digest: str | None = None
        # Spec 113: what the last ``oauth_cli`` auth probe for a slug came
        # back with — exit code, exception class, stderr category, duration —
        # so a dispatch refused for "not signed in" can carry structured
        # diagnostics without re-running the probe. Enums and ints only.
        self._last_auth_probe: dict[str, dict] = {}
        # Slugs whose last dispatch failed with an authentication error
        # (see ``note_auth_failure``). In-memory: a restart forgets it, and
        # the next failing dispatch records it again.
        self._auth_failures: set[str] = set()

    # ------------------------------------------------------------------
    # Expired sign-in record (1018 bench item 9)
    # ------------------------------------------------------------------

    def note_auth_failure(self, slug: str, text: str | None) -> bool:
        """Record that ``slug``'s dispatch failed with an authentication
        error, when ``text`` says so. Returns whether it was recorded.

        The ``oauth_cli`` probe reads the CLI's local token file, so an
        expired token still probes as signed in; the failed dispatch is the
        only evidence, and no network call is added to the probe. Drops the
        cached sweep so the card turns on its next read.
        """
        if not is_auth_failure_text(text):
            return False
        if slug not in self._auth_failures:
            self._auth_failures.add(slug)
            self._patch_auth_record(slug)
            logger.info("%s: dispatch failed with an authentication error; "
                        "marking its sign-in as expired", slug)
        return True

    def clear_auth_failure(self, slug: str) -> None:
        """Forget the expired-sign-in record: a turn succeeded, a login was
        confirmed, or the user logged out."""
        if slug in self._auth_failures:
            self._auth_failures.discard(slug)
            self._patch_auth_record(slug)

    def _patch_auth_record(self, slug: str) -> None:
        """Turn the cached entry's ``auth_expired`` to match the record.

        The record changes nothing a probe reads, so the sweep is patched
        instead of dropped (a dropped sweep re-probes every agent on the next
        Settings read). A sweep in flight may already have read the old
        record: then drop it, as before, so its result is not cached.
        """
        if self._probe_lock.locked():
            self.invalidate_cache()
            return
        cached = self._check_all_cache
        if cached is None:
            return
        results = []
        for status in cached[0]:
            if getattr(status, "slug", None) == slug:
                status = copy.copy(status)
                status.auth_expired = bool(
                    getattr(status, "installed", False)
                    and getattr(status, "credentials_configured", False)
                    and slug in self._auth_failures)
            results.append(status)
        self._check_all_cache = (results, cached[1])
        self._publish(results)

    def auth_expired(self, slug: str) -> bool:
        return slug in self._auth_failures

    def set_sub_agent_config_store(self, store) -> None:
        """Late-bind the sub-agent config store (called by app factory)."""
        self._sub_agent_config_store = store

    @property
    def data_dir(self) -> str:
        """Absolute path of the daemon's data dir (where agents are installed)."""
        return self._data_dir

    @property
    def sub_agent_config_store(self):
        """The bound sub-agent config store, or None."""
        return self._sub_agent_config_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_agent(self, slug: str, *,
                    handshake_wait: float | None = None) -> AgentSetupStatus:
        """Full read-only status check for one agent.

        ``handshake_wait``: how long to wait for an ACP agent's handshake
        verdict when none is cached — None waits for it (a single-agent
        check, the user's Refresh); the routine sweep passes 0 and reports a
        pending handshake as "unknown" rather than stall Settings.
        """
        manifest = self._registry.get(slug)
        if manifest is None:
            raise ValueError(f"Unknown agent: {slug}")

        # Built-in agents are always installed
        if manifest.runtime.adapter == "built_in":
            return AgentSetupStatus(
                slug=manifest.slug,
                name=manifest.name,
                installed=True,
                binary_path=None,
                version=manifest.version,
                dependencies_met=True,
                missing_dependencies=[],
                credentials_configured=True,
                missing_credentials=[],
                setup_actions=[],
            )

        # Resolve binary
        binary = self.resolve_binary(manifest)
        installed = binary is not None
        override = self._override_for(manifest.slug)
        override_missing = manifest.slug in self._override_missing
        override_path = (self._override_missing.get(manifest.slug)
                         or (override["binary"] if override and manifest.slug in self._override_hits
                             else None))

        # Check version via check_command
        version = None
        if installed and manifest.setup.check_command:
            cmd = self._substitute_binary(manifest.setup.check_command, manifest.runtime.command, binary)
            check_env = (self._override_env(override)
                         if manifest.slug in self._override_hits else None)
            version = self._version_for(manifest.slug, binary, cmd, check_env)

        # Spec 116 D1/D4: the manifest's version floor, checked offline
        # against the version just read. No version → no floor (the agent
        # is simply not installed, or its CLI could not answer).
        min_version = manifest.setup.min_version
        update_needed = self._below_floor(installed, version, min_version)

        # Dependencies
        deps_met, missing_deps = self.check_dependencies(manifest)

        # Credentials. A CLI that is not installed is not probed: the bare
        # command can only fail, and it logged an "auth probe failed" WARNING
        # per absent agent on every sweep (1018 bench item 9).
        missing_creds, credential_state = self._evaluate_credentials(
            manifest, binary, probe_cli_auth=installed)
        creds_ok = not missing_creds

        # A dispatch already failed with an authentication error. A probe
        # that now says signed-out supersedes the record (that state speaks
        # for itself, and a later sign-in must read as signed in); one that
        # still says signed in cannot see the expiry, so it does not clear it.
        if manifest.slug in self._auth_failures and installed and not creds_ok:
            # Discarded, not ``clear_auth_failure``: this status is the one
            # being built, so there is no cached entry to patch.
            self._auth_failures.discard(manifest.slug)
        auth_expired = installed and creds_ok and manifest.slug in self._auth_failures

        # ACP agents: does the binary actually speak the protocol?
        if handshake_wait is None:
            handshake_wait = getattr(self._sweep_local, "handshake_wait", None)
        handshake = (self._handshake_for(manifest, binary, wait=handshake_wait)
                     if installed else None)

        # Build setup actions
        actions = self._build_actions(manifest, installed, missing_deps, missing_creds,
                                      update_needed=update_needed)

        return AgentSetupStatus(
            slug=manifest.slug,
            name=manifest.name,
            installed=installed,
            binary_path=binary,
            version=version,
            dependencies_met=deps_met,
            missing_dependencies=missing_deps,
            credentials_configured=creds_ok,
            missing_credentials=missing_creds,
            setup_actions=actions,
            credential_state=credential_state,
            update_needed=update_needed,
            min_version=min_version,
            auth_expired=auth_expired,
            override_path=override_path,
            override_missing=override_missing,
            source=self.resolved_source(manifest.slug) if installed else None,
            handshake=handshake,
        )

    def _version_for(self, slug: str, binary: str, cmd: str,
                     check_env: dict[str, str] | None) -> str | None:
        """The version check's output, cached by binary identity (B.3).

        Keyed on the realpath's stat (npm, Homebrew and the claude installer
        replace the file or repoint its symlink on update), the desktop app's
        own version (its launcher script never changes), the command, and the
        locate card's PATH. A failed check is not cached: it is asked again
        next sweep.
        """
        key = self._version_key(slug, binary, cmd, check_env)
        if key is not None:
            cached = self._version_cache.get(key)
            if cached is not None:
                return cached
        # Positional-only when there is nothing to add: test seams stub
        # this method with a one-argument callable.
        version = (self._run_check_command(cmd, env=check_env)
                   if check_env is not None else self._run_check_command(cmd))
        if version is not None:
            version = self._version_line(version)
        if key is not None and version is not None:
            self._version_cache[key] = version
        return version

    def _version_key(self, slug: str, binary: str, cmd: str,
                     check_env: dict[str, str] | None) -> tuple | None:
        try:
            real = os.path.realpath(binary)
            st = os.stat(real)
        except OSError:
            return None
        source = self._resolved_sources.get(slug) or {}
        return (slug, real, st.st_mtime_ns, st.st_size, source.get("app_version"),
                cmd, (check_env or {}).get("PATH"))

    def forget_versions(self) -> None:
        """Drop every cached version answer — the user's Refresh."""
        self._version_cache.clear()
        self._start_checks.clear()

    def _starts(self, manifest: AgentManifest, binary: str, *,
                check_env: dict[str, str] | None = None,
                extra_dir: str | None = None) -> bool:
        """Spec 144: does this candidate launch? Its manifest's version check
        (``<binary> --version`` without one), no stdin, 5 s. Only the copy
        about to be chosen is asked, once per file identity: the answer is
        cached, and a success seeds the version cache so ``check_agent`` does
        not launch it a second time. Sign-in and models are not asked.

        A timeout counts as a start: the process is there, merely slow (a
        cold Windows start, an antivirus scan of a freshly updated exe), and
        skipping it would swap the new build for an old one.

        ``check_env`` is the locate card's env (the version check's own);
        ``extra_dir`` a directory the caller appends to the daemon PATH on
        acceptance (an npm shim needs its ``node`` beside it).
        """
        command = manifest.runtime.command
        check = manifest.setup.check_command or ""
        if check and command and (check == command or check.startswith(command + " ")):
            cmd = self._substitute_binary(check, command, binary)
        else:
            cmd = (f'"{binary}"' if " " in binary else binary) + " --version"
        version_key = self._version_key(manifest.slug, binary, cmd, check_env)
        if version_key is None:
            return False
        key = version_key + (extra_dir,)
        cached = self._start_checks.get(key)
        if cached is not None:
            return cached
        run_env = check_env
        if extra_dir and run_env is None:
            run_env = dict(os.environ)
            run_env["PATH"] = os.pathsep.join(
                p for p in (run_env.get("PATH", ""), extra_dir) if p)
        output = None
        try:
            result = subprocess.run(
                cmd, shell=True, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=5,
                env=run_env, creationflags=win_no_window_flags(),
            )
            ok = result.returncode == 0
            output = (result.stdout or "").strip() if ok else None
            if not ok:
                logger.info("%s: %s does not start (exit %s): %s", manifest.slug,
                            binary, result.returncode,
                            ((result.stderr or result.stdout or "").strip()
                             .splitlines() or [""])[-1][:200])
        except subprocess.TimeoutExpired:
            ok = True
        except OSError as exc:
            ok = False
            logger.info("%s: %s does not start: %s", manifest.slug, binary, exc)
        self._start_checks[key] = ok
        if output and check:
            self._version_cache[version_key] = self._version_line(output)
        return ok

    def _accept(self, manifest: AgentManifest, path: str, source: dict, *,
                check_env: dict[str, str] | None = None,
                extra_dir: str | None = None) -> bool:
        """Pin ``path`` if it starts (spec 144); otherwise leave no pin."""
        self._pin(manifest.slug, path, source)  # the version key reads the source
        if self._starts(manifest, path, check_env=check_env, extra_dir=extra_dir):
            return True
        self._resolved_paths.pop(manifest.slug, None)
        self._resolved_sources.pop(manifest.slug, None)
        return False

    # ------------------------------------------------------------------
    # ACP handshake (readiness verdict)
    # ------------------------------------------------------------------

    def _handshake_for(self, manifest: AgentManifest, binary: str, *,
                       wait: float | None = None) -> dict | None:
        """The ``initialize`` verdict for an ACP agent, cached per binary
        identity. None for agents on other transports."""
        if getattr(manifest.runtime, "transport", "") != "acp-sdk":
            return None
        import concurrent.futures

        future = self._handshake_future(manifest, binary)
        try:
            return dict(future.result(timeout=wait))
        except concurrent.futures.TimeoutError:
            # Still running: unknown never blocks, and a "failed" verdict
            # expires the sweep cache when it lands (_run_handshake_inner).
            return {"state": "unknown", "detail": None, "pending": True,
                    "agent_name": None, "agent_version": None}
        except Exception:  # noqa: BLE001 — a crashed probe is "unknown", never "failed"
            logger.warning("handshake probe crashed for %s", manifest.slug, exc_info=True)
            return {"state": "unknown", "detail": None,
                    "agent_name": None, "agent_version": None}

    def _start_handshakes(self) -> None:
        """Start the handshake of every installed ACP agent whose verdict is
        not cached, so the sweep only waits for the slowest of them."""
        for manifest in self._registry.list_all():
            runtime = getattr(manifest, "runtime", None)
            if getattr(runtime, "transport", "") != "acp-sdk":
                continue
            try:
                binary = self.resolve_binary(manifest)
            except Exception:  # noqa: BLE001 — the per-agent check reports it
                continue
            if binary is not None:
                self._handshake_future(manifest, binary)

    def _handshake_future(self, manifest: AgentManifest, binary: str):
        """The (single-flight) handshake for this binary identity, already
        resolved when the verdict is cached."""
        import concurrent.futures

        identity = self._binary_identity(manifest.slug, binary)
        with self._handshake_lock:
            cached = self._handshakes.get(identity)
            if cached is not None:
                done: concurrent.futures.Future = concurrent.futures.Future()
                done.set_result(dict(cached))
                return done
            future = self._handshake_futures.get(identity)
            if future is None:
                if self._handshake_pool is None:
                    self._handshake_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=6, thread_name_prefix="acp-handshake")
                future = self._handshake_pool.submit(
                    self._run_handshake, manifest, binary, identity)
                self._handshake_futures[identity] = future
            return future

    def _run_handshake(self, manifest: AgentManifest, binary: str,
                       identity: tuple) -> dict:
        from agent_os.agents.handshake import acp_handshake

        try:
            return self._run_handshake_inner(manifest, binary, identity, acp_handshake)
        finally:
            with self._handshake_lock:
                self._handshake_futures.pop(identity, None)

    def _run_handshake_inner(self, manifest: AgentManifest, binary: str,
                             identity: tuple, acp_handshake) -> dict:
        env = None
        try:
            # The spawn's own env (credentials, locate-card PATH), so the
            # verdict is about the process a dispatch would start.
            env = self.get_adapter_config(manifest.slug, "").get("env")
        except ValueError:
            pass
        result = acp_handshake([binary] + list(manifest.runtime.args or []),
                               env=env, cwd=self._data_dir if os.path.isdir(
                                   self._data_dir) else None)
        if result["state"] == "failed":
            logger.warning("ACP handshake failed for %s (%s): %s",
                           manifest.slug, binary, result.get("detail"))
        if result["state"] != "unknown":
            # A timeout is retried on the next sweep; a verdict is kept
            # until the binary changes.
            with self._handshake_lock:
                self._handshakes[identity] = dict(result)
        if result["state"] == "failed":
            # A sweep may already have reported this agent "unknown" while
            # the handshake ran; let the next caller see the verdict.
            cached = self._check_all_cache
            if cached is not None:
                self._check_all_cache = (cached[0], 0.0)
        return result

    def _binary_identity(self, slug: str, binary: str) -> tuple:
        try:
            st = os.stat(binary)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = (None, None)
        source = self._resolved_sources.get(slug) or {}
        manifest = self._registry.get(slug)
        args = tuple(getattr(getattr(manifest, "runtime", None), "args", None) or ())
        # A desktop launcher script never changes when the app updates in
        # place; the app's own version does. The slug and launch args are
        # part of it: one binary driven two ways is two verdicts.
        return (binary, *stamp, source.get("app_version"), slug, args)

    def forget_handshakes(self) -> None:
        """Drop every cached handshake verdict so the next sweep asks again —
        the user's Refresh, which promises current state."""
        with self._handshake_lock:
            self._handshakes.clear()

    @classmethod
    def _below_floor(cls, installed: bool, version: str | None,
                     min_version: str | None) -> bool:
        """Spec 116 D1: installed, version readable, floor declared, and the
        parsed version is older than the floor. Any missing piece → False."""
        if not installed or not version or not min_version:
            return False
        detected = cls._extract_version(version)
        if not detected:
            return False
        return not cls._version_gte(detected, min_version)

    def cached_status(self, slug: str) -> AgentSetupStatus | None:
        """The slug's entry in the last ``check_all`` sweep, stale or not —
        never probes. For callers who want what the UI last showed (the
        dispatch gates, the login job's diagnostics) without paying for a
        subprocess on the event loop."""
        cached = self._check_all_cache
        if cached is None:
            return None
        for status in cached[0]:
            if status.slug == slug:
                return status
        return None

    def check_all_cached(self) -> list[AgentSetupStatus] | None:
        """The last ``check_all`` sweep's statuses, stale or not — never
        probes, never waits. A missing or expired result starts one
        background sweep; None until a sweep has finished. For routes that
        must answer at once (``GET /settings`` is also the desktop shell's
        boot health check)."""
        cached = self._check_all_cache
        if cached is None or time.monotonic() >= cached[1]:
            self._refresh_in_background()
        return None if cached is None else cached[0]

    def last_auth_probe(self, slug: str) -> dict | None:
        """Structured outcome of the last auth probe for ``slug`` (spec 113):
        ``{exit_code, error_class, stderr_category, duration_ms}``, or None
        when no probe has run yet."""
        probe = self._last_auth_probe.get(slug)
        return dict(probe) if probe is not None else None

    def check_all(self, allow_stale: bool = False, *,
                  handshake_wait: float = 0.0,
                  refresh_if_stale: bool = True) -> list[AgentSetupStatus]:
        """Status check for all registered agents.

        Cached for ``CHECK_ALL_CACHE_TTL_SECONDS``. Use ``invalidate_cache()``
        to force a re-check (e.g. after an install or login action).

        ``allow_stale=True`` is for callers on a latency-critical path (turn
        start, spec 101): an expired result is returned at once and a single
        background sweep refreshes it. Only a cache that was never filled, or
        was invalidated, still probes before returning.

        ``refresh_if_stale=False`` (with ``allow_stale``) is a pure read for
        the pin menu: opening a menu never starts a scan. Sweeps come from
        Settings and turn starts, and announce changes over the WS.
        """
        cached = self._check_all_cache
        if cached is not None:
            results, expires_at = cached
            if time.monotonic() < expires_at:
                return results
            if allow_stale:
                if refresh_if_stale:
                    self._refresh_in_background()
                return results
        return self._probe_all(handshake_wait=handshake_wait)

    def _probe_all(self, *, handshake_wait: float = 0.0) -> list[AgentSetupStatus]:
        """Run the sweep once, however many callers ask at the same time."""
        with self._probe_lock:
            # Another caller may have finished a sweep while we waited.
            cached = self._check_all_cache
            if cached is not None and time.monotonic() < cached[1]:
                return cached[0]
            generation = self._cache_generation
            started = time.monotonic()
            self._sweep_local.in_sweep = True
            try:
                self._start_handshakes()
            finally:
                self._sweep_local.in_sweep = False
            slugs = [m.slug for m in self._registry.list_all()]
            memo = _SweepMemo()
            timings: dict[str, float] = {}

            def _one(slug: str) -> AgentSetupStatus:
                # The sweep's handshake budget and memo reach check_agent
                # through a thread-local, so the per-agent call keeps its
                # plain signature (test seams stub it with one argument).
                local = self._sweep_local
                local.handshake_wait = handshake_wait
                local.in_sweep = True
                local.dep_memo = memo
                t0 = time.monotonic()
                try:
                    return self.check_agent(slug)
                finally:
                    timings[slug] = time.monotonic() - t0
                    local.handshake_wait = None
                    local.in_sweep = False
                    local.dep_memo = None

            pool = self._sweep_executor()
            # Results in registry order, whatever order the agents finish in.
            results = [f.result() for f in [pool.submit(_one, s) for s in slugs]]
            if timings:
                slowest = max(timings, key=timings.get)
                logger.info("check_all sweep: %d agents in %.2fs (slowest: %s %.2fs)",
                            len(results), time.monotonic() - started,
                            slowest, timings[slowest])
            if generation == self._cache_generation:
                self._check_all_cache = (
                    results,
                    time.monotonic() + CHECK_ALL_CACHE_TTL_SECONDS,
                )
                self._publish(results)
            return results

    def _sweep_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._sweep_pool is None:
            self._sweep_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=SWEEP_MAX_WORKERS, thread_name_prefix="check-all")
        return self._sweep_pool

    def refresh_agent(self, slug: str) -> AgentSetupStatus:
        """Re-probe one agent and patch it into the cached sweep (spec 136 A.5).

        For the single-agent changes — a login, logout, saved credential,
        located binary, finished install — so the next read costs this
        agent's probe instead of every agent's. The expiry is kept. Under the
        probe lock, so a sweep in flight finishes first (its answer for this
        agent may predate the change); not written when the cache was dropped
        meanwhile. Waits for an ACP handshake verdict, like any single-agent
        check.
        """
        with self._probe_lock:
            generation = self._cache_generation
            status = self.check_agent(slug)
            cached = self._check_all_cache
            if cached is not None and generation == self._cache_generation:
                results = [status if getattr(r, "slug", None) == slug else r
                           for r in cached[0]]
                self._check_all_cache = (results, cached[1])
                self._publish(results)
            return status

    def set_status_listener(self, listener) -> None:
        """``listener(results)`` is called, from the sweeping thread, when a
        cached result differs from the previous one. The first sweep is the
        baseline: nothing was shown before it."""
        self._status_listener = listener

    def _publish(self, results) -> None:
        try:
            digest = _status_digest(results)
        except Exception:  # noqa: BLE001 — never fail a sweep over its digest
            logger.warning("sweep digest failed", exc_info=True)
            return
        previous, self._published_digest = self._published_digest, digest
        if previous is None or previous == digest or self._status_listener is None:
            return
        try:
            self._status_listener(results)
        except Exception:  # noqa: BLE001 — a dead WS must not fail the sweep
            logger.warning("sub-agent status listener failed", exc_info=True)

    def warm(self) -> None:
        """Start the first sweep in the background (daemon start), so the
        first Settings open finds a result instead of running the sweep."""
        self._refresh_in_background()

    def _refresh_in_background(self) -> None:
        """Start one daemon-thread sweep unless one is already running."""
        if self._probe_lock.locked():
            return
        thread = self._refresh_thread
        if thread is not None and thread.is_alive():
            return

        def _run() -> None:
            try:
                self._probe_all()
            except Exception:
                logger.warning("background check_all refresh failed", exc_info=True)

        thread = threading.Thread(target=_run, name="check-all-refresh", daemon=True)
        self._refresh_thread = thread
        thread.start()

    def invalidate_cache(self) -> None:
        """Drop the cached ``check_all()`` result.

        Call this after any orbital-driven action that could change
        agent install/auth status (install, login, logout, etc.) so that
        the next ``check_all()`` re-probes the system. This also drops the
        copy ``allow_stale`` would serve, and keeps a sweep already in flight
        from writing its (possibly pre-install) result back.
        """
        self._cache_generation += 1
        self._check_all_cache = None

    def invalidate_resolved_path(self, slug: str) -> None:
        """Forget the cached binary path for one agent.

        ``resolve_binary`` caches for the daemon's whole lifetime and nothing
        else ever drops an entry, so an agent that gains (or loses) a binary
        mid-run keeps reporting the stale answer — the same shape as the
        claude-code stale-binary-path issue. Call this after any action that
        changes what is on disk for ``slug``.
        """
        self._resolved_paths.pop(slug, None)
        self._resolved_sources.pop(slug, None)
        self._resolved_misses.pop(slug, None)
        self._override_hits.discard(slug)
        self._override_missing.pop(slug, None)

    # ------------------------------------------------------------------
    # The locate card: user-chosen location
    # ------------------------------------------------------------------

    def _override_for(self, slug: str) -> dict | None:
        store = self._sub_agent_config_store
        get = getattr(store, "get_override", None) if store is not None else None
        if not callable(get):
            return None
        try:
            return get(slug)
        except Exception:  # noqa: BLE001 — a broken store must not hide the agent
            logger.warning("Failed to read the location override for %s", slug, exc_info=True)
            return None

    @staticmethod
    def _prepend_path(path_prepend: list[str], base: str | None = None) -> str:
        """``path_prepend`` in front of ``base`` (the daemon PATH by default),
        each directory once."""
        existing = [p for p in (base if base is not None
                                else os.environ.get("PATH", "")).split(os.pathsep) if p]
        front = [p for p in path_prepend if p]
        return os.pathsep.join(front + [p for p in existing if p not in front])

    def _override_env(self, override: dict | None) -> dict[str, str] | None:
        """The daemon env with the override's PATH entries in front, for the
        version check and the Test button. None when there is nothing to add."""
        if not override or not override.get("path_prepend"):
            return None
        env = dict(os.environ)
        env["PATH"] = self._prepend_path(list(override["path_prepend"]))
        return env

    def test_override(self, slug: str, binary: str,
                      path_prepend: list[str] | None = None) -> dict:
        """Run the manifest's ``check_command`` against ``binary`` with
        ``path_prepend`` in front of PATH — the locate card's Test. No shell:
        the binary is argv[0] (a path with spaces survives), the rest of the
        check command is split like a shell would.

        Returns ``{"ok", "version_output"}`` or ``{"ok": False, "error"}``
        with the CLI's own words."""
        manifest = self._registry.get(slug)
        if manifest is None:
            return {"ok": False, "error": f"Unknown agent: {slug}"}
        if not os.path.isfile(binary):
            return {"ok": False, "error": f"No file at {binary}."}
        result = self._test_override_version(manifest, binary, path_prepend)
        if not result.get("ok") or getattr(manifest.runtime, "transport", "") != "acp-sdk":
            return result
        # An ACP agent must also answer the handshake: a binary that prints a
        # version but takes another launch contract (dsh 0.1 vs 0.2) would
        # otherwise be saved and fail every dispatch.
        from agent_os.agents.handshake import acp_handshake
        env = {"PATH": self._prepend_path(list(path_prepend or []))}
        handshake = acp_handshake([binary] + list(manifest.runtime.args or []), env=env)
        if handshake["state"] == "failed":
            return {"ok": False, "error": (
                f"{os.path.basename(binary)} started but did not answer Orbital's "
                f"handshake: {handshake.get('detail') or 'no details'}")}
        return result

    def _test_override_version(self, manifest: AgentManifest, binary: str,
                               path_prepend: list[str] | None) -> dict:
        """The version half of the locate card's Test."""
        check = manifest.setup.check_command or ""
        if not check:
            # Spec 127 D2: an agent with no check command (dsh is an ACP
            # stdio server that takes only ``--config``) cannot be asked
            # for a version — running it bare would block on stdin until
            # the timeout. Existence plus executability is the whole test;
            # the card says so with an empty ``version_output``.
            if not os.access(binary, os.X_OK):
                return {"ok": False, "error": f"{binary} is not executable."}
            return {"ok": True, "version_output": ""}
        command = manifest.runtime.command or ""
        rest = ""
        if check and command and (check == command or check.startswith(command + " ")):
            rest = check[len(command):]
        elif check:
            # A check command that does not start with the agent's own
            # command (unusual) — run the binary with no arguments.
            rest = ""
        try:
            argv = [binary] + shlex.split(rest)
        except ValueError as exc:
            return {"ok": False, "error": f"Bad check command: {exc}"}
        env = dict(os.environ)
        env["PATH"] = self._prepend_path(list(path_prepend or []))
        try:
            result = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15, env=env,
                creationflags=win_no_window_flags(),
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"{os.path.basename(binary)} did not answer within 15 s."}
        except PermissionError:
            return {"ok": False, "error": f"{binary} is not executable."}
        except (FileNotFoundError, OSError) as exc:
            return {"ok": False, "error": f"Could not run {binary}: {exc}"}
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0:
            tail = (err or out).splitlines()[-3:]
            return {"ok": False, "error": " ".join(tail) or
                    f"{os.path.basename(binary)} exited with code {result.returncode}."}
        if not out and not err:
            return {"ok": False, "error": f"{os.path.basename(binary)} printed nothing."}
        return {"ok": True, "version_output": out or err}

    def resolve_binary(self, manifest: AgentManifest) -> str | None:
        """Find the agent binary.

        Strategy:
        0. The user-chosen location (the locate card). Its file gone → the
           rest of the strategies, because vendor updates move their CLI (the
           Codex desktop app keeps it under ``bin/<hash>/``); only when they
           find nothing either is it reported as ``override_missing``.
        1. A vendor desktop app declared ``order: before_cli`` (the vendor's
           primary distribution — DSH Desktop).
        2. shutil.which(command) — works if PATH is correct
        3. Expand and probe auto_detect paths for current OS
        4. A desktop app declared ``order: after_cli`` (a standalone CLI the
           user installed wins over the copy inside ChatGPT.app).
        5. Try check_command via subprocess

        A candidate counts only if it also starts (spec 144, ``_starts``):
        one that is there but fails its version check is passed over for the
        next, in this same order.

        Returns absolute path or None. The answer is pinned until its file
        disappears — an app update that moves its CLI, a self-updating CLI
        that relocates (the claude-code stale-path bug) — and then resolved
        afresh instead of being reported "installed" until a restart.
        """
        slug = manifest.slug
        cached = self._resolved_paths.get(slug)
        if cached is not None:
            # Only a pin this resolver made (it always records a source) is
            # re-checked; a path seeded into the cache directly is the
            # caller's assertion and is taken as given.
            if slug not in self._resolved_sources or os.path.isfile(cached):
                return cached
            logger.info("resolved binary for %s is gone (%s) — resolving again",
                        slug, cached)
            self.invalidate_resolved_path(slug)

        # Spec 136 B.4: inside a sweep, a recent "not found" stands.
        miss = self._resolved_misses.get(slug)
        if (miss is not None and time.monotonic() < miss
                and getattr(self._sweep_local, "in_sweep", False)):
            return None
        found = self._resolve_uncached(manifest)
        if found is None:
            self._resolved_misses[slug] = time.monotonic() + CHECK_ALL_CACHE_TTL_SECONDS
        else:
            self._resolved_misses.pop(slug, None)
        return found

    def _resolve_uncached(self, manifest: AgentManifest) -> str | None:
        """``resolve_binary``'s strategies 0-5, without the caches."""
        slug = manifest.slug

        # Strategy 0: the user chose the location.
        override = self._override_for(slug)
        if override is not None:
            chosen = override["binary"]
            exists = os.path.isfile(chosen)
            if exists and self._accept(manifest, chosen, {"kind": "override"},
                                       check_env=self._override_env(override)):
                self._override_missing.pop(slug, None)
                self._override_hits.add(slug)
                # D3 rule, same reason as strategy 3: the dependency checks
                # (``node --version``) run on the daemon PATH.
                for directory in override.get("path_prepend") or []:
                    _append_to_daemon_path(directory)
                return chosen
            self._override_hits.discard(slug)
            found = self._detect(manifest)
            # "Gone" only when the file is: one that is there but does not
            # start reads as plain not-found (spec 144).
            if found is None and not exists:
                self._override_missing[slug] = chosen
            else:
                self._override_missing.pop(slug, None)
            return found

        return self._detect(manifest)

    def _detect(self, manifest: AgentManifest) -> str | None:
        """``resolve_binary``'s strategies 1-5: finding the agent unaided."""
        slug = manifest.slug
        current_os = detect_os()

        # Strategy 1: the vendor's desktop app, when it is the primary route.
        hit = self._desktop_app_binary(manifest, current_os, "before_cli")
        if hit is not None:
            return hit

        command = manifest.runtime.command
        if not command:
            return None

        # Strategy 2: shutil.which
        path = shutil.which(command)
        if path and self._accept(manifest, path, {"kind": "path"}):
            return path

        # Strategy 3: auto_detect paths for current OS
        auto_paths = manifest.setup.auto_detect.get(current_os, [])
        for raw_path in auto_paths:
            # ${ORBITAL_DATA_DIR} is resolved before expandvars: it is not an
            # environment variable, and dev daemons never export one.
            substituted = raw_path.replace(
                ORBITAL_DATA_DIR_TOKEN, self._data_dir)
            expanded = os.path.expandvars(os.path.expanduser(substituted))
            if ORBITAL_DATA_DIR_TOKEN in raw_path:
                # Token entries mix separators on Windows (the manifest half
                # is written with forward slashes, the data dir is OS-native).
                # isfile() tolerates that; the spawn command should not.
                # Scoped to token entries only: plain auto_detect paths are
                # returned verbatim, as they always were.
                expanded = os.path.normpath(expanded)
            for candidate in _expand_auto_detect_entry(expanded):
                if not os.path.isfile(candidate):
                    continue
                if not self._accept(manifest, candidate, {"kind": "auto_detect"},
                                    extra_dir=os.path.dirname(candidate)):
                    continue
                # Spec 115 D3 rule: a binary found here is one shutil.which
                # could not see, so its directory is not on PATH. An npm
                # entry point is a `#!/usr/bin/env node` shim and the
                # manifest's Node.js dependency is checked with
                # `node --version` on the same PATH — both would fail, and
                # the agent would read "installed" and die at dispatch.
                _append_to_daemon_path(os.path.dirname(candidate))
                return candidate

        # Strategy 4: the CLI inside a desktop app, behind a standalone CLI.
        hit = self._desktop_app_binary(manifest, current_os, "after_cli")
        if hit is not None:
            return hit

        # Strategy 5: try check_command via subprocess
        if manifest.setup.check_command:
            result = self._run_check_command(manifest.setup.check_command)
            if result is not None:
                # check_command succeeded, so the command is available
                # Use shutil.which one more time or fall back to bare command
                path = shutil.which(command)
                if path and self._accept(manifest, path, {"kind": "check_command"}):
                    return path

        return None

    def _pin(self, slug: str, path: str, source: dict) -> None:
        self._resolved_paths[slug] = path
        self._resolved_sources[slug] = source

    def _desktop_app_binary(self, manifest: AgentManifest, current_os: str,
                            order: str) -> str | None:
        """The CLI inside the first installed desktop app of this ``order``.

        Not appended to the daemon PATH: these launchers are self-contained
        (DSH's runs the app's own Electron as node; ChatGPT's execs a signed
        binary beside it), and putting an app's private bin dir on PATH
        would shadow the user's own tools.
        """
        from agent_os.agents.desktop_apps import find_desktop_app

        for app in manifest.setup.desktop_apps:
            if app.order != order:
                continue
            hit = find_desktop_app(app, current_os)
            if hit is None:
                continue
            if not self._accept(manifest, hit.executable, {
                "kind": "desktop_app", "label": hit.label,
                "app_path": hit.app_path, "app_version": hit.version,
            }):
                continue
            return hit.executable
        return None

    def resolved_source(self, slug: str) -> dict | None:
        """Where the pinned binary for ``slug`` came from, or None."""
        source = self._resolved_sources.get(slug)
        return dict(source) if source else None

    def check_dependencies(self, manifest: AgentManifest) -> tuple[bool, list[str]]:
        """Run each dependency check_command via subprocess.

        Returns (all_met, missing_names).
        """
        missing: list[str] = []
        # Spec 136 B.2: within one sweep, each distinct check runs once.
        memo = getattr(self._sweep_local, "dep_memo", None)
        for dep in manifest.setup.dependencies:
            if memo is not None:
                ok = memo.get(("dependency", dep.check_command, dep.min_version),
                              lambda dep=dep: self._check_single_dependency(
                                  dep.check_command, dep.min_version))
            else:
                ok = self._check_single_dependency(dep.check_command, dep.min_version)
            if not ok:
                missing.append(dep.name)
        return (len(missing) == 0, missing)

    def check_credentials(self, manifest: AgentManifest, resolved_binary: str | None = None) -> tuple[bool, list[str]]:
        """Check required credentials in credential_store or env vars.

        Returns (all_configured, missing_keys).
        """
        missing, _state = self._evaluate_credentials(manifest, resolved_binary)
        return (len(missing) == 0, missing)

    def _evaluate_credentials(self, manifest: AgentManifest,
                              resolved_binary: str | None, *,
                              probe_cli_auth: bool = True) -> tuple[list[str], str | None]:
        """Missing required credential keys, plus the ``model_provider`` state.

        The state is None unless the manifest declares a ``model_provider``
        credential; only its "missing" answer counts as a missing key.

        ``probe_cli_auth=False`` (the agent is not installed) reports every
        probed ``oauth_cli`` credential missing without spawning its probe.
        """
        missing: list[str] = []
        state: str | None = None
        for cred in manifest.setup.credentials:
            if not cred.required:
                continue
            if cred.type == "model_provider":
                state = self._model_provider_state(manifest, cred, resolved_binary)
                if state == "missing":
                    missing.append(cred.key)
                continue
            # oauth_cli: check via CLI command
            if cred.type == "oauth_cli":
                if not cred.check_command:
                    # The CLI owns the login but offers no status probe
                    # (MiniMax Code). Orbital can neither confirm nor deny
                    # the sign-in, so say so instead of reporting a missing
                    # credential that would read as "not logged in"; the
                    # spawn surfaces the CLI's own auth error if it is.
                    state = state or "unknown"
                    continue
                if not probe_cli_auth:
                    missing.append(cred.key)
                    continue
                verdict = self._cli_auth_verdict(cred, manifest.runtime.command,
                                                 resolved_binary, slug=manifest.slug)
                remembered = (manifest.slug, cred.key)
                if verdict is None:
                    # Spec 136 decision 2: the probe timed out. Keep the last
                    # answer; with none, the sign-in is unknown — never a
                    # Login button for a CLI that simply did not answer.
                    verdict = self._auth_verdicts.get(remembered)
                    if verdict is None:
                        state = state or "unknown"
                        continue
                else:
                    self._auth_verdicts[remembered] = verdict
                if not verdict:
                    missing.append(cred.key)
                continue
            # The agent's own store wins over anything Orbital injects
            # (DSH Desktop's credentials file) — a key there is configured.
            if cred.agent_store and agent_store_has(cred.agent_store):
                continue
            # Check credential store
            if self._credential_store is not None:
                value = self._credential_store.get(cred.key)
                if value:
                    continue
            # Check environment variable
            env_key = cred.env_var or cred.key
            if os.environ.get(env_key):
                continue
            missing.append(cred.key)
        return missing, state

    def _model_provider_state(self, manifest: AgentManifest, cred,
                              resolved_binary: str | None) -> str:
        """Readiness of a credential that follows the configured model's provider.

        "configured": a credential card holds the key Orbital would inject for
        that provider, or the CLI's own auth probe answers ready. "missing":
        the probe says the provider has no credentials. "unknown": no model is
        configured (the CLI picks its own default), the binary is absent, or
        the probe gave no verdict — dispatchable, but only the CLI can tell.
        Which credential the CLI finally uses is its own resolution order;
        this never claims one.
        """
        model = self._configured_model(manifest.slug)
        if not model or resolved_binary is None:
            return "unknown"
        if self._card_key_for_model(cred, model)[1]:
            return "configured"
        verdict = self._probe_model_auth(
            cred, manifest.runtime.command, resolved_binary, model)
        if verdict is None:
            return "unknown"
        return "configured" if verdict else "missing"

    def _configured_model(self, slug: str) -> str | None:
        from agent_os.daemon_v2.sub_agent_config_store import resolve_params
        return resolve_params(slug, self._sub_agent_config_store).get("model") or None

    def _card_key_for_model(self, cred, model: str) -> tuple[str, str]:
        """(env_var, key) from the user's cards for ``model``'s provider, or ("", "").

        Exactly one entry can match — the one naming the model's provider — so
        at most one card key is ever looked up, let alone injected.
        """
        if self._card_store is None or "/" not in model:
            return "", ""
        provider = model.split("/", 1)[0]
        for entry in cred.provider_keys:
            if entry.get("provider") != provider or not entry.get("env_var"):
                continue
            try:
                key = self._card_store.key_for_provider(
                    entry.get("card_provider") or provider, entry.get("card_region"))
            except Exception:
                logger.warning("credential-card lookup failed for provider %s",
                               provider, exc_info=True)
                return "", ""
            return (entry["env_var"], key) if key else ("", "")
        return "", ""

    def _probe_model_auth(self, cred, command: str | None, resolved_binary: str,
                          model: str) -> bool | None:
        """Run the CLI's auth probe for ``model``: True ready, False missing, None unknown.

        Built as an argv list — the model is user text and never meets a shell.
        """
        import json as _json
        import shlex
        try:
            argv = shlex.split(cred.check_command)
        except ValueError:
            return None
        if not argv:
            return None
        if command and argv[0] == command:
            argv[0] = resolved_binary
        argv = [model if token == "{model}" else token for token in argv]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=win_no_window_flags(),
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        lines = (result.stdout or "").strip().splitlines()
        try:
            data = _json.loads(lines[-1]) if lines else None
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        actual = str(_json_field(data, cred.check_field))
        if actual == cred.check_value:
            return True
        if cred.check_missing_value and actual == cred.check_missing_value:
            return False
        return None

    def _check_cli_auth(self, cred, command: str | None = None,
                        resolved_binary: str | None = None, *,
                        slug: str | None = None) -> bool:
        """Whether the CLI says it is signed in; a timeout reads as False."""
        return self._cli_auth_verdict(cred, command, resolved_binary, slug=slug) is True

    def _cli_auth_verdict(self, cred, command: str | None = None,
                          resolved_binary: str | None = None, *,
                          slug: str | None = None) -> bool | None:
        """Run a CLI command to check auth status (e.g. claude auth status --json).

        None when the probe timed out: no answer, not a "signed out" one.

        Parses JSON output and checks check_field == check_value.

        Spec 116 D7: every False used to be silent — a CLI too old to know
        ``auth status`` (exit 1), a probe past the 10 s timeout, or garbage
        output all read as "not logged in" with nothing in the log. Each
        False now writes ONE warning naming the slug, the binary, and the
        exit code or exception class plus the first 200 chars of stderr; the
        structured outcome is kept for the dispatch gate's telemetry.
        """
        import json as _json
        if not cred.check_command:
            return False
        check_cmd = self._substitute_binary(cred.check_command, command, resolved_binary)
        key = slug or cred.key
        started = time.monotonic()

        def _record(ok: bool, *, exit_code: int | None, exc: BaseException | None,
                    stderr: str, why: str) -> bool:
            duration_ms = int((time.monotonic() - started) * 1000)
            self._last_auth_probe[key] = {
                "exit_code": exit_code,
                "error_class": error_class_of(exc),
                "stderr_category": classify_stderr(stderr or why),
                "duration_ms": duration_ms,
            }
            if not ok:
                # ``check_cmd`` is the probe only: the resolved binary plus
                # the manifest's status args (never a login/update command).
                logger.warning(
                    "auth probe failed for %s (%s): `%s` -> %s; stderr: %r",
                    key, cred.key, check_cmd, why, (stderr or "")[:200],
                )
            return ok

        try:
            result = subprocess.run(
                check_cmd,
                shell=True,
                capture_output=True,
                text=True,
                # CLIs emit UTF-8; the Windows ANSI default crashes the
                # decode on non-ASCII output (see test_subprocess_encoding).
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=win_no_window_flags(),
            )
            # Tests hand back MagicMock results: only a real string is text.
            stderr = result.stderr if isinstance(result.stderr, str) else ""
            stdout = result.stdout if isinstance(result.stdout, str) else ""
            if result.returncode != 0:
                return _record(False, exit_code=result.returncode, exc=None,
                               stderr=stderr or stdout,
                               why=f"exit code {result.returncode}")
            if not cred.check_field:
                # Exit-code mode: a CLI whose status command prints plain
                # text (`codex login status`) and signals via its exit code.
                # Shell-neutral, unlike wrapping it in `&& echo '{...}'`,
                # which cmd.exe cannot run.
                return _record(True, exit_code=0, exc=None, stderr="", why="")
            try:
                data = _json.loads(result.stdout)
            except (_json.JSONDecodeError, ValueError) as exc:
                return _record(False, exit_code=0, exc=exc, stderr=stderr or stdout,
                               why=f"{type(exc).__name__} parsing stdout")
            actual = str(_json_field(data, cred.check_field))
            ok = actual.lower() == cred.check_value.lower()
            return _record(ok, exit_code=0, exc=None,
                           stderr=stderr or f"{cred.check_field}={actual}",
                           why=f"{cred.check_field}={actual!r}, expected {cred.check_value!r}")
        except subprocess.TimeoutExpired as exc:
            _record(False, exit_code=None, exc=exc, stderr="",
                    why="TimeoutExpired after 10s")
            return None
        except (FileNotFoundError, OSError, ValueError) as exc:
            return _record(False, exit_code=None, exc=exc, stderr=str(exc),
                           why=f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _substitute_binary(cmd: str, command: str | None, resolved: str | None) -> str:
        """Replace a bare command name with the resolved absolute path.

        E.g. 'claude auth status --json' → '/Users/x/.claude/local/claude auth status --json'
        Only substitutes if the command appears at the start of the string.
        """
        if not resolved or not command:
            return cmd
        if cmd == command or cmd.startswith(command + " "):
            # The command runs through a shell: a launcher inside an app
            # bundle ("/Applications/DeepSeek Harness.app/…", Claude's
            # "Application Support") would split at its first space.
            quoted = f'"{resolved}"' if " " in resolved else resolved
            return quoted + cmd[len(command):]
        return cmd

    def get_resolved_path(self, slug: str) -> str | None:
        """Return cached resolved binary path. Call check_agent first."""
        return self._resolved_paths.get(slug)

    def get_adapter_config(
        self,
        slug: str,
        project_workspace: str,
        credential_overrides: dict | None = None,
    ) -> dict:
        """Build adapter-config-compatible dict from manifest + resolved path.

        This is the bridge between manifest world and existing CLIAdapter world.

        Raises ValueError if agent not found or not installed.
        """
        manifest = self._registry.get(slug)
        if manifest is None:
            raise ValueError(f"Unknown agent: {slug}")

        # Through resolve_binary, never the raw cache: a pinned path whose
        # file has gone must re-resolve, not spawn into ENOENT.
        binary = self.resolve_binary(manifest)
        if binary is None:
            raise ValueError(f"Agent '{slug}' not installed or binary not found")

        # Build env from manifest credentials + overrides
        env: dict[str, str] = {}
        location = self._override_for(slug) if slug in self._override_hits else None
        for cred in manifest.setup.credentials:
            if cred.type == "model_provider":
                # Only the key for the configured model's provider, under the
                # CLI's own env var — never every saved card key, never argv.
                model = self._configured_model(slug)
                env_var, key = self._card_key_for_model(cred, model) if model else ("", "")
                if key:
                    env[env_var] = key
                continue
            value = (credential_overrides or {}).get(cred.key)
            if value is None and self._credential_store is not None:
                value = self._credential_store.get(cred.key)
            if value is None:
                value = os.environ.get(cred.env_var or cred.key, "")
            if value and cred.env_var:
                env[cred.env_var] = value

        # The locate card: the user's PATH entries go in front of the
        # daemon's PATH for this spawn, and the variables the agent asked
        # for ride along — under a credential the manifest injects, never
        # over it.
        if location is not None:
            if location.get("path_prepend"):
                env["PATH"] = self._prepend_path(list(location["path_prepend"]))
            for name, value in (location.get("env") or {}).items():
                if value and name not in env:
                    env[name] = value

        # Params delivered by environment (dsh's DSH_PERMISSION_MODE). Never
        # over a credential the manifest injects.
        if self._sub_agent_config_store is not None:
            build_env = getattr(self._sub_agent_config_store, "build_extra_env", None)
            if callable(build_env):
                try:
                    for name, value in (build_env(slug) or {}).items():
                        env.setdefault(name, value)
                except Exception:
                    logger.warning("Failed to build sub-agent env for %s",
                                   slug, exc_info=True)

        # Merge daemon-level sub-agent config (model/effort/etc.) into argv.
        # Persisted overrides win when no manifest arg conflicts; we simply
        # append, leaving the CLI to take the last-wins value if duplicated.
        extra_args: list[str] = []
        if self._sub_agent_config_store is not None:
            try:
                extra_args = list(
                    self._sub_agent_config_store.build_extra_args(slug)
                )
            except Exception:
                logger.warning(
                    "Failed to build sub-agent extra args for %s",
                    slug, exc_info=True,
                )

        # ACP session options (dsh's reasoning_effort) and what the rendered
        # file would copy from the user's own setup of the agent: neither is
        # argv or env, but both decide what a spawn runs with.
        acp_config_options: dict[str, str] = {}
        if self._sub_agent_config_store is not None:
            build_options = getattr(self._sub_agent_config_store,
                                    "build_acp_config_options", None)
            if callable(build_options):
                try:
                    acp_config_options = dict(build_options(slug) or {})
                except Exception:
                    logger.warning("Failed to build ACP config options for %s",
                                   slug, exc_info=True)
        composition_inputs = None
        mirror_spec = getattr(manifest.runtime, "config_mirror", None)
        if mirror_spec:
            from agent_os.agents.composition import (
                acp_model_value, effective_model, read_mirror,
            )
            from agent_os.daemon_v2.sub_agent_config_store import resolve_params
            composition_inputs = read_mirror(mirror_spec)[2]
            # The patch sets the model a NEW session boots with; a resumed one
            # keeps the model recorded in it (dsh's own rule). Switching it
            # in-session through dsh's ACP ``model`` option makes every spawn
            # run the model Settings shows — applied only if dsh offers it.
            pair = effective_model(manifest, resolve_params(
                slug, self._sub_agent_config_store))
            if pair is not None:
                acp_config_options.setdefault("model", acp_model_value(pair))

        return {
            "command": binary,
            "args": list(manifest.runtime.args) + extra_args,
            "acp_config_options": acp_config_options,
            "composition_inputs": composition_inputs,
            "workspace": project_workspace,
            "approval_patterns": [
                p.get("regex", "") for p in manifest.runtime.approval_patterns
            ],
            "env": env,
            "network_domains": list(manifest.permissions.network_domains),
            "interactive": manifest.runtime.interactive,
        }

    def fingerprint_config(self, slug: str, config_dict: dict) -> str:
        """``spawn_fingerprint`` of a config built by ``get_adapter_config``."""
        params: dict = {}
        if self._sub_agent_config_store is not None:
            try:
                params = dict(self._sub_agent_config_store.get(slug) or {})
            except Exception:
                logger.warning("Failed to read sub-agent params for %s",
                               slug, exc_info=True)
        return spawn_fingerprint(config_dict, params)

    def spawn_fingerprint(self, slug: str, project_workspace: str) -> str | None:
        """What a spawn of ``slug`` would be launched with right now, or None
        when no config can be built (binary gone, unknown agent)."""
        try:
            config_dict = self.get_adapter_config(
                slug=slug, project_workspace=project_workspace)
        except ValueError:
            return None
        return self.fingerprint_config(slug, config_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_check_command(self, command: str,
                           env: dict[str, str] | None = None) -> str | None:
        """Run a check command and return its stripped stdout, or None on failure."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=env,
                creationflags=win_no_window_flags(),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    def _check_single_dependency(
        self, check_command: str, min_version: str | None
    ) -> bool:
        """Run a single dependency check_command.

        Returns True if the command succeeds (and version >= min_version if specified).
        """
        try:
            result = subprocess.run(
                check_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                creationflags=win_no_window_flags(),
            )
            if result.returncode != 0:
                return False
            if min_version and result.stdout.strip():
                detected = self._extract_version(result.stdout.strip())
                if detected and not self._version_gte(detected, min_version):
                    return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    @staticmethod
    def _extract_version(raw: str) -> str | None:
        """Extract a version number from command output like 'v20.11.0'."""
        import re
        match = re.search(r"(\d+\.\d+\.\d+)", raw)
        return match.group(1) if match else None

    @staticmethod
    def _version_line(raw: str) -> str:
        """The line of ``--version`` output that carries the version.

        CLIs print notices around it (Gemini on Windows: "Rename failed with
        EPERM, retrying…"; Node: ExperimentalWarning), and the card shows
        this string as-is.
        """
        import re
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            if re.search(r"\d+\.\d+", ln):
                return ln
        return lines[0] if lines else raw

    @staticmethod
    def _version_gte(detected: str, minimum: str) -> bool:
        """Simple version comparison: is detected >= minimum?"""
        def parts(v: str) -> list[int]:
            try:
                return [int(x) for x in v.split(".")]
            except ValueError:
                return [0]
        return parts(detected) >= parts(minimum)

    def _build_actions(
        self,
        manifest: AgentManifest,
        installed: bool,
        missing_deps: list[str],
        missing_creds: list[str],
        update_needed: bool = False,
    ) -> list[SetupAction]:
        """Build the list of setup actions needed."""
        actions: list[SetupAction] = []
        current_os = detect_os()

        # Spec 116 D4: an installed CLI older than the floor. Listed before
        # the credential actions — a too-old CLI cannot be logged in through
        # Orbital, so the update is the step that unblocks everything else.
        if update_needed:
            actions.append(SetupAction(
                action="update_agent",
                label=manifest.setup.update_label or f"Update {manifest.name}",
                command=manifest.setup.update_command,
                blocking=True,
            ))

        # Missing dependencies
        for dep in manifest.setup.dependencies:
            if dep.name in missing_deps:
                install_cmd = dep.install.get(current_os)
                actions.append(SetupAction(
                    action="install_dependency",
                    label=f"Install {dep.name}" + (
                        f" {dep.min_version}+" if dep.min_version else ""
                    ),
                    command=install_cmd,
                    blocking=True,
                ))

        # Agent not installed
        if not installed and manifest.setup.install_command:
            actions.append(SetupAction(
                action="install_agent",
                label=f"Install {manifest.name}",
                command=manifest.setup.install_command,
                blocking=True,
            ))

        # Missing credentials
        for key in missing_creds:
            # Find the credential from manifest
            cred_obj = None
            label = key
            for cred in manifest.setup.credentials:
                if cred.key == key:
                    cred_obj = cred
                    label = cred.label or key
                    break
            if cred_obj and cred_obj.type == "oauth_cli":
                actions.append(SetupAction(
                    action="run_cli_auth",
                    label=cred_obj.setup_label or f"Authenticate {label}",
                    command=cred_obj.setup_command or None,
                    blocking=True,
                ))
            else:
                actions.append(SetupAction(
                    action="configure_credential",
                    label=f"Set {label}",
                    command=None,
                    blocking=True,
                ))

        return actions
