# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Daemon-level sub-agent configuration store.

Persists per-sub-agent CLI parameter overrides (model, effort/thinking level,
permission mode, etc.) to ``~/.orbital/sub_agent_config.json``. Read at
sub-agent dispatch time; merged into the CLI invocation as appropriate flags.

Validation is intentionally narrow: only the "stable parameters" documented
in the sub-agent findings doc are accepted. Experimental flags are out of
scope; users wanting them can pass them via env or the CLI's own settings
files. See SCHEMA below for the canonical list.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass

from agent_os.daemon_v2.models import detect_os

logger = logging.getLogger(__name__)


# The user-chosen agent location (the locate card). Kept in the same per-slug
# JSON object as the CLI params, under keys no param schema may use, so the
# override follows the user account exactly like the params do. ``binary`` is
# the launcher to run, ``path_prepend`` the directories for the front of
# PATH (a version manager's node, say), ``env`` the variables the agent said
# it needs (names from the agent, values from the user, possibly empty).
OVERRIDE_KEYS = ("binary", "path_prepend", "env")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_WINDOWS_SHIMS = (".cmd", ".bat", ".ps1")


# Per-sub-agent stable parameter schema. Each entry maps a slug to a dict of
# parameter names; each parameter has an ``allowed`` list (None = free-text,
# validated as non-empty string) and a ``flag_template``. The flag_template
# is a Python format string with ``{value}`` placeholder; if it produces a
# string with a space, it is split into multiple argv tokens. ``None`` marks a
# param that is delivered some other way than argv (dsh renders its params into
# a composition file) — validated and surfaced to the UI like any other, but
# contributing nothing to the command line.
@dataclass(frozen=True)
class _ParamSchema:
    name: str
    allowed: tuple[str, ...] | None  # None = free-text
    flag_template: str | None  # e.g. "--model {value}"; None = not argv
    # Effective runtime value when the user has not persisted an override.
    # Most CLI parameters deliberately have no Orbital default and continue
    # to inherit the provider's own default.
    default: str | None = None
    # Delivered as this environment variable instead of argv (dsh reads its
    # sandbox + approval mode from DSH_PERMISSION_MODE).
    env_var: str | None = None
    # Delivered as this ACP session config option (session/set_config_option
    # after session/new or resume) — the agent's own knob, applied only when
    # the agent advertises the option and the value.
    acp_config_option: str | None = None


# Authoritative tier-5 stable parameters for each sub-agent. Anything not on
# this list is rejected by ``validate()`` (returns 400 from the API).
SCHEMA: dict[str, dict[str, _ParamSchema]] = {
    "claude-code": {
        "model": _ParamSchema(
            name="model",
            # FALLBACK only: the settings route replaces this with the CLI's
            # live account list (claude_models, initialize → models[]) and
            # validates saves against it, so a new generation needs no code
            # change. This list is what users see when that probe fails.
            # Aliases (fable/haiku/sonnet/opus) track the CLI's own mapping;
            # the pins are the current generation (was: claude-fable-5 /
            # claude-opus-4-8 / claude-sonnet-5 / claude-haiku-4-5).
            allowed=(
                "fable",
                "haiku",
                "sonnet",
                "opus",
                "claude-opus-5-5",
                "claude-fable-5-1",
                "claude-opus-5",
                "claude-fable-5",
                "claude-opus-4-8",
                "claude-sonnet-5",
                "claude-haiku-4-5",
            ),
            flag_template="--model {value}",
        ),
        "effort": _ParamSchema(
            name="effort",
            allowed=("low", "medium", "high", "xhigh", "max"),
            flag_template="--effort {value}",
        ),
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=(
                "acceptEdits",
                "auto",
                "bypassPermissions",
                "default",
                "dontAsk",
                "plan",
            ),
            flag_template="--permission-mode {value}",
        ),
    },
    "codex": {
        # Codex model IDs evolve too quickly to enumerate; accept any non-empty
        # string. Other tunables go via -c key=value TOML config — too broad
        # for a v1 dropdown, so they are not surfaced here.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="-m {value}",
        ),
        # Reasoning effort (spec 149). Rendered as codex's own config
        # override; the app-server transport lifts it out of argv and sends
        # it as ``turn/start.effort`` (the protocol's per-turn knob).
        "effort": _ParamSchema(
            name="effort",
            allowed=("low", "medium", "high"),
            flag_template="-c model_reasoning_effort={value}",
        ),
    },
    "cursor": {
        # Cursor model IDs are account-dependent and change independently of
        # Orbital releases, so this stays free-text.  The ACP transport moves
        # this global option before the final ``acp`` subcommand.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        # This is an Orbital permission policy consumed by ACPSDKTransport,
        # not a Cursor CLI flag.  ``auto`` preserves unattended queue
        # execution; ``ask`` forwards native ACP permission requests to
        # Orbital's permission card.
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=("auto", "ask"),
            flag_template="--orbital-permission-mode {value}",
            default="auto",
        ),
    },
    "codebuddy": {
        # CodeBuddy's model catalogue is account- and region-dependent (the
        # China build lists hy3/deepseek/glm/kimi/minimax ids, the global build
        # gpt/gemini/glm/kimi ids, plus role aliases like default-model), so
        # this stays free-text. Delivered through the ACP transport, which
        # lifts ``--model`` out of argv and applies it to the session.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        # Orbital's own ACP permission policy (same contract as cursor), not
        # a CodeBuddy flag: ``auto`` answers every permission request itself
        # so queue runs never block; ``ask`` routes them to the approval card.
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=("auto", "ask"),
            flag_template="--orbital-permission-mode {value}",
            default="auto",
        ),
    },
    "kimi-code": {
        # Kimi Code's models are the aliases in its own config.toml
        # (kimi-code/k3, kimi-code/kimi-for-coding, ...), which move with the
        # CLI and the account, so this stays free-text. Delivered through the
        # ACP transport, which lifts ``--model`` out of argv (Kimi takes it
        # before the ``acp`` subcommand) and applies it to the session.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        # Orbital's own ACP permission policy (same contract as cursor), not
        # a Kimi flag: ``auto`` answers every permission request itself so
        # queue runs never block; ``ask`` routes them to the approval card.
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=("auto", "ask"),
            flag_template="--orbital-permission-mode {value}",
            default="auto",
        ),
    },
    "minimax-code": {
        # MiniMax Code accepts only its session's internal model ids
        # (m:minimax:MiniMax-M3:v:thinking); anything else is not an
        # advertised value and the transport keeps the CLI default. The
        # catalogue changes with the account's token plan, so free-text.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        # Orbital's own ACP permission policy (same contract as cursor), not
        # a MiniMax Code flag.
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=("auto", "ask"),
            flag_template="--orbital-permission-mode {value}",
            default="auto",
        ),
    },
    "gemini-cli": {
        # gemini-cli may not be installed today; this entry exists so the UI
        # can render the right dropdowns the moment it shows up in the
        # registry.
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        "approval-mode": _ParamSchema(
            name="approval-mode",
            allowed=("default", "auto_edit", "plan", "yolo"),
            flag_template="--approval-mode {value}",
        ),
    },
    "dsh": {
        # None of these is argv. The model is ``provider/model`` and goes into
        # the per-spawn patch's ``acp`` row (agent_os/agents/composition.py);
        # the dropdown is dsh's own live list (the providers the user set up
        # in DSH Desktop, mirrored into the patch), which widens this static
        # fallback at save time. Unset follows the default model picked in
        # the app. A change rebuilds an idle worker on its next dispatch.
        "model": _ParamSchema(
            name="model",
            allowed=(
                "deepseek-official/deepseek-flash",
                "deepseek-official/deepseek-v4-pro",
            ),
            flag_template=None,
        ),
        # dsh's own reasoning control (ACP config option ``reasoning_effort``,
        # offered for reasoning models). Unset = dsh's default.
        "effort": _ParamSchema(
            name="effort",
            allowed=("off", "low", "high", "max"),
            flag_template=None,
            acp_config_option="reasoning_effort",
        ),
        # dsh's own enforcement layer (its sandbox policy, and its approval
        # policy: "never" ask only under danger-full-access) — Orbital's
        # project sandbox does not govern transport-backed sub-agents.
        "permission-mode": _ParamSchema(
            name="permission-mode",
            allowed=("read-only", "workspace-write", "danger-full-access"),
            flag_template=None,
            default="workspace-write",
            env_var="DSH_PERMISSION_MODE",
        ),
    },
    "pi": {
        # `provider/model`, exactly as Pi's own --model takes it. Free text:
        # Pi's catalogue spans 40+ providers and moves with every release.
        # The provider half also picks which credential-card key (if any)
        # Orbital hands the process (see the pi manifest).
        "model": _ParamSchema(
            name="model",
            allowed=None,
            flag_template="--model {value}",
        ),
        # Pi's reasoning level, surfaced under the shared `effort` key. Levels
        # a model does not support are clamped by Pi itself.
        "effort": _ParamSchema(
            name="effort",
            allowed=("off", "minimal", "low", "medium", "high", "xhigh", "max"),
            flag_template="--thinking {value}",
        ),
    },
}


# Spec 149: the params a single conversation may override over Settings (the
# pin menu's model + thinking). Stored with the session, layered at dispatch.
SESSION_PARAMS = ("model", "effort")


def resolve_params(slug: str, store: "SubAgentConfigStore | None" = None,
                   overrides: dict[str, str] | None = None,
                   ) -> dict[str, str]:
    """Effective params for ``slug``: SCHEMA defaults under persisted overrides,
    under this conversation's ``overrides`` (spec 149) when given.

    The argv path (``build_extra_args``) and the composition renderer both read
    configuration through this, so a param means the same thing either way.
    """
    params = {
        name: schema.default
        for name, schema in SCHEMA.get(slug, {}).items()
        if schema.default is not None
    }
    if store is not None:
        params.update(store.get(slug))
    for key, value in (overrides or {}).items():
        if key in SESSION_PARAMS and key in SCHEMA.get(slug, {}) and value:
            params[key] = str(value)
    return params


def validate_session_params(slug: str, params: dict, *,
                            extra_allowed: dict[str, list[str]] | None = None
                            ) -> dict[str, str]:
    """Cleaned spec-149 conversation overrides or :class:`SubAgentConfigError`.

    Only ``SESSION_PARAMS`` the agent's schema has; values checked the same
    way a Settings save is. Empty / None values are dropped (= "Default")."""
    if not isinstance(params, dict):
        raise SubAgentConfigError("params must be an object")
    for key in params:
        if key not in SESSION_PARAMS:
            raise SubAgentConfigError(
                f"'{key}' cannot be set per conversation")
    return SubAgentConfigStore.validate(slug, params, extra_allowed=extra_allowed)


class SubAgentConfigError(ValueError):
    """Raised when a config payload fails validation."""


class SubAgentConfigStore:
    """Read/write daemon-level sub-agent configuration.

    Storage: a single JSON file. Map of ``slug -> {param: value}``. Empty
    dict for a slug means "no overrides; use the CLI's own defaults".
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        # slug -> {param: str, plus the OVERRIDE_KEYS in their native shapes}
        self._data: dict[str, dict[str, object]] = {}
        self._load()

    @property
    def path(self) -> str:
        return self._path

    def _load(self) -> None:
        if not os.path.exists(self._path):
            self._data = {}
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Coerce param values to str; the override keys keep their
                # native shapes (str / list[str] / dict[str, str]); drop
                # anything malformed.
                cleaned: dict[str, dict[str, object]] = {}
                for slug, params in raw.items():
                    if not isinstance(params, dict):
                        continue
                    entry: dict[str, object] = {}
                    for k, v in params.items():
                        k = str(k)
                        if k == "binary" and isinstance(v, str) and v:
                            entry[k] = v
                        elif k == "path_prepend" and isinstance(v, list):
                            entry[k] = [str(x) for x in v if isinstance(x, str) and x]
                        elif k == "env" and isinstance(v, dict):
                            entry[k] = {str(a): str(b) for a, b in v.items()
                                        if isinstance(a, str) and _ENV_NAME.match(a)}
                        elif k in OVERRIDE_KEYS:
                            continue
                        elif isinstance(v, (str, int, float, bool)):
                            entry[k] = str(v)
                    cleaned[slug] = entry
                self._data = cleaned
            else:
                logger.warning(
                    "sub_agent_config.json was not a dict; ignoring (%s)",
                    self._path,
                )
                self._data = {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load %s: %s", self._path, exc)
            self._data = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._path)

    @staticmethod
    def _params_of(entry: dict[str, object]) -> dict[str, str]:
        return {k: str(v) for k, v in entry.items() if k not in OVERRIDE_KEYS}

    def get(self, slug: str) -> dict[str, str]:
        """Return the persisted CLI params for ``slug``, or empty dict. The
        location override is not a param: see :meth:`get_override`."""
        with self._lock:
            return self._params_of(self._data.get(slug, {}))

    def all(self) -> dict[str, dict[str, str]]:
        """Return a snapshot of every persisted slug's params."""
        with self._lock:
            return {k: self._params_of(v) for k, v in self._data.items()}

    # ------------------------------------------------------------------
    # Location override (the locate card)
    # ------------------------------------------------------------------

    def get_override(self, slug: str) -> dict | None:
        """``{"binary", "path_prepend", "env"}`` for ``slug``, or None when
        the user has not chosen a location."""
        with self._lock:
            entry = self._data.get(slug, {})
            binary = entry.get("binary")
            if not isinstance(binary, str) or not binary:
                return None
            return {
                "binary": binary,
                "path_prepend": list(entry.get("path_prepend") or []),
                "env": dict(entry.get("env") or {}),
            }

    @staticmethod
    def manifest_launches_via_shim(manifest) -> bool:
        """True when every Windows launcher the manifest knows is a
        ``.cmd``/``.bat``/``.ps1`` shim — dsh's ``dsh.cmd`` (npm's, or the
        one inside DSH Desktop) is the only spawn target it has. For such an
        agent a pasted shim is the same file Orbital itself runs, and "paste
        the real executable instead" names a file that does not exist, so
        the shim refusal below would reject the only path that works (spec
        127 V4 on Windows, 2026-10-22). An agent that also ships a real
        executable (claude-code's ``claude.exe``) keeps the refusal: there
        the spec 116 failure mode — a stopped shim leaving its node child
        running — has a better answer."""
        try:
            entries = [e for e in (manifest.setup.auto_detect.get("windows", []) or [])
                       if isinstance(e, str) and e.strip()]
        except AttributeError:
            return False
        # The launcher inside a desktop app (DSH Desktop's dsh.cmd) counts:
        # it is the file discovery itself would hand the spawn.
        for app in getattr(manifest.setup, "desktop_apps", None) or []:
            spec = getattr(app, "platforms", {}).get("windows")
            if spec is not None and getattr(spec, "executable", ""):
                entries.append(spec.executable)
        return bool(entries) and all(
            e.lower().endswith(_WINDOWS_SHIMS) for e in entries
        )

    @staticmethod
    def validate_override(binary: str, path_prepend: list[str] | None = None,
                          env: dict[str, str] | None = None, *,
                          allow_shim: bool = False) -> dict:
        """Cleaned override or :class:`SubAgentConfigError`.

        The binary must be an absolute path to an existing file. On Windows
        an npm ``.cmd``/``.bat``/``.ps1`` shim is refused with the real
        executable named: ``cmd.exe`` re-parses the arguments and a stopped
        shim leaves its ``node`` child running (spec 116 hit this for claude).
        ``allow_shim`` lifts that for an agent whose manifest launches via a
        shim anyway (:meth:`manifest_launches_via_shim`). PATH entries must
        be absolute; env keys must be variable names.
        """
        binary = (binary or "").strip()
        if not binary:
            raise SubAgentConfigError("No executable path given.")
        if not os.path.isabs(binary):
            raise SubAgentConfigError(
                f"'{binary}' is not an absolute path to the executable.")
        if (not allow_shim and detect_os() == "windows"
                and binary.lower().endswith(_WINDOWS_SHIMS)):
            base = os.path.basename(binary)
            stem = os.path.splitext(base)[0]
            raise SubAgentConfigError(
                f"{base} is a Windows shim script; Orbital cannot run it. Paste "
                f"the real executable instead (for example {stem}.exe, or the "
                f"file the shim points to).")
        if not os.path.isfile(binary):
            raise SubAgentConfigError(f"'{binary}' is not a file on this machine.")
        cleaned_path: list[str] = []
        for entry in path_prepend or []:
            entry = str(entry).strip()
            if not entry:
                continue
            if not os.path.isabs(entry):
                raise SubAgentConfigError(
                    f"PATH entry '{entry}' is not an absolute directory.")
            if entry not in cleaned_path:
                cleaned_path.append(entry)
        cleaned_env: dict[str, str] = {}
        for name, value in (env or {}).items():
            name = str(name).strip()
            if not _ENV_NAME.match(name):
                raise SubAgentConfigError(
                    f"'{name}' is not an environment variable name.")
            cleaned_env[name] = "" if value is None else str(value)
        return {"binary": binary, "path_prepend": cleaned_path, "env": cleaned_env}

    def set_override(self, slug: str, *, binary: str,
                     path_prepend: list[str] | None = None,
                     env: dict[str, str] | None = None,
                     allow_shim: bool = False) -> dict:
        """Validate and persist the user-chosen location for ``slug``,
        leaving its CLI params as they are."""
        cleaned = self.validate_override(binary, path_prepend, env,
                                         allow_shim=allow_shim)
        with self._lock:
            entry = dict(self._data.get(slug, {}))
            entry.update(cleaned)
            self._data[slug] = entry
            self._save()
        return dict(cleaned)

    def clear_override(self, slug: str) -> None:
        """Forget the user-chosen location; params stay."""
        with self._lock:
            entry = dict(self._data.get(slug, {}))
            if not any(k in entry for k in OVERRIDE_KEYS):
                return
            for k in OVERRIDE_KEYS:
                entry.pop(k, None)
            if entry:
                self._data[slug] = entry
            else:
                self._data.pop(slug, None)
            self._save()

    def set(self, slug: str, params: dict[str, str], *,
            extra_allowed: dict[str, list[str]] | None = None) -> dict[str, str]:
        """Validate and persist ``params`` for ``slug``.

        Only keys listed in :data:`SCHEMA` for the slug are accepted. Pass an
        empty dict to clear all overrides for the slug. ``extra_allowed``
        widens a whitelisted param with values the caller verified live (the
        claude-code account model list).

        Raises :class:`SubAgentConfigError` on any invalid key/value.
        """
        validated = self.validate(slug, params, extra_allowed=extra_allowed)
        with self._lock:
            # The location override lives beside the params; a params
            # write replaces only the params.
            kept = {k: v for k, v in self._data.get(slug, {}).items()
                    if k in OVERRIDE_KEYS}
            merged: dict[str, object] = {**kept, **validated}
            if merged:
                self._data[slug] = merged
            else:
                self._data.pop(slug, None)
            self._save()
        return dict(validated)

    @staticmethod
    def validate(slug: str, params: dict[str, str], *,
                 extra_allowed: dict[str, list[str]] | None = None
                 ) -> dict[str, str]:
        """Return a cleaned copy of ``params`` or raise SubAgentConfigError.

        - Unknown slug: rejects any non-empty params (empty is fine; no-op).
        - Unknown param: rejected.
        - Param with ``allowed`` whitelist: value must be in the list.
        - Param with ``allowed=None``: value must be a non-empty string.
        """
        if not isinstance(params, dict):
            raise SubAgentConfigError("params must be an object")
        slug_schema = SCHEMA.get(slug)
        if slug_schema is None:
            if not params:
                return {}
            raise SubAgentConfigError(
                f"sub-agent '{slug}' has no configurable parameters"
            )
        cleaned: dict[str, str] = {}
        for key, raw_value in params.items():
            if key not in slug_schema:
                raise SubAgentConfigError(
                    f"unknown parameter '{key}' for sub-agent '{slug}'"
                )
            value = "" if raw_value is None else str(raw_value).strip()
            if value == "":
                # Empty string clears the param; do not persist.
                continue
            schema = slug_schema[key]
            extra = list((extra_allowed or {}).get(key) or [])
            if (schema.allowed is not None and value not in schema.allowed
                    and value not in extra):
                allowed = extra + [v for v in schema.allowed if v not in extra]
                raise SubAgentConfigError(
                    f"invalid value for {slug}.{key}: '{value}'. "
                    f"Allowed: {allowed}"
                )
            cleaned[key] = value
        return cleaned

    def build_extra_args(self, slug: str,
                         overrides: dict[str, str] | None = None) -> list[str]:
        """Return extra argv tokens for the CLI invocation, from persisted config.

        Reads ``self.get(slug)`` and renders each param via its schema's
        ``flag_template``, splitting on whitespace so each token is a
        separate argv element. Params with no ``flag_template`` are delivered
        by another mechanism (a composition file) and emit nothing here.
        ``overrides`` (spec 149) replaces a param's value, never adds a
        second copy of its flag.
        """
        slug_schema = SCHEMA.get(slug, {})
        argv: list[str] = []
        for key, value in resolve_params(slug, self, overrides).items():
            schema = slug_schema.get(key)
            if schema is None or not schema.flag_template:
                continue
            argv.extend(schema.flag_template.format(value=value).split())
        return argv

    def build_extra_env(self, slug: str) -> dict[str, str]:
        """Environment variables for the params whose schema names one."""
        slug_schema = SCHEMA.get(slug, {})
        env: dict[str, str] = {}
        for key, value in resolve_params(slug, self).items():
            schema = slug_schema.get(key)
            if schema is not None and schema.env_var and value:
                env[schema.env_var] = str(value)
        return env

    def build_acp_config_options(self, slug: str,
                                 overrides: dict[str, str] | None = None,
                                 ) -> dict[str, str]:
        """ACP session config options for the params whose schema names one."""
        slug_schema = SCHEMA.get(slug, {})
        options: dict[str, str] = {}
        for key, value in resolve_params(slug, self, overrides).items():
            schema = slug_schema.get(key)
            if schema is not None and schema.acp_config_option and value:
                options[schema.acp_config_option] = str(value)
        return options

    def schema_for(self, slug: str) -> dict[str, dict[str, object]]:
        """Return the parameter schema for ``slug`` in JSON-friendly shape.

        Used by ``GET /api/v2/settings/sub-agents`` to populate UI dropdowns.
        """
        slug_schema = SCHEMA.get(slug, {})
        return {
            name: {
                "allowed": list(s.allowed) if s.allowed is not None else None,
                "default": s.default,
            }
            for name, s in slug_schema.items()
        }
