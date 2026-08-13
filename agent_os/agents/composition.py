# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-spawn composition rendering for harnesses configured by a file.

Some agents take their whole runtime configuration from a composition file
rather than argv (dsh, whose Cordis plugin list carries model, sandbox mode,
session store, and persona). Orbital renders one such file per spawn: unique
name, written under the workspace's sub-agent dir, deleted when the adapter
stops. Never one shared file — same-slug dispatch is unrestricted across
projects and sessions, so a shared config is a race by construction.

Rendering is programmatic (``safe_load`` → set fields → ``safe_dump``), never
string templating. The persona is the real rendered sub-agent prompt: arbitrary
prose that would otherwise be a YAML injection vector.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import yaml

logger = logging.getLogger(__name__)

# Rendered files are named ``cordis-<8 hex>.yml``. Both halves are load-bearing:
# the prefix is what ``gc_stale`` is willing to delete, and the random suffix is
# what makes concurrent renders for one handle collision-free.
_RENDERED_PREFIX = "cordis-"
_RENDERED_SUFFIX = ".yml"

# Plugin ids the renderer knows how to configure. These are composition-local
# ids (the ``id:`` key), not package names.
_SANDBOX_POLICY_ID = "sandbox-policy"
_ACP_AGENT_ID = "acp-agent"


class CompositionError(ValueError):
    """Raised when a composition template is missing or malformed."""


def render_composition(
    template_path: str,
    *,
    model: str | None,
    sandbox_mode: str | None,
    persona: str | None,
    persistence_root: str,
    out_dir: str,
) -> str:
    """Render one spawn's composition file and return its absolute path.

    ``persistence_root`` is absolutised: the shipped template's ``./.sessions``
    is resolved against the harness's cwd, which would scatter session stores
    outside the project.

    A ``None`` for model, sandbox_mode, or persona leaves the template's own
    value in place — the template ships working defaults precisely so it stays
    bootable when a caller has nothing to say.
    """
    blocks = _load_template(template_path)

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_id = block.get("id")
        if block_id == _SANDBOX_POLICY_ID:
            _set_if(_config_of(block), "mode", sandbox_mode)
        elif block_id == _ACP_AGENT_ID:
            config = _config_of(block)
            _set_if(config, "model", model)
            _set_if(config, "persona", persona)
            config["persistenceRoot"] = os.path.abspath(persistence_root)

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"{_RENDERED_PREFIX}{uuid.uuid4().hex[:8]}{_RENDERED_SUFFIX}"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            blocks, f, sort_keys=False, allow_unicode=True,
            default_flow_style=False, width=10**6,
        )
    return out_path


def unlink_rendered(path: str | None) -> None:
    """Delete a rendered composition. Never raises."""
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Failed to remove rendered composition %s", path,
                       exc_info=True)


def gc_stale(out_dir: str, max_age_days: int = 7) -> int:
    """Delete rendered compositions older than ``max_age_days``.

    Opportunistic cleanup for the paths that never reach ``unlink_rendered``
    (daemon crash, an unkillable adapter that keeps its slot). Returns the
    number removed; never raises.
    """
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0
    for name in names:
        if not (name.startswith(_RENDERED_PREFIX)
                and name.endswith(_RENDERED_SUFFIX)):
            continue
        path = os.path.join(out_dir, name)
        try:
            if os.path.getmtime(path) >= cutoff:
                continue
            os.unlink(path)
            removed += 1
        except OSError:
            continue
    return removed


def _load_template(template_path: str) -> list:
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            blocks = yaml.safe_load(f)
    except OSError as exc:
        raise CompositionError(
            f"Composition template not readable: {template_path} ({exc})"
        ) from exc
    except yaml.YAMLError as exc:
        raise CompositionError(
            f"Composition template is not valid YAML: {template_path} ({exc})"
        ) from exc
    if not isinstance(blocks, list):
        raise CompositionError(
            f"Composition template must be a YAML list of plugin blocks: "
            f"{template_path}"
        )
    return blocks


def _set_if(config: dict, key: str, value) -> None:
    """Set ``key`` unless the caller had nothing to say (``None``)."""
    if value is not None:
        config[key] = value


def _config_of(block: dict) -> dict:
    """The block's ``config`` mapping, created if absent or malformed."""
    config = block.get("config")
    if not isinstance(config, dict):
        config = {}
        block["config"] = config
    return config
