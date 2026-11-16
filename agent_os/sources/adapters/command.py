# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A check command an agent registered through the ``orbital-sources`` skill.

The daemon runs the command line through the platform shell on the
source's interval (never before the user turns the source on) and reads the
JSON it prints: a list of items, or ``{"items": [...]}``. Only ``id``,
``title``, ``url``, ``body``, ``subject``, ``sender`` and ``time`` are read;
anything else (``project``, ``agent``, ``cwd``, ``permissions`` …) is ignored,
so a source can never choose where its work goes or what it may do.

It runs with the user's own permissions, like a script the user started
(the user saw the exact command line before turning it on).
"""

from __future__ import annotations

import json
import os

from agent_os.sources.adapters import Adapter
from agent_os.sources.context import CommandTimeout
from agent_os.sources.model import Item, SourceFailure, Verified, VerifyError

TIMEOUT_S = 60
OUTPUT_CAP = 5_000_000


def parse_items(stdout: str) -> list[Item]:
    """The command's stdout → items. Raises ``ValueError`` when it is not
    the expected JSON."""
    text = (stdout or "").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list):
        raise ValueError("expected a JSON list of items")
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rid, title = row.get("id"), row.get("title")
        if isinstance(rid, (int, float)) and not isinstance(rid, bool):
            rid = str(rid)
        if not isinstance(rid, str) or not rid.strip() or len(rid) > 200:
            continue
        if not isinstance(title, str) or not title.strip():
            continue

        def _s(key):
            v = row.get(key)
            return v if isinstance(v, str) else None
        out.append(Item(id=rid.strip(), kind="item", title=title, subject=_s("subject") or rid.strip(),
                        body=_s("body") or "", url=_s("url"), sender=_s("sender"), at=_s("time")))
    return out


class CommandAdapter(Adapter):
    mode = "poll"
    interval_s = None  # per source: command.interval_s

    async def verify(self, ctx) -> Verified:
        # Registration never runs the command: it arrives switched off and
        # runs only once the user turns it on.
        if not ctx.command or not (ctx.command.get("run") or "").strip():
            raise VerifyError("invalid", "command", "command.run is required")
        cwd = ctx.command.get("cwd")
        if cwd and not os.path.isdir(os.path.expanduser(cwd)):
            raise VerifyError("not_found", "cwd", "the working folder does not exist")
        return Verified(label=None)

    async def poll(self, ctx) -> list:
        cmd = ctx.command or {}
        cwd = os.path.expanduser(cmd.get("cwd") or "~")
        if not os.path.isdir(cwd):
            cwd = os.path.expanduser("~")
        try:
            res = await ctx.run_shell(cmd.get("run") or "", timeout=TIMEOUT_S, cwd=cwd)
        except CommandTimeout as t:
            raise SourceFailure("command_timeout", {"seconds": int(t.seconds)})
        if res.code != 0:
            raise SourceFailure("command_exit", {"code": res.code})
        if len(res.stdout) > OUTPUT_CAP:
            raise SourceFailure("command_output", {})
        try:
            return parse_items(res.stdout)
        except ValueError:
            raise SourceFailure("command_output", {})


ADAPTER = CommandAdapter
