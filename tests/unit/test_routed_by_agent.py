# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keyless Workbench routing (2026-11-23): a task a signed-in agent routed
carries ``routed_by: "agent"`` (spec 180's "Orbital chose this project"
mark, a third value). Stamped on the session_start meta, read back by the
session loader and the disk index, accepted by the inject request. An
unknown value still reads as no mark, so a downgrade is harmless."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from agent_os.agent.session import Session
from agent_os.api.routes.agents_v2 import InjectRequest
from agent_os.daemon_v2.session_index import derive_disk_entry


def _session(tmp_path, sid="s1"):
    s = Session.new(sid, str(tmp_path))
    s.append({"role": "user", "content": "hello", "timestamp": "2026-11-23T08:00:00+00:00"})
    return s


def _start_meta(s) -> dict:
    with open(s._filepath, encoding="utf-8") as f:
        return next(json.loads(line) for line in f
                    if '"session_start"' in line)


@pytest.mark.parametrize("value", ["model", "agent", "fallback"])
def test_each_mark_round_trips(tmp_path, value):
    s = _session(tmp_path)
    s.set_routed_by(value)
    assert s.routed_by == value
    assert _start_meta(s)["routed_by"] == value
    loaded = Session.load(s._filepath)
    assert loaded.routed_by == value
    assert derive_disk_entry(s._filepath, s.session_uuid)["routed_by"] == value


def test_an_unknown_mark_reads_as_none(tmp_path):
    s = _session(tmp_path)
    s.set_routed_by("telepathy")
    assert s.routed_by is None


def test_inject_request_accepts_agent():
    assert InjectRequest(content="x", routed_by="agent").routed_by == "agent"
    with pytest.raises(ValidationError):
        InjectRequest(content="x", routed_by="telepathy")
